#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AiriLife v2 流式 TTS 服务（GSV-TTS-Lite，纯 PyTorch GPU 推理）

引擎：GSV-TTS-Lite 0.4.7。基于 GPT-SoVITS V2ProPlus 的 torch 推理（CUDA Graph / Nested KV Cache /
Continuous Batching / fp16），质量与官方 torch 一致 —— 取代 genie-tts ONNX 方案（有音色退化 + 尾字截断）。
Token 级流式：首块延迟实测 ~450ms（RTX 4070 Laptop，参考已缓存）。

用法：
  python tools/tts_server.py                 # 启动服务 0.0.0.0:9880
  python tools/tts_server.py --bench         # 测各情绪首字延迟(TTFT) + 存 wav 到 output/bench/
  python tools/tts_server.py --compare       # 生成与 torch 基准同文本音频到 output/compare/
  python tools/tts_server.py --bench --emotion 平淡 活泼   # 只处理指定情绪

API：
  POST /api/tts  {"text": "...", "emotion": "温柔", "speaker": "firefly"}
      -> 200, Content-Type: audio/L16; rate=32000; channels=1
         响应体 = 原始 int16 PCM 逐块流（token 级流式），首个 PCM 块到达即"首字延迟"
  GET  /health   引擎/模型/GPU/情绪列表
  GET  /demo     浏览器流式试听页（Web Audio 实时播放，验证流式）

环境：
  conda env `gsv`（Python 3.11）+ torch cu128 + gsv-tts-lite（见 requirements.txt）
  首次运行自动下载预训练模型（cnhubert/roberta/g2p/sv）到 gsv_models/
"""
from __future__ import annotations

import argparse
import asyncio
import html
import json
import logging
import os
import threading
import time
import wave
from contextlib import aclosing, closing
from pathlib import Path

import numpy as np
import torch
from anyio import CancelScope

if __package__:
    from .voice_registry import is_ready, load_speakers, public_speakers
else:
    from voice_registry import is_ready, load_speakers, public_speakers

ROOT = Path(os.environ.get('AIRILIFE_VOICE_RESOURCE_ROOT', Path(__file__).resolve().parent.parent)).resolve()
MODELS_DIR = ROOT / "gsv_models"
# 模型可选：默认 v2 中文；设环境变量 GSV_GPT/GSV_SOV 切 v3 多语言（如 firefly_v3）
GPT = Path(os.environ.get("GSV_GPT", ROOT / "models" / "gpt_firefly_678orig-e15.ckpt"))
SOV = Path(os.environ.get("GSV_SOV", ROOT / "models" / "sovits_firefly_678orig_e10.pth"))
REF_DIR = ROOT / "reference_audio"
SR = 32000
PORT = 9880

# 情绪 -> (参考音频文件名, 参考文本, 短句, 长句)。
# 参考文本必须与参考音频内容一致；短句/长句与 WSL torch 推理（infer_emotions.py）同文本，便于 A/B。
EMO = {
    "平淡":   ("平淡.wav", "我叫流萤，是鸢尾花家系的艺者。",
              "我在呢，有什么想聊的吗？", "今天也平安无事就好。对了，你有没有好好吃饭？别总是忙到忘记这些小事。"),
    "活泼":   ("活泼.wav", "好啦！看上去真不错，你好上相呀。",
              "太好啦！我们出去玩吧！", "欸嘿，今天天气这么好，不出去走走太可惜了！走嘛走嘛，我保证不会让你无聊的。"),
    "撒娇甜": ("撒娇甜.wav", "我最喜欢这家店的橡木蛋糕卷，每天都要吃一个。",
              "哥哥～陪我一会儿嘛，好不好嘛？", "你都不理我……人家好不容易才等到你回来的。再陪我待一会儿，就一小会儿，好不好嘛？"),
    "温柔":   ("温柔.wav", "这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。",
              "别担心啦，有我在呢，一切都会好好的。", "没关系的，慢慢来就好。累了就靠一下，我会一直在这里的。你已经做得很好了。"),
    "悲伤":   ("悲伤.wav", "嗯，是我不好。对不起。",
              "呜…为什么今天这么不顺心啊…", "有时候……会忽然觉得，有些话再也来不及说了。对不起，让你看到我这样。"),
    "惊讶":   ("惊讶.wav", "真的吗？看来…卡芙卡教给我的没错。",
              "哇！真的假的？！太让人意外啦！", "等一下，你刚才说什么？！这、这是真的吗？我完全没想到会是这样……"),
    "生气":   ("生气.wav", "绝对不行，难道你忘了吗？作为星核猎手，我有一项不惜一切代价的「零点任务」——",
              "不行，绝对不行！", "我不管你是谁，立刻给我住手！再这样下去，我可真要生气了！"),
    "紧张":   ("紧张.wav", "小心，更多怪物进来了！准备开火！",
              "小心！", "小心！前面好像有什么动静…我们得小声一点，别惊动它们。"),
}

# 基准文本：短句开头 + 多句，TTFT=第一句时间（真实首字延迟），后续句验证流式
BENCH_TEXT = "你好呀！今天过得怎么样？"

from gsv_tts import TTS  # noqa: E402
from gsv_tts.GPT_SoVITS.G2P import text_to_phonemes  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("airi_tts")

tts: "TTS | None" = None
_active_speaker = None
_loaded_models = None
_reference_signatures = {}
_loading_speaker = None
_speaker_lock = asyncio.Lock()


def _speakers():
    return load_speakers(ROOT, GPT, SOV, EMO)


def _load_speaker(speaker, reference):
    """Keep common models, one voice pair and references for that pair.

    GSV prompt semantics depend on SoVITS weights. Clear those caches when
    changing weights; the native prompt cache is only keyed by audio path.
    """
    global tts, _active_speaker, _loaded_models
    signature = (str(speaker['gpt']), speaker['gpt'].stat().st_mtime_ns,
                 str(speaker['sovits']), speaker['sovits'].stat().st_mtime_ns)
    if tts is None:
        tts = TTS(models_dir=str(MODELS_DIR), use_bert=True,
                  always_load_cnhubert=True, always_load_sv=True)
    engine = tts
    with engine._infer_lock:
        if _loaded_models != signature:
            _active_speaker = None
            _loaded_models = None
            engine.unload_gpt_model(*engine.get_gpt_list())
            engine.unload_sovits_model(*engine.get_sovits_list())
            engine.del_spk_audio(*engine.get_spk_audio_list())
            engine.del_prompt_audio(*engine.get_prompt_audio_list())
            _reference_signatures.clear()
            engine.load_gpt_model(str(speaker['gpt']))
            engine.load_sovits_model(str(speaker['sovits']))
            _loaded_models = signature
        audio = str(reference['audio'])
        ref_signature = (reference['audio'].stat().st_mtime_ns, reference['text'],
                         reference.get('language', 'zh'))
        if _reference_signatures.get(audio) != ref_signature:
            engine.del_spk_audio(audio)
            engine.del_prompt_audio(audio)
            engine.cache_spk_audio(audio, sovits_model=str(speaker['sovits']))
            engine.cache_prompt_audio(prompt_audio_paths=audio, prompt_audio_texts=reference['text'],
                                     prompt_language=reference.get('language', 'zh'))
            _reference_signatures[audio] = ref_signature
        _active_speaker = dict(speaker, signature=signature)
    logger.info('音色就绪: %s (%s + %s)', speaker['id'], speaker['gpt'].name, speaker['sovits'].name)


def _init() -> None:
    speaker = _speakers()['firefly']
    _load_speaker(speaker, speaker['references']['平淡'])
    # Chinese references already initialize Chinese G2P. English names in
    # ordinary Chinese replies otherwise initialize the English dictionaries
    # on the first audio request. Warm that frontend before reporting ready.
    started = time.perf_counter()
    text_to_phonemes('Hello.', 'en')
    logger.info('英文文本前端就绪: %.0f ms', (time.perf_counter() - started) * 1000)


async def _await_worker(future):
    """Cancellation waits for the native worker before model locks may release."""
    try:
        return await asyncio.shield(future)
    except asyncio.CancelledError:
        with CancelScope(shield=True):
            while not future.done():
                try:
                    await asyncio.shield(future)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            if not future.cancelled() and (error := future.exception()) is not None:
                logger.error('取消期间后台语音任务失败', exc_info=(type(error), error, error.__traceback__))
        raise


async def _engine_stream(engine, **arguments):
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    stopped = threading.Event()

    def worker():
        try:
            with engine._infer_lock:
                with closing(engine.infer_stream(**arguments)) as audio_stream:
                    for audio in audio_stream:
                        if stopped.is_set():
                            break
                        loop.call_soon_threadsafe(queue.put_nowait, audio)
        except BaseException as error:
            loop.call_soon_threadsafe(queue.put_nowait, error)
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    future = loop.run_in_executor(None, worker)
    try:
        while True:
            audio = await queue.get()
            if audio is None:
                break
            if isinstance(audio, BaseException):
                raise audio
            yield audio
    finally:
        stopped.set()
        with CancelScope(shield=True):
            await _await_worker(future)


def _to_pcm(audio) -> bytes:
    """AudioClip(float32) → int16 PCM bytes。"""
    return (audio.audio_data * 32767).astype(np.int16).tobytes()


def _write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(pcm)


_STREAM_KW = dict(
    text_language="auto", prompt_language="zh",  # auto：支持中/英/日/混合（v3 多语言）
    stream_mode="token", boost_first_chunk=True, debug=False,
)


async def _stream(text: str, ref: str, ref_txt: str, stream_mode: str = "token",
                  text_language: str | None = None, prompt_language: str | None = None,
                  *, gpt_model: str | None = None, sovits_model: str | None = None):
    """逐块推流原始 int16 PCM。infer_stream_async 内部已用 _infer_lock 串行化。

    token 模式：首字延迟最低（~120ms），供后端实时播放；
    sentence 模式：句子级块，听感更稳（无 boost 首字重复），供 demo/试听。
    """
    kw = dict(_STREAM_KW)
    if text_language is not None:
        kw['text_language'] = text_language
    if prompt_language is not None:
        kw['prompt_language'] = prompt_language
    kw["stream_mode"] = stream_mode
    if stream_mode == "sentence":
        kw["boost_first_chunk"] = False
    t0 = time.perf_counter()
    emitted = 0
    async with aclosing(_engine_stream(tts,
        spk_audio_path=ref, prompt_audio_path=ref, prompt_audio_text=ref_txt,
        text=text, gpt_model=gpt_model, sovits_model=sovits_model, **kw,
    )) as audio_stream:
        async for audio in audio_stream:
            if emitted == 0:
                logger.info(f"[TTFT] 首块 {(time.perf_counter() - t0) * 1000:.0f} ms")
            emitted += len(audio.audio_data)
            yield _to_pcm(audio)
    logger.info(f"[DONE] 共 {emitted / SR:.2f}s 音频")


# ==================== 离线：基准 / 对比 ====================

async def bench(emotions: list[str]) -> None:
    out_dir = ROOT / "output" / "bench"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"基准句: {BENCH_TEXT!r} → {out_dir}")
    print("情绪\tTTFT(ms)\t总(ms)\t音频(s)")
    for emo in emotions:
        fn, txt, _, _ = EMO[emo]
        ref = str(REF_DIR / fn)
        t0 = time.perf_counter()
        first_ms, pcm_chunks = None, []
        async for audio in tts.infer_stream_async(
            spk_audio_path=ref, prompt_audio_path=ref, prompt_audio_text=txt,
            text=BENCH_TEXT, **_STREAM_KW,
        ):
            if first_ms is None:
                first_ms = (time.perf_counter() - t0) * 1000
            pcm_chunks.append(_to_pcm(audio))
        total_ms = (time.perf_counter() - t0) * 1000
        audio_s = sum(len(c) // 2 for c in pcm_chunks) / SR
        _write_wav(out_dir / f"{emo}.wav", b"".join(pcm_chunks))
        print(f"{emo}\t{first_ms:.0f}\t{total_ms:.0f}\t{audio_s:.2f}")


async def compare(emotions: list[str]) -> None:
    """生成与 WSL torch 推理同文本音频（<情绪>_短句/长句.wav）到 output/compare/，便于 A/B。"""
    out_dir = ROOT / "output" / "compare"
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"对比输出 → {out_dir}（与 runs/Airi_678orig_v2ProPlus_v2/推理 同名文件 A/B）")
    for emo in emotions:
        fn, ref_txt, short, longt = EMO[emo]
        ref = str(REF_DIR / fn)
        for tag, text in (("短句", short), ("长句", longt)):
            pcm_chunks = []
            async for audio in tts.infer_stream_async(
                spk_audio_path=ref, prompt_audio_path=ref, prompt_audio_text=ref_txt,
                text=text, **_STREAM_KW,
            ):
                pcm_chunks.append(_to_pcm(audio))
            p = out_dir / f"{emo}_{tag}.wav"
            _write_wav(p, b"".join(pcm_chunks))
            logger.info(f"  {emo}_{tag} → {p}")


# ==================== 流式 HTTP 服务 ====================

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import HTMLResponse, StreamingResponse  # noqa: E402
from starlette.background import BackgroundTask
from pydantic import BaseModel  # noqa: E402

app = FastAPI(title="AiriLife v2 TTS (GSV)")


class _ClosingStreamingResponse(StreamingResponse):
    async def __call__(self, scope, receive, send):
        try:
            await super().__call__(scope, receive, send)
        finally:
            if self.background is not None:
                await self.background()


class TTSRequest(BaseModel):
    text: str
    emotion: str = "平淡"
    speaker: str = "firefly"
    stream_mode: str = "token"   # token(最低首字延迟) | sentence(句子级，听感更稳)


@app.on_event("startup")
async def _startup() -> None:
    await asyncio.get_running_loop().run_in_executor(None, _init)


@app.get("/health")
async def health() -> dict:
    speaker = _active_speaker or _speakers()['firefly']
    ready = tts is not None and _active_speaker is not None and _loading_speaker is None
    return {"ok": ready, "engine": "gsv-tts-lite",
            "model": f"{speaker['gpt'].stem} + {speaker['sovits'].stem}",
            "device": "gpu", "emotions": list(speaker['references']),
            "speaker": speaker['id'], "voices": public_speakers(_speakers()),
            "status": 'ready' if ready else 'loading' if _loading_speaker else 'unavailable',
            "loadingSpeaker": _loading_speaker,
            "residentVoiceModels": 0 if _loaded_models is None else 1,
            "gpuAllocatedMiB": round(torch.cuda.memory_allocated() / 1048576, 1),
            "gpuReservedMiB": round(torch.cuda.memory_reserved() / 1048576, 1)}


@app.get('/api/voices')
@app.get('/api/speakers')
async def voices() -> dict:
    return {'default': 'firefly', 'voices': public_speakers(_speakers())}


@app.post("/api/tts")
async def tts_endpoint(req: TTSRequest) -> StreamingResponse:
    global _loading_speaker
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="text 不能为空")
    speakers = _speakers()
    if req.speaker not in speakers:
        raise HTTPException(status_code=400, detail=f'未知音色: {req.speaker}')
    speaker = speakers[req.speaker]
    if not is_ready(speaker):
        raise HTTPException(status_code=503, detail=f'音色尚未就绪: {req.speaker}')
    if req.emotion not in speaker['references']:
        available = '/'.join(speaker['references'])
        raise HTTPException(status_code=400, detail=f'未知情绪: {req.emotion}，可选 {available}')
    if req.stream_mode not in ("token", "sentence"):
        raise HTTPException(status_code=400, detail="stream_mode 只能为 token 或 sentence")
    reference = speaker['references'][req.emotion]
    ref = str(reference['audio'])
    if not os.path.exists(ref):
        raise HTTPException(status_code=500, detail=f"缺参考音频: {ref}")
    await _speaker_lock.acquire()
    prepare_start = time.perf_counter()
    _loading_speaker = req.speaker
    try:
        future = asyncio.get_running_loop().run_in_executor(None, _load_speaker, speaker, reference)
        await _await_worker(future)
    except Exception as error:
        _speaker_lock.release()
        logger.exception('音色加载失败: %s', req.speaker)
        raise HTTPException(status_code=503, detail=f'音色加载失败: {req.speaker}') from error
    except BaseException:
        _speaker_lock.release()
        raise
    finally:
        _loading_speaker = None
    prepare_ms = (time.perf_counter() - prepare_start) * 1000
    released = False

    async def release():
        nonlocal released
        if not released:
            released = True
            _speaker_lock.release()

    async def stream():
        try:
            async with aclosing(_stream(req.text, ref, reference['text'], req.stream_mode,
                                        speaker.get('text_language', 'zh'), reference.get('language', 'zh'),
                                        gpt_model=str(speaker['gpt']), sovits_model=str(speaker['sovits']))) as chunks:
                async for chunk in chunks:
                    yield chunk
        finally:
            await release()

    response_stream = stream()

    async def cleanup():
        try:
            await response_stream.aclose()
        finally:
            await release()

    return _ClosingStreamingResponse(response_stream, media_type="audio/L16; rate=32000; channels=1",
                                     headers={'X-TTS-Speaker': speaker['id'], 'X-TTS-Prepare-Ms': f'{prepare_ms:.1f}'},
                                     background=BackgroundTask(cleanup))


_DEMO_HTML = """<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>AiriLife v2 流式试听</title>
<style>
body{background:#111;color:#eee;font-family:system-ui,sans-serif;max-width:680px;margin:40px auto;padding:0 20px}
textarea{width:100%;background:#1a1a1a;color:#eee;border:1px solid #333;border-radius:8px;padding:10px;font-size:15px;box-sizing:border-box}
select,button{padding:8px 14px;margin:8px 4px 0 0;border-radius:8px;background:#2a2a2a;color:#eee;border:1px solid #444;font-size:14px}
button{background:#2a7a4a;cursor:pointer}
#status{margin-top:14px;font-family:monospace;font-size:13px;white-space:pre-wrap;color:#9cf}
b{color:#7f7}
</style>
</head>
<body>
<h2>AiriLife v2 流式 TTS 试听 <small>(GSV-TTS-Lite / GPU)</small></h2>
<p>点「生成并播放」：浏览器实时接收流式 PCM 并播放。首块约 100-500ms 出声，之后连续播放 —— 耳朵验证流式。</p>
<textarea id="text" rows="2">你好呀！今天过得怎么样？我们出去走走吧！</textarea>
<label for="speaker">音色</label><select id="speaker">{SPEAKER_OPTIONS}</select>
<select id="emotion">{EMOTION_OPTIONS}</select>
<button id="btn">生成并播放</button>
<div id="status"></div>
<script>
const $ = s => document.querySelector(s);
const speakerInfos = {SPEAKER_METADATA};
$('#speaker').onchange = () => {
  const speaker = speakerInfos.find(item => item.id === $('#speaker').value);
  $('#emotion').replaceChildren();
  for (const emotion of speaker.emotions) {
    const option = document.createElement('option');
    option.value = emotion;
    option.textContent = emotion;
    $('#emotion').appendChild(option);
  }
};
let ctx = null;
const log = m => { $('#status').textContent += m + '\\n'; };

async function play() {
  $('#status').textContent = '';
  const text = $('#text').value.trim();
  const emotion = $('#emotion').value;
  if (!text) { log('请输入文本'); return; }
  const t0 = performance.now();
  ctx = ctx || new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 32000 });
  if (ctx.state === 'suspended') await ctx.resume();
  let nextTime = ctx.currentTime + 0.05;
  let firstMs = null, chunks = 0, totalSamples = 0;

  const resp = await fetch('/api/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, emotion, speaker: $('#speaker').value, stream_mode: 'sentence' }),
  });
  if (!resp.ok) { log('HTTP ' + resp.status + ' ' + await resp.text()); return; }

  const reader = resp.body.getReader();
  let buf = new Uint8Array(0);
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (!value) continue;
      const merged = new Uint8Array(buf.length + value.length);
      merged.set(buf); merged.set(value, buf.length); buf = merged;
      const n = buf.length - (buf.length % 2);
      if (n > 0) {
        const int16 = new Int16Array(buf.buffer, 0, n / 2);
        const f32 = new Float32Array(n / 2);
        for (let i = 0; i < f32.length; i++) f32[i] = int16[i] / 32768;
        const ab = ctx.createBuffer(1, f32.length, 32000);
        ab.copyToChannel(f32, 0);
        const src = ctx.createBufferSource();
        src.buffer = ab; src.connect(ctx.destination);
        // 钳制：start 时间不落过去，避免块重叠同时播放
        nextTime = Math.max(nextTime, ctx.currentTime + 0.03);
        src.start(nextTime);
        nextTime += ab.duration;
        chunks++; totalSamples += f32.length;
        if (firstMs === null) {
          firstMs = performance.now() - t0;
          log('首音频块: <b>' + firstMs.toFixed(0) + ' ms</b>  ← 首字延迟');
        }
        buf = buf.slice(n);
      }
    }
  } catch (e) {
    log('播放错误: ' + e.message);
  }
  log('块数: ' + chunks + '，音频总长: ' + (totalSamples / 32000).toFixed(2) + 's，端到端: ' + (performance.now() - t0).toFixed(0) + ' ms');
}

$('#btn').onclick = play;
</script>
</body>
</html>"""


@app.get("/demo", response_class=HTMLResponse)
async def demo_page() -> str:
    opts = "".join(f'<option value="{e}">{e}</option>' for e in EMO)
    speakers = public_speakers(_speakers())
    speaker_options = ''.join(
        '<option value="{}" {}>{}</option>'.format(html.escape(speaker['id'], quote=True),
            '' if speaker['ready'] else 'disabled', html.escape(speaker['label']))
        for speaker in speakers)
    metadata = json.dumps(speakers, ensure_ascii=False).replace('<', r'\u003c')
    return (_DEMO_HTML.replace("{EMOTION_OPTIONS}", opts)
            .replace('{SPEAKER_OPTIONS}', speaker_options).replace('{SPEAKER_METADATA}', metadata))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bench", action="store_true", help="离线测各情绪首字延迟")
    ap.add_argument("--compare", action="store_true", help="生成与 torch 基准同文本音频到 output/compare/")
    ap.add_argument("--emotion", nargs="*", help="bench/compare 时只处理指定情绪（默认全部）")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    emos = args.emotion or list(EMO)
    for e in emos:
        if e not in EMO:
            raise SystemExit(f"未知情绪: {e}")

    if args.compare or args.bench:
        _init()
        if args.compare:
            asyncio.run(compare(emos))
        else:
            asyncio.run(bench(emos))
        return

    import uvicorn  # noqa: E402
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
