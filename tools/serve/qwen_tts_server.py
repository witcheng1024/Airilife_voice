#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AiriLife Qwen3-TTS 常驻流式服务（faster-qwen3-tts，Base 模型 ICL 克隆）

与 tools/tts_server.py（GSV）保持同一 HTTP 协议，AiriLife 无需新增 provider：
  POST /api/tts  {"text": "...", "emotion": "温柔", "speaker": "zihan", "stream_mode": "token"}
      -> 200, Content-Type: audio/L16; rate=24000; channels=1
         响应体 = 原始 int16 PCM 逐块流；首个 PCM 块到达即首包延迟
  GET  /health      引擎/模型/状态/音色（engine = "qwen3-tts"）
  GET  /api/voices  音色与情绪清单

音色清单见 voices/qwen3_voices.json：每个情绪是一条定稿母版 wav + 逐字台词。
不在请求路径调用 VoiceDesign —— 声线以已定稿的母版为准，重新设计会漂移。

用法（conda env qwen3-tts-fast）：
  python tools/serve/qwen_tts_server.py                       # 0.0.0.0:9880
  python tools/serve/qwen_tts_server.py --chunk-size 8        # 吞吐更好，首包更慢
  python tools/serve/qwen_tts_server.py --bench               # 不起服务，测各音色首包与实时倍率
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
from pydantic import BaseModel

ROOT = Path(os.environ.get('AIRILIFE_VOICE_RESOURCE_ROOT', Path(__file__).resolve().parents[2])).resolve()
MODELS = {
    '0.6b': ROOT / 'models' / 'base' / 'qwen3-tts' / 'Qwen3-TTS-12Hz-0.6B-Base',
    '1.7b': ROOT / 'models' / 'base' / 'qwen3-tts' / 'Qwen3-TTS-12Hz-1.7B-Base',
}
ENGINE = 'qwen3-tts'
PORT = 9880
# 采样参数沿用 DSH 基准（bench_faster_qwen3.py）
TEMPERATURE = 0.9
TOP_K = 50
WARM_TEXT = '嗯。'
NEUTRAL_LABELS = ('普通', '平淡')

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logger = logging.getLogger('qwen_tts')


def _resource(value: str) -> Path:
    path = (ROOT / value).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError(f'音色资源不在仓库内: {value}')
    return path


def load_voices(path: Path) -> tuple[str, dict]:
    config = json.loads(path.read_text(encoding='utf-8'))
    voices = {}
    for voice_id, voice in config['voices'].items():
        references = {
            emotion: {'audio': _resource(item['audio']), 'text': item['text']}
            for emotion, item in voice['references'].items()
        }
        voices[voice_id] = {
            'id': voice_id,
            'label': voice.get('label', voice_id),
            'language': voice.get('language', 'Chinese'),
            'references': references,
        }
    default = config.get('default') or next(iter(voices))
    return default, voices


def is_ready(voice: dict) -> bool:
    return all(item['audio'].is_file() for item in voice['references'].values())


def public_voices(voices: dict) -> list[dict]:
    return [{'id': v['id'], 'label': v['label'], 'ready': is_ready(v), 'emotions': list(v['references'])}
            for v in voices.values()]


def to_pcm16(audio) -> bytes:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype('<i2').tobytes()


class Engine:
    """单 GPU 串行推理；生成在工作线程，按块回传事件循环。"""

    def __init__(self, model_key: str, chunk_size: int, voices_path: Path):
        self.model_key = model_key
        self.model_dir = MODELS[model_key]
        self.chunk_size = chunk_size
        self.default_voice, self.voices = load_voices(voices_path)
        self.model = None
        self.sample_rate = 24000
        self.status = 'loading'
        self.loading_voice: str | None = self.default_voice
        self.error: str | None = None
        self.lock = asyncio.Lock()

    def load(self) -> None:
        import torch
        from faster_qwen3_tts import FasterQwen3TTS

        try:
            started = time.perf_counter()
            # dtype 必须传 torch 对象，传 'bf16' 字符串会炸（DSH 踩坑记录）
            self.model = FasterQwen3TTS.from_pretrained(
                str(self.model_dir), device='cuda', dtype=torch.bfloat16, backend='torch', local_files_only=True,
            )
            self.sample_rate = int(self.model.sample_rate)
            self.model.warmup()
            # 每条参考都跑一次：建立 ICL 参考特征缓存并预热本 chunk_size 的 CUDA Graph，首个真实请求不再付这笔
            for voice in self.voices.values():
                if not is_ready(voice):
                    logger.warning('音色 %s 缺参考音频，跳过预热', voice['id'])
                    continue
                self.loading_voice = voice['id']
                for reference in voice['references'].values():
                    for _ in self._generate(voice, reference, WARM_TEXT, threading.Event()):
                        pass
            self.status = 'ready'
            logger.info('Qwen3-TTS 就绪: model=%s chunk=%d sr=%d 用时 %.1fs', self.model_dir.name,
                        self.chunk_size, self.sample_rate, time.perf_counter() - started)
        except Exception as error:  # 加载失败要能从 /health 看到，而不是进程静默
            self.status = 'unavailable'
            self.error = str(error)
            logger.exception('Qwen3-TTS 加载失败')
        finally:
            self.loading_voice = None

    def _generate(self, voice: dict, reference: dict, text: str, cancelled: threading.Event):
        stream = self.model.generate_voice_clone_streaming(
            text=text, language=voice['language'], ref_audio=str(reference['audio']), ref_text=reference['text'],
            chunk_size=self.chunk_size, temperature=TEMPERATURE, top_k=TOP_K,
        )
        try:
            for audio, _sr, _timing in stream:
                if cancelled.is_set():
                    break
                yield to_pcm16(audio)
        finally:
            stream.close()

    async def stream(self, voice: dict, reference: dict, text: str):
        """在工作线程跑生成器；客户端断开时置取消标志，线程在下一块边界退出并释放 GPU。"""
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        cancelled = threading.Event()
        done = object()

        def worker():
            try:
                for chunk in self._generate(voice, reference, text, cancelled):
                    loop.call_soon_threadsafe(queue.put_nowait, chunk)
            except Exception as error:
                loop.call_soon_threadsafe(queue.put_nowait, error)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, done)

        thread = threading.Thread(target=worker, name='qwen-tts-gen', daemon=True)
        started = time.perf_counter()
        thread.start()
        first = True
        try:
            while True:
                item = await queue.get()
                if item is done:
                    break
                if isinstance(item, Exception):
                    raise item
                if first:
                    first = False
                    logger.info('首包 %.0fms speaker=%s chars=%d', (time.perf_counter() - started) * 1000,
                                voice['id'], len(text))
                yield item
        finally:
            cancelled.set()
            # 等线程退出再放锁，避免下一个请求与残留生成并发占用 GPU
            await asyncio.to_thread(thread.join)
            logger.info('合成结束 %.0fms speaker=%s', (time.perf_counter() - started) * 1000, voice['id'])


engine: Engine | None = None


# 必须在模块顶层：本文件启用了 postponed annotations，函数内定义的模型 FastAPI 解析不到，会被当成查询参数
class TTSRequest(BaseModel):
    text: str
    emotion: str = '平淡'
    speaker: str | None = None
    stream_mode: str = 'token'  # 与 GSV 协议兼容；Qwen 始终按块流式


def create_app():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import StreamingResponse

    app = FastAPI(title='AiriLife Qwen3-TTS')

    @app.on_event('startup')
    async def startup() -> None:
        # 后台加载：/health 立即可答 loading，Electron 据此判定"启动中"
        asyncio.get_running_loop().run_in_executor(None, engine.load)

    @app.get('/health')
    async def health() -> dict:
        import torch
        voice = engine.voices[engine.default_voice]
        return {
            'ok': engine.status == 'ready', 'engine': ENGINE, 'status': engine.status,
            'model': engine.model_dir.name, 'device': 'gpu', 'speaker': voice['id'],
            'emotions': list(voice['references']), 'voices': public_voices(engine.voices),
            'loadingSpeaker': engine.loading_voice, 'sampleRate': engine.sample_rate,
            'chunkSize': engine.chunk_size, 'error': engine.error,
            'gpuAllocatedMiB': round(torch.cuda.memory_allocated() / 1048576, 1),
            'gpuReservedMiB': round(torch.cuda.memory_reserved() / 1048576, 1),
        }

    @app.get('/api/voices')
    @app.get('/api/speakers')
    async def voices() -> dict:
        return {'default': engine.default_voice, 'voices': public_voices(engine.voices)}

    @app.post('/api/tts')
    async def tts(req: TTSRequest) -> StreamingResponse:
        if not req.text.strip():
            raise HTTPException(status_code=400, detail='text 不能为空')
        if engine.status != 'ready':
            raise HTTPException(status_code=503, detail=f'Qwen3-TTS 未就绪: {engine.status}')
        voice_id = req.speaker or engine.default_voice
        voice = engine.voices.get(voice_id)
        if voice is None:
            raise HTTPException(status_code=400, detail=f'未知音色: {voice_id}')
        if not is_ready(voice):
            raise HTTPException(status_code=503, detail=f'音色尚未就绪: {voice_id}')
        reference = voice['references'].get(req.emotion)
        if reference is None:
            # 调用方拿不到能力列表时会发旧标签；未知情绪回落中性（新标签「普通」/ 旧标签「平淡」）
            fallback = next((label for label in NEUTRAL_LABELS if label in voice['references']), None)
            if fallback is not None:
                logger.info('音色 %s 无情绪 %s，本次使用 %s', voice['id'], req.emotion, fallback)
                reference = voice['references'][fallback]
        if reference is None:
            available = '/'.join(voice['references'])
            raise HTTPException(status_code=400, detail=f'未知情绪: {req.emotion}，可选 {available}')

        waited = time.perf_counter()
        await engine.lock.acquire()
        wait_ms = (time.perf_counter() - waited) * 1000

        async def body():
            try:
                async for chunk in engine.stream(voice, reference, req.text):
                    yield chunk
            finally:
                engine.lock.release()

        return StreamingResponse(
            body(), media_type=f'audio/L16; rate={engine.sample_rate}; channels=1',
            headers={'X-TTS-Speaker': voice['id'], 'X-TTS-Prepare-Ms': f'{wait_ms:.1f}'},
        )

    return app


def bench(texts: list[str]) -> None:
    engine.load()
    if engine.status != 'ready':
        raise SystemExit(f'加载失败: {engine.error}')
    for voice in engine.voices.values():
        reference = voice['references']['平淡']
        for text in texts:
            started = time.perf_counter()
            first = None
            samples = 0
            for chunk in engine._generate(voice, reference, text, threading.Event()):
                first = first or time.perf_counter() - started
                samples += len(chunk) // 2
            total = time.perf_counter() - started
            audio = samples / engine.sample_rate
            print(f'{voice["id"]:8s} {len(text):3d}字 首包 {first * 1000:6.0f}ms  音频 {audio:5.2f}s / 耗时 '
                  f'{total:5.2f}s  {audio / total:4.2f}x 实时', flush=True)


def main() -> None:
    global engine
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='0.0.0.0')
    ap.add_argument('--port', type=int, default=PORT)
    ap.add_argument('--model', default='0.6b', choices=sorted(MODELS))
    # 0.6B 实测（RTX 4070 Laptop）：chunk 4 首包 245–313ms、1.2–1.9x 实时；8 为 390–480ms；再小会跟不上实时
    ap.add_argument('--chunk-size', type=int, default=4, help='越小首包越快、吞吐越差')
    ap.add_argument('--voices', default=str(ROOT / 'voices' / 'qwen3_voices.json'))
    ap.add_argument('--bench', action='store_true', help='不起服务，测首包与实时倍率')
    args = ap.parse_args()

    engine = Engine(args.model, args.chunk_size, Path(args.voices))
    if args.bench:
        bench(['嗯？你回来啦。', '今天好像有点晚呢……我刚刚还在想，你是不是又忙着写代码，忘记时间了。'])
        return
    import uvicorn
    uvicorn.run(create_app(), host=args.host, port=args.port, log_level='warning')


if __name__ == '__main__':
    main()
