"""单参考 + 情绪指令试听（AiriLife M53-07，用户 2026-10-21 定：0.6B、bedtime 作唯一参考、情绪靠指令）。

faster-qwen3-tts 的 Base 克隆接口接受 instruct（作为 user 轮前置；ICL 模式下可用，
x-vector 模式不可靠）。官方 Base 未专门训练指令，是否听从只能人耳判断。
本脚本按情绪出「无指令 / 有指令」对照，各两遍，不打分。

用法（conda env qwen3-tts-fast，先停掉常驻服务腾显存）：
    python tools/qwen/instruct_emotion_grid.py
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "base" / "qwen3-tts" / "Qwen3-TTS-12Hz-0.6B-Base"
PROMPTS_FILE = ROOT / "voices" / "qwen3_voice_prompts.json"
REF_SCENE = "bedtime"
OUT_ROOT = ROOT / "output" / "instruct_grid"

TEMPERATURE = 0.9
TOP_K = 50
CHUNK_SIZE = 4
TAKES = 2

# 情绪键与 AiriLife 发送的标签一致；台词与情绪匹配，便于听指令是否起作用
EMOTIONS = [
    ("平淡", "用自然、平静的日常语气说。", "嗯，文件我已经帮你放到桌面上了，记得看一下。"),
    ("活泼", "用开心、轻快、充满活力的语气说。", "真的吗！那周末我们就去那家新开的甜品店吧！"),
    ("温柔", "用温柔、轻声、安抚人的语气说。", "没关系的，慢慢来，我会一直在这里陪着你。"),
    ("撒娇甜", "用撒娇、甜甜的、有点黏人的语气说。", "哼，你都好久没理我了，今天要多陪我一会儿嘛。"),
    ("害羞", "用害羞、小声、有点不好意思的语气说。", "那、那个……你一直这样看着我，我会不好意思的啦。"),
    ("悲伤", "用低落、难过、有点委屈的语气说，语速放慢。", "嗯……我知道你很忙。只是今天，有一点点想你了。"),
    ("惊讶", "用非常惊讶、意外的语气说。", "诶？你什么时候回来的？我怎么一点都没听见！"),
    ("生气", "用生气、带着责备的语气说，语气加重。", "你又熬到三点才睡！说了多少次了，身体是你自己的呀！"),
    ("紧张", "用紧张、着急、语速稍快的语气说。", "快、快看一下，电脑怎么突然这么烫，是不是要坏了？"),
]


# 第二轮（用户实听：活泼 / 悲伤 / 紧张 有无指令都不行）：每种三版写法对照
#   A 中文·声学特征（音调、语速、停顿）  B 英文·声学特征  C 中文·情境
VARIANTS = {
    "活泼": [
        ("A", "语气明亮上扬，语速偏快，音调比平时高一些，句尾轻快上扬，带着藏不住的笑意和兴奋。"),
        ("B", "Speak in a bright, bubbly, excited tone with a higher pitch, a faster pace and rising intonation, smiling while talking."),
        ("C", "像刚收到好消息一样兴奋地说，声音有活力、有跳跃感，不要轻声细语。"),
    ],
    "悲伤": [
        ("A", "声音低沉、轻而无力，语速很慢，句间有停顿，带一点哽咽和委屈，音调整体下沉。"),
        ("B", "Speak in a sad, quiet, low-energy voice with a slow pace and falling intonation, slightly choked up as if about to cry."),
        ("C", "像强忍着眼泪一样小声说，声音有点发颤，说完一句停一下。"),
    ],
    "紧张": [
        ("A", "语速明显加快，声音发紧、音调偏高，带一点慌乱和急促的呼吸，句子之间几乎不停顿。"),
        ("B", "Speak nervously and urgently, with a fast pace and a tense, slightly higher-pitched voice, a bit panicked and breathless."),
        ("C", "像突然遇到意外、很着急一样，又快又慌地说，可以有一点结巴。"),
    ],
}


def main() -> int:
    import numpy as np
    import soundfile as sf
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--variants", action="store_true", help="只跑 VARIANTS 里的情绪，每版指令一行")
    ap.add_argument("--ref", default=REF_SCENE, help="参考片场景 key（qwen_v2/<key>_t1.wav）")
    ap.add_argument("--emotions", nargs="+", default=None, help="只跑指定情绪")
    args = ap.parse_args()

    scene = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))["zihan_v2"]["scenes"][args.ref]
    ref_wav = ROOT / "models" / "zelia" / "qwen_v2" / f"{args.ref}_t1.wav"
    run_dir = OUT_ROOT / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    model = FasterQwen3TTS.from_pretrained(
        str(MODEL_DIR), device="cuda", dtype=torch.bfloat16, backend="torch", local_files_only=True
    )
    model.warmup()

    texts = {emotion: text for emotion, _i, text in EMOTIONS}
    if args.variants:
        plan = [(emotion, [("plain", None), *VARIANTS[emotion]]) for emotion in VARIANTS]
    else:
        plan = [(emotion, [("plain", None), ("instruct", instruct)]) for emotion, instruct, _t in EMOTIONS]
    if args.emotions:
        plan = [p for p in plan if p[0] in args.emotions]

    records = []
    for emotion, variants in plan:
        text = texts[emotion]
        for variant, inst in variants:
            for take in range(1, TAKES + 1):
                chunks, sr = [], 24000
                for audio, chunk_sr, _t in model.generate_voice_clone_streaming(
                    text=text, language="Chinese", ref_audio=str(ref_wav), ref_text=scene["text"],
                    chunk_size=CHUNK_SIZE, temperature=TEMPERATURE, top_k=TOP_K, instruct=inst,
                ):
                    sr = int(chunk_sr)
                    chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
                name = f"{emotion}_{variant}_{take}.wav"
                sf.write(str(run_dir / name), np.concatenate(chunks), sr)
                records.append({"emotion": emotion, "variant": variant, "take": take, "instruct": inst, "text": text, "file": name})
                print(name, flush=True)

    rows = []
    for emotion, variants in plan:
        rows.append(f"<tr><th colspan='2' style='text-align:left'>{emotion}：{html.escape(texts[emotion])}</th></tr>")
        for variant, inst in variants:
            audios = "".join(
                f"<audio controls preload='none' src='{html.escape(f'{emotion}_{variant}_{t}.wav')}'></audio>"
                for t in range(1, TAKES + 1)
            )
            label = "不带指令" if inst is None else f"{variant}：{html.escape(inst)}"
            rows.append(f"<tr><td style='max-width:420px'><small>{label}</small></td><td>{audios}</td></tr>")
    page = f"""<!doctype html><meta charset="utf-8"><title>情绪指令试听</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px;vertical-align:top;font-size:13px}}audio{{width:200px;display:block;margin:2px 0}}</style>
<h1>梓涵 v2：{args.ref} 参考 + 情绪指令（0.6B）</h1>
<p>参考原片：<audio controls src="../../../{ref_wav.relative_to(ROOT).as_posix()}"></audio></p>
<p>每行同一句台词：左边不带指令，右边带指令，各两遍。听两件事：①还是不是梓涵；②右边的情绪是否明显更贴。</p>
<table><tr><th>写法</th><th>两遍</th></tr>{''.join(rows)}</table>"""
    (run_dir / "index.html").write_text(page, encoding="utf-8")
    (run_dir / "manifest.json").write_text(json.dumps({
        "ref": ref_wav.relative_to(ROOT).as_posix(), "ref_text": scene["text"], "model": MODEL_DIR.name,
        "params": {"temperature": TEMPERATURE, "top_k": TOP_K, "chunk_size": CHUNK_SIZE, "mode": "icl-streaming"},
        "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n试听页：{run_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
