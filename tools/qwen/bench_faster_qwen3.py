"""Faster Qwen3-TTS（faster-qwen3-tts, CUDA Graph 加速）在本机的基准测试。

三个指标，和 AiriLife 的关心点对齐：
  TTFA  首个可播放音频块的时间（streaming 模式下第一个 yield）
  RTF   （整段生成耗时 / 音频时长）> 1 表示比实时快 …… 反之亦然，见下方打印说明
  显存  torch allocated/reserved + 整卡 used（两者口径不同，分开记）

用法（conda env qwen3-tts-fast）：
    python tools/qwen/bench_faster_qwen3.py --model 1.7b-base --chunks 8 4 2 1
    python tools/qwen/bench_faster_qwen3.py --model 0.6b-base
    python tools/qwen/bench_faster_qwen3.py --model 1.7b-design --scenes daily bedtime

注意：CUDA Graph 首次捕获极慢，脚本会先 warmup，再 discard 第一轮，只报稳定态。
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QWEN_DIR = ROOT / "models" / "base" / "qwen3-tts"
PROMPTS_FILE = ROOT / "voices" / "qwen3_voice_prompts.json"

MODELS = {
    "1.7b-base": QWEN_DIR / "Qwen3-TTS-12Hz-1.7B-Base",
    "0.6b-base": QWEN_DIR / "Qwen3-TTS-12Hz-0.6B-Base",
    "1.7b-design": QWEN_DIR / "Qwen3-TTS-12Hz-1.7B-VoiceDesign",
}

# 与 AiriLife 的消息量级对齐：一条桌宠回复通常 20~60 字
TEXTS = {
    "short": "嗯？你回来啦。",
    "mid": "今天好像有点晚呢……我刚刚还在想，你是不是又忙着写代码，忘记时间了。",
    "long": "哥哥听好啦～今晚就用我的声音陪你入睡。晚安啦，我最喜欢的哥哥，梦里也要梦到我哦。明天见，好梦～",
}

# 参考音频：梓涵 prompt 已设计好的 wav；克隆时需要 ref_text
REF_DIR = ROOT / "models" / "zelia" / "qwen_v2"


def vram(torch) -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "alloc_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "device_used_mb": round((total - free) / 1024**2, 1),
        "device_total_mb": round(total / 1024**2, 1),
    }


def load_prompts() -> dict:
    with PROMPTS_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="1.7b-base", choices=sorted(MODELS))
    ap.add_argument("--chunks", nargs="+", type=int, default=[8], help="streaming chunk_size，可给多个")
    ap.add_argument("--texts", nargs="+", default=["short", "mid", "long"], choices=sorted(TEXTS))
    ap.add_argument("--reps", type=int, default=2, help="每个组合跑几次，取最好值")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--ref", default=str(REF_DIR / "daily_t1.wav"))
    ap.add_argument("--ref-text", default=None, help="参考音频文本，默认取 voices/qwen3_voice_prompts.json 的 daily 台词")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    import numpy as np
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    model_dir = MODELS[args.model]
    if not model_dir.exists():
        raise SystemExit(f"模型不存在：{model_dir}")

    prompts = load_prompts()
    ref_text = args.ref_text or prompts["zihan"]["scenes"]["daily"]["text"]
    instruct = prompts["zihan"]["instruct"]

    print(f"== 模型 {args.model} ({model_dir.name}) dtype={args.dtype}", flush=True)
    print(f"== 基线显存 {vram(torch)}", flush=True)

    t0 = time.perf_counter()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    model = FasterQwen3TTS.from_pretrained(
        str(model_dir), device="cuda", dtype=dtype, backend="torch", local_files_only=True
    )
    load_s = time.perf_counter() - t0
    print(f"== 加载 {load_s:.1f}s，显存 {vram(torch)}", flush=True)

    # warmup：CUDA Graph 捕获发生在这里，不计入结果
    t0 = time.perf_counter()
    model.warmup()
    warm_s = time.perf_counter() - t0
    print(f"== warmup {warm_s:.1f}s（含 CUDA Graph 捕获），显存 {vram(torch)}", flush=True)

    rows = []
    is_design = args.model.endswith("design")
    for ck in args.chunks:
        for key in args.texts:
            text = TEXTS[key]
            kwargs = dict(
                text=text,
                language="Chinese",
                chunk_size=ck,
                temperature=0.9,
                top_k=50,
            )
            best = None
            for rep in range(args.reps):
                # 第一轮预热该 chunk_size 的 CUDA Graph
                t_start = time.perf_counter()
                first = None
                audio_parts = []
                if is_design:
                    gen = model.generate_voice_design_streaming(instruct=instruct, **kwargs)
                else:
                    gen = model.generate_voice_clone_streaming(
                        ref_audio=args.ref, ref_text=ref_text, **kwargs
                    )
                for wav, sr, _meta in gen:
                    if first is None:
                        first = time.perf_counter() - t_start
                    audio_parts.append(wav)
                total = time.perf_counter() - t_start
                audio_sec = sum(len(p) for p in audio_parts) / sr
                row = {
                    "model": args.model,
                    "chunk_size": ck,
                    "text": key,
                    "chars": len(text),
                    "ttfa_ms": round(first * 1000, 1) if first else None,
                    "total_s": round(total, 3),
                    "audio_sec": round(audio_sec, 3),
                    "rtf": round(total / max(audio_sec, 1e-6), 3),
                    "x_realtime": round(audio_sec / max(total, 1e-6), 2),
                    "vram": vram(torch),
                    "sample_rate": sr,
                }
                if rep > 0:
                    if best is None or (row["ttfa_ms"] or 1e9) < (best["ttfa_ms"] or 1e9):
                        best = row
                else:
                    best = best or row
            rows.append(best)
            print(
                f"  chunk={ck:2d} {key:5s} {best['chars']:3d}字  TTFA {best['ttfa_ms']:7.1f}ms  "
                f"音频 {best['audio_sec']:5.2f}s / 耗时 {best['total_s']:5.2f}s  "
                f"{best['x_realtime']:4.2f}x 实时  显存 {best['vram']['reserved_mb']}MB",
                flush=True,
            )

    summary = {
        "model": args.model,
        "dtype": args.dtype,
        "model_load_sec": round(load_s, 3),
        "warmup_sec": round(warm_s, 3),
        "vram_after_warmup": vram(torch),
        "rows": rows,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
    }
    out = Path(args.json_out) if args.json_out else ROOT / "output" / "bench" / f"faster_qwen3_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n结果 → {out}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    sys.exit(main())
