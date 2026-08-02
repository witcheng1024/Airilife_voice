#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 CosyVoice3(instruct2) 批量生成英/日克隆数据集（firefly 音色）。

引擎：CosyVoice3 Fun-CosyVoice3-0.5B（.venv 环境，transformers 4.51.3）。
方法：inference_instruct2 —— 已验证可跑（cross_lingual 在 Windows 上会卡死，弃用）。
支持 --watch：轮询 translations.json，翻译出多少就生成多少（与翻译并行）。

用法（必须用 .venv）:
  E:/PROJECT/Airilife_voice/.venv/Scripts/python.exe tools/gen_multilingual.py \
      --limit 6 --out output/multilingual_demo          # 先小批验证
  ... tools/gen_multilingual.py --watch --out data/firefly_multilingual   # 边翻译边生成（全量）

输入：output/multilingual/translations.json（含 zh/en/ja，由 prep_multilingual.py 增量写入）
输出：<out>/{en,ja}/<lang>_<n>.wav + <out>/manifest.json（已生成的跳过，可断点续传）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
COSYVOICE_ROOT = Path(r"E:/PROJECT/Airilife_voice/CosyVoice")
MODEL_DIR = Path(r"E:/PROJECT/Airilife_voice/pretrained_models/Fun-CosyVoice3-0.5B")
REF_AUDIO = ROOT / "reference_audio" / "活泼.wav"
DEFAULT_SRC = ROOT / "output" / "multilingual" / "translations.json"
DEFAULT_OUT = ROOT / "data" / "firefly_multilingual"

CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

INSTRUCT = {
    "en": "You are a helpful assistant. Please speak in English.<|endofprompt|>",
    "ja": "You are a helpful assistant. 请用日语说这句话。<|endofprompt|>",
}


def peak_normalize(audio, peak: float = 0.95):
    max_abs = torch.max(torch.abs(audio))
    if max_abs > peak:
        audio = audio * (peak / max_abs)
    return audio


def synth(cv, lang: str, text: str):
    chunks = []
    gen = cv.inference_instruct2(
        tts_text=text, instruct_text=INSTRUCT[lang], prompt_wav=str(REF_AUDIO), stream=False,
    )
    for out in gen:
        torch.cuda.synchronize()
        chunks.append(out["tts_speech"])
    return peak_normalize(torch.cat(chunks, dim=1))


def load_cv():
    from cosyvoice.cli.cosyvoice import AutoModel

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    print("加载 CosyVoice3...", flush=True)
    t0 = time.time()
    cv = AutoModel(model_dir=str(MODEL_DIR), fp16=True)
    print(f"模型加载 {time.time() - t0:.0f}s, sr={cv.sample_rate}", flush=True)
    return cv


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(DEFAULT_SRC))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--limit", type=int, default=0, help="只生成前 N 条（验证用）")
    ap.add_argument("--langs", nargs="*", default=["en", "ja"])
    ap.add_argument("--watch", action="store_true", help="轮询 src，增量生成（与翻译并行）")
    ap.add_argument("--watch-sleep", type=int, default=30)
    args = ap.parse_args()

    src_path = Path(args.src)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    def process_batch(cv, items, sr):
        """生成 items 里还没生成的（按文件存在判断）。返回本轮生成数。"""
        done = 0
        for i, item in enumerate(items):
            for lang in args.langs:
                text = item.get(lang)
                if not text:
                    continue
                lang_dir = out_root / lang
                lang_dir.mkdir(parents=True, exist_ok=True)
                out_path = lang_dir / f"{lang}_{i + 1}.wav"
                if out_path.exists():
                    continue
                t1 = time.time()
                try:
                    audio = synth(cv, lang, text)
                except Exception as e:
                    print(f"[{i}] {lang} 失败 {type(e).__name__}: {e}", flush=True)
                    continue
                dur = audio.shape[1] / sr
                peak = float(torch.max(torch.abs(audio)).item())
                if dur < 0.5 or peak < 0.01 or peak >= 1.0:
                    print(f"[{i}] {lang} BAD(dur={dur:.2f}s peak={peak:.2f}) 跳过", flush=True)
                    continue
                sf.write(str(out_path), audio.cpu().numpy().reshape(-1), sr)
                done += 1
                print(f"[{i}] {lang}: {dur:.2f}s ({time.time() - t1:.1f}s) → {out_path.name}", flush=True)
        return done

    cv = load_cv()
    sr = cv.sample_rate

    total_done = 0
    idle = 0
    while True:
        if not src_path.exists():
            print("translations.json 不存在，等待...", flush=True)
        else:
            items = json.loads(src_path.read_text(encoding="utf-8"))
            if args.limit:
                items = items[: args.limit]
            n = process_batch(cv, items, sr)
            total_done += n
            if n == 0:
                idle += 1
            else:
                idle = 0
        if not args.watch:
            break
        if idle >= 12:  # 连续 ~6 分钟无新数据，视为完成
            print("连续多轮无新数据，结束", flush=True)
            break
        time.sleep(args.watch_sleep)

    print(f"完成：本轮生成 {total_done} 条 → {out_root}", flush=True)


if __name__ == "__main__":
    main()
