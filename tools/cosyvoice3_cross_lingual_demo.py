#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CosyVoice3 跨语言音色克隆 demo。

用流萤(火萤/Firefly)的中文原声参考音频做音色提示(prompt_wav)，
通过 inference_cross_lingual 合成英语/日语文本，保持流萤音色。

用法:
    D:/miniforge3/envs/neuro/python.exe tools/cosyvoice3_cross_lingual_demo.py

输出:
    E:/PROJECT/Airilife_voice_v2/output/cosyvoice_demo/{en,ja}_*.wav
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------- env/paths
ROOT = Path(__file__).resolve().parent.parent          # E:/PROJECT/Airilife_voice_v2
COSYVOICE_ROOT = Path(r"E:/PROJECT/Airilife_voice/CosyVoice")
MODEL_DIR = Path(r"E:/PROJECT/Airilife_voice/pretrained_models/Fun-CosyVoice3-0.5B")
REF_AUDIO = ROOT / "reference_audio" / "活泼.wav"
OUT_DIR = ROOT / "output" / "cosyvoice_demo"

CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

# 活泼.wav 的参考中文文本（zero_shot 回退时使用）
PROMPT_TEXT = "好啦！看上去真不错，你好上相呀。"

# (语言, 文件名, 文本)
SENTENCES = [
    ("en", "en_1", "Hi! I'm Firefly, an artist from the Iris family. It's nice to meet you!"),
    ("en", "en_2", "Let's go for a walk together! The weather is so nice today."),
    ("en", "en_3", "My dream is to dance with everyone under the starry night sky."),
    ("ja", "ja_1", "こんにちは！私はイリス家系の芸者、ホタルです。"),
    ("ja", "ja_2", "一緒に散歩に行こうよ！今日はいい天気だね。"),
    ("ja", "ja_3", "夜の空の下で、みんなと一緒に踊りたいな。"),
]


def log(msg: str) -> None:
    print(msg, flush=True)


def peak_normalize(audio, peak: float = 0.95):
    import torch
    max_abs = torch.max(torch.abs(audio))
    if max_abs > peak:
        audio = audio * (peak / max_abs)
    return audio


def synth_one(cosyvoice, text: str, mode: str) -> tuple:
    """Run synthesis. mode: 'cross_lingual' | 'zero_shot'. Returns list of tensor chunks."""
    chunks = []
    if mode == "cross_lingual":
        gen = cosyvoice.inference_cross_lingual(text, str(REF_AUDIO), stream=False)
    else:
        gen = cosyvoice.inference_zero_shot(
            tts_text=text, prompt_text=PROMPT_TEXT, prompt_wav=str(REF_AUDIO), stream=False
        )
    for out in gen:
        chunks.append(out["tts_speech"])
    return chunks


def main() -> None:
    import soundfile as sf
    import torch

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    assert REF_AUDIO.exists(), f"reference audio missing: {REF_AUDIO}"

    log(f"CosyVoice root : {COSYVOICE_ROOT}")
    log(f"Model dir      : {MODEL_DIR}")
    log(f"Reference audio: {REF_AUDIO}")
    log(f"Output dir     : {OUT_DIR}")

    from cosyvoice.cli.cosyvoice import AutoModel

    t0 = time.time()
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR))
    sr = cosyvoice.sample_rate
    frontend = getattr(cosyvoice, "frontend", None)
    tf = getattr(frontend, "text_frontend", "n/a")
    log(f"Model loaded in {time.time() - t0:.1f}s | sample_rate={sr} | text_frontend={tf}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)} | VRAM used: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    manifest = {"model_dir": str(MODEL_DIR), "ref_audio": str(REF_AUDIO), "sample_rate": sr, "items": []}
    for lang, fname, text in SENTENCES:
        out_path = OUT_DIR / f"{fname}.wav"
        mode_used = "cross_lingual"
        try:
            chunks = synth_one(cosyvoice, text, "cross_lingual")
        except Exception as exc:
            log(f"  [!] cross_lingual failed ({exc}); falling back to zero_shot")
            mode_used = "zero_shot"
            chunks = synth_one(cosyvoice, text, "zero_shot")

        audio = torch.cat(chunks, dim=1)
        audio = peak_normalize(audio)
        dur = audio.shape[1] / sr
        peak = float(torch.max(torch.abs(audio)).item())
        rms = float(torch.sqrt(torch.mean(audio ** 2)).item())
        flag = "OK"
        if dur < 0.3 or peak < 0.005:
            flag = "BAD-SILENT"
        elif peak >= 1.0:
            flag = "BAD-CLIP"
        log(f"[{lang}] {fname}: {dur:.2f}s peak={peak:.3f} rms={rms:.4f} mode={mode_used} {flag}")

        if flag.startswith("BAD"):
            # 不提交坏样本：跳过落盘（或替换为备用句）
            manifest["items"].append({"file": fname, "lang": lang, "text": text, "mode": mode_used,
                                      "dur_s": round(dur, 3), "peak": round(peak, 4), "status": flag})
            log(f"  !! {flag} -> skip saving {out_path.name}")
            continue

        sf.write(str(out_path), audio.cpu().numpy().reshape(-1), sr)
        manifest["items"].append({"file": fname, "lang": lang, "text": text, "mode": mode_used,
                                  "dur_s": round(dur, 3), "peak": round(peak, 4), "rms": round(rms, 4),
                                  "status": "OK", "path": str(out_path)})
        log(f"  -> saved {out_path}")

    import json
    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Manifest: {manifest_path}")
    log("DONE")


if __name__ == "__main__":
    main()
