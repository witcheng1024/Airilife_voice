#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Single-call multi-emotion CosyVoice3 test."""
import os
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).parent
LOCAL_CACHE = ROOT / ".cache"
LOCAL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(LOCAL_CACHE / "numba"))

sys.path.insert(0, str(ROOT / "CosyVoice"))
sys.path.insert(0, str(ROOT / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import torchaudio


MODEL_DIR = ROOT / "CosyVoice" / "pretrained_models" / "Fun-CosyVoice3-0.5B"
REF_AUDIO = ROOT / "reference_audio" / "xiaoxuan_neutral.wav"
OUT_DIR = ROOT / "runs" / "20260617_171049_single_call"
REF_TEXT = "You are a helpful assistant.<|endofprompt|>" + (
    ROOT / "reference_audio" / "ref_text.txt"
).read_text(encoding="utf-8").strip()

CLEAN_TEXT = (
    "哥哥……你终于回来了呀，"
    "我还以为、还以为你今天又要丢下我一个人了呢……"
    "不过没关系啦！"
    "只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~"
)

FINE_TEXT = (
    "哥哥……你终于回来了呀，"
    "[breath]我还以为、还以为你今天又要丢下我一个人了呢……"
    "不过<strong>没关系啦</strong>！"
    "[laughter]只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~"
)

INSTRUCT = "请用先惊喜撒娇、再委屈带哭腔、然后强撑甜笑、最后调皮甜软的语气说这句话。"


def log(msg: str) -> None:
    print(msg, flush=True)


def save_outputs(cosyvoice, text: str, out_name: str, mode: str) -> dict:
    instruct_text = "You are a helpful assistant. " + INSTRUCT + "<|endofprompt|>"
    log(f"\n=== {out_name} ===")
    log(f"mode: {mode}")
    log(f"text: {text}")
    start = time.time()
    chunks = []
    if mode == "zero_shot":
        generator = cosyvoice.inference_zero_shot(text, REF_TEXT, str(REF_AUDIO), stream=False)
    elif mode == "instruct2":
        generator = cosyvoice.inference_instruct2(text, instruct_text, str(REF_AUDIO), stream=False)
    else:
        raise ValueError(f"unknown mode: {mode}")
    for item in generator:
        chunks.append(item["tts_speech"])
    audio = torch.concat(chunks, dim=1)
    elapsed = time.time() - start
    audio_s = audio.shape[1] / cosyvoice.sample_rate
    rtf = elapsed / audio_s if audio_s else 0
    out_path = OUT_DIR / f"{out_name}.wav"
    torchaudio.save(str(out_path), audio.cpu(), cosyvoice.sample_rate)
    log(f"saved: {out_path}")
    log(f"audio={audio_s:.2f}s synth={elapsed:.2f}s rtf={rtf:.2f}")
    return {
        "file": out_path.name,
        "mode": mode,
        "audio_s": round(audio_s, 3),
        "synth_s": round(elapsed, 3),
        "rtf": round(rtf, 3),
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    log(f"out_dir: {OUT_DIR}")
    log(f"model: {MODEL_DIR}")
    log(f"ref: {REF_AUDIO}")

    from cosyvoice.cli.cosyvoice import AutoModel

    load_start = time.time()
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR))
    load_s = time.time() - load_start
    log(f"model loaded: {load_s:.2f}s sample_rate={cosyvoice.sample_rate}")

    results = {
        "model_load_s": round(load_s, 3),
        "instruct": INSTRUCT,
        "clean_text": CLEAN_TEXT,
        "fine_text": FINE_TEXT,
        "plain": save_outputs(cosyvoice, CLEAN_TEXT, "single_plain", "zero_shot"),
        "clean": save_outputs(cosyvoice, CLEAN_TEXT, "single_clean", "instruct2"),
        "fine": save_outputs(cosyvoice, FINE_TEXT, "single_fine", "instruct2"),
    }

    import json

    (OUT_DIR / "manifest.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"\nmanifest: {OUT_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
