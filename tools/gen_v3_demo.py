#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 多语言模型验证：用 GSV-TTS-Lite 加载 v3 权重，合成中/英/日/混合样例。

输出：output/v3_demo/{v3_zh,v3_en,v3_ja,v3_mix}.wav（用户醒来试听）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "gsv_models"
GPT = ROOT / "models" / "gpt_firefly_v3-e15.ckpt"
SOV = ROOT / "models" / "sovits_firefly_v3_e10.pth"
REF = ROOT / "reference_audio" / "活泼.wav"
REF_TXT = "好啦！看上去真不错，你好上相呀。"
OUT = ROOT / "output" / "v3_demo"

SAMPLES = [
    ("zh", "我叫流萤，是鸢尾花家系的艺者。", "v3_zh.wav"),
    ("en", "Hi! I'm Firefly, an artist from the Iris family. It's nice to meet you!", "v3_en.wav"),
    ("ja", "こんにちは！私はイリス家系の芸者、ホタルです。", "v3_ja.wav"),
    ("auto", "嗨！今天天气真好，Hi there! How's your day going? 要不要一起去散步？", "v3_mix.wav"),
]


def main() -> None:
    import soundfile as sf

    from gsv_tts import TTS

    print("加载 v3 模型...", flush=True)
    tts = TTS(models_dir=str(MODELS_DIR), use_bert=True)
    tts.load_gpt_model(str(GPT))
    tts.load_sovits_model(str(SOV))
    OUT.mkdir(parents=True, exist_ok=True)

    for lang, text, fname in SAMPLES:
        audio = tts.infer(
            spk_audio_path=str(REF), prompt_audio_path=str(REF), prompt_audio_text=REF_TXT,
            text=text, text_language=lang, prompt_language="zh",
        )
        p = OUT / fname
        sf.write(str(p), audio.audio_data, audio.samplerate)
        print(f"{fname}: {len(audio.audio_data)/audio.samplerate:.2f}s → {p}", flush=True)


if __name__ == "__main__":
    main()
