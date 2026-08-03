#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""桌宠对话 TTS：支持「（动作）台词」格式 + 标点分档停顿。

格式：对话文本可含动作描述（全角/半角括号），动作不读出、只停顿。
停顿按标点分级（参考口语停顿规律）：
  句号/问号/叹号(。？！) > 分号(；) > 逗号(，) > 顿号(、)

用法：
  E:/PROJECT/Airilife_voice/.venv/Scripts/python.exe tools/tts_dialogue.py --text "..." --gpt ... --sovits ... --out xxx.wav
  （也可不带 --text，用内置示例）
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# 停顿分级（秒）。一拍≈0.28s：。？！=2拍 0.55s；；=1.5拍 0.4s；，=1拍 0.28s；、=0.5拍 0.15s
SENT_END = "。？！；"
PAUSE_SEC = {
    "。": 0.55, "？": 0.55, "！": 0.55,
    "；": 0.40, "…": 0.45,
    "，": 0.28, "、": 0.15,
    "action": 0.55,  # 动作（...）停顿
}
EXAMPLE = "（眼睛亮晶晶地看着你手里的丝袜）嗯...今天想穿白色的！（接过丝袜坐在床边，抬起纤细的腿）哥哥帮我穿好不好？"


def parse_dialogue(text: str) -> list[tuple[str, str]]:
    """拆成 chunk 列表：(type, content)。type ∈ {'speech','action'}。"""
    chunks: list[tuple[str, str]] = []
    buf = ""
    i = 0
    while i < len(text):
        c = text[i]
        if c in "（(":  # 动作开始
            if buf.strip():
                chunks.extend(_split_sentences(buf))
                buf = ""
            j = text.find("）" if c == "（" else ")", i + 1)
            action = text[i : j + 1] if j != -1 else text[i :]
            chunks.append(("action", action, ""))
            i = j + 1 if j != -1 else len(text)
        else:
            buf += c
            i += 1
    if buf.strip():
        chunks.extend(_split_sentences(buf))
    return chunks


def _split_sentences(text: str) -> list[tuple[str, str, str]]:
    """按句末标点切成句子（动作已剔除），返回 (type, text, end_punct)。"""
    result: list[tuple[str, str, str]] = []
    buf = ""
    for c in text:
        buf += c
        if c in SENT_END:
            result.append(("speech", buf.strip(), c))
            buf = ""
    if buf.strip():
        result.append(("speech", buf.strip(), ""))
    return result


def synth_dialogue(tts, ref: str, ref_txt: str, chunks: list[tuple[str, str, str]], lang: str = "zh"):
    """逐 chunk 合成，插入标点/动作停顿，返回 (audio, sr)。

    chunk: ('action', '（...）', '') 或 ('speech', text, end_punct)
    """
    sr = 32000
    parts: list[np.ndarray] = []
    prev_punct = ""
    for typ, content, end_punct in chunks:
        if typ == "action":
            parts.append(np.zeros(int(PAUSE_SEC["action"] * sr), dtype=np.float32))
            prev_punct = ""
        else:  # speech
            audio = tts.infer(
                spk_audio_path=ref, prompt_audio_path=ref, prompt_audio_text=ref_txt,
                text=content, text_language=lang, prompt_language="zh",
            )
            sr = audio.samplerate
            if parts and prev_punct:  # 上句结束的标点停顿
                parts.append(np.zeros(int(PAUSE_SEC.get(prev_punct, 0.3) * sr), dtype=np.float32))
            parts.append(audio.audio_data)
            prev_punct = end_punct
    full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    return full, sr


def main() -> None:
    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default=EXAMPLE)
    ap.add_argument("--gpt", default=str(ROOT / "models" / "gpt_firefly_v3-e20.ckpt"))
    ap.add_argument("--sovits", default=str(ROOT / "models" / "sovits_firefly_v3_e40_cont.pth"))
    ap.add_argument("--ref", default=str(ROOT / "reference_audio" / "活泼.wav"))
    ap.add_argument("--ref-text", default="好啦！看上去真不错，你好上相呀。")
    ap.add_argument("--lang", default="zh", help="text_language: zh 或 auto")
    ap.add_argument("--out", default=str(ROOT / "output" / "v3_style" / "dialogue_demo.wav"))
    args = ap.parse_args()

    chunks = parse_dialogue(args.text)
    print("解析结果:")
    for typ, content, end_punct in chunks:
        tag = end_punct if end_punct else ""
        print(f"  [{typ}]{tag} {content!r}")

    from gsv_tts import TTS

    tts = TTS(models_dir=str(ROOT / "gsv_models"), use_bert=True)
    tts.load_gpt_model(args.gpt)
    tts.load_sovits_model(args.sovits)
    full, sr = synth_dialogue(tts, args.ref, args.ref_text, chunks, lang=args.lang)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), full, sr)
    print(f"合成完成: {len(full) / sr:.2f}s → {out}")


if __name__ == "__main__":
    main()
