#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""桌宠对话 TTS：支持「（动作）台词」格式 + 标点分档停顿 + 长文本拆分控首字延迟。

- 动作 `（...）`：不读出，只停顿（可多处）
- 停顿按标点分级（。？！0.55s > ；0.40s > ，0.28s > 、0.15s，动作 0.55s）
- 分句合成绕开 s1 提前 EOS 截断
- 含日文假名的分句自动用 auto 语言（否则 zh 不念日文）
- 超长句子（> max_chars）按逗号再拆，保证首字延迟低
- synth_dialogue 是生成器：逐 chunk 产出音频，服务端可流式播放

用法：
  E:/PROJECT/Airilife_voice/.venv/Scripts/python.exe tools/tts_dialogue.py --text "..." --out xxx.wav
"""
from __future__ import annotations

import argparse
import os
import re
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
    "action": 1.00,  # 动作停顿 > 句号，便于区分
}
# 含外文(日文假名/英文)的段用 auto：zh 模式遇英文词会慢 3-4 倍、不念日文
FOREIGN_RE = re.compile(r"[ぁ-んァ-ヶ]|[a-zA-Z]")
MAX_CHARS = 30  # 超长句按逗号再拆（控首字延迟）；不拆顿号，避免把日文词拆散

EXAMPLE = (
    "（轻轻拉了拉你的衣角）哥哥，今天想不想试试这个 Python 的新玩法？"
    "……嗯，我觉得超有趣哦！（凑到你耳边小声说）不过也别太累；"
    "累了就歇歇、喝口水吧——有我在呢。ごめんね、刚才是不是吓到你啦？"
)


def parse_dialogue(text: str, max_chars: int = MAX_CHARS) -> list[tuple[str, str, str]]:
    """拆成 chunk 列表：(type, content, end_punct)。type ∈ {'speech','action'}。"""
    chunks: list[tuple[str, str, str]] = []
    buf = ""
    i = 0
    while i < len(text):
        c = text[i]
        if c in "（(":  # 动作开始
            if buf.strip():
                chunks.extend(_split_sentences(buf, max_chars))
                buf = ""
            j = text.find("）" if c == "（" else ")", i + 1)
            action = text[i : j + 1] if j != -1 else text[i :]
            chunks.append(("action", action, ""))
            i = j + 1 if j != -1 else len(text)
        else:
            buf += c
            i += 1
    if buf.strip():
        chunks.extend(_split_sentences(buf, max_chars))
    return chunks


def _split_sentences(text: str, max_chars: int) -> list[tuple[str, str, str]]:
    """按句末标点切句；超长句再按逗号/顿号拆成子句（控首字延迟）。"""
    result: list[tuple[str, str, str]] = []
    buf = ""
    for c in text:
        buf += c
        if c in SENT_END:
            result.extend(_subsplit_long(buf.strip(), max_chars, c))
            buf = ""
    if buf.strip():
        result.extend(_subsplit_long(buf.strip(), max_chars, ""))
    return result


def _subsplit_long(sentence: str, max_chars: int, end_punct: str) -> list[tuple[str, str, str]]:
    """超长句按逗号拆子句（控首字延迟）。只拆 ，不拆 、，避免把日文词/并列拆散。"""
    eff = sum(2 if ord(ch) > 127 else 1 for ch in sentence)  # 汉字算2
    if eff <= max_chars:
        return [("speech", sentence, end_punct)]
    parts = []
    buf = ""
    for c in sentence:
        buf += c
        if c == "，" and sum(2 if ord(x) > 127 else 1 for x in buf) >= 10:
            parts.append(("speech", buf.strip(), c))
            buf = ""
    if buf.strip():
        parts.append(("speech", buf.strip(), end_punct))
    return parts or [("speech", sentence, end_punct)]


def main() -> None:
    import time

    import soundfile as sf

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--text", default=EXAMPLE)
    ap.add_argument("--gpt", default=str(ROOT / "models" / "gpt_firefly_v3-e20.ckpt"))
    ap.add_argument("--sovits", default=str(ROOT / "models" / "sovits_firefly_v3_e40_cont.pth"))
    ap.add_argument("--ref", default=str(ROOT / "reference_audio" / "活泼.wav"))
    ap.add_argument("--ref-text", default="好啦！看上去真不错，你好上相呀。")
    ap.add_argument("--lang", default="zh", help="text_language: zh 或 auto")
    ap.add_argument("--max-chars", type=int, default=MAX_CHARS)
    ap.add_argument("--out", default=str(ROOT / "output" / "v3_style" / "dialogue_demo.wav"))
    args = ap.parse_args()

    chunks = parse_dialogue(args.text, args.max_chars)
    print("解析结果:")
    for typ, content, end_punct in chunks:
        print(f"  [{typ}] {end_punct} {content!r}")

    from gsv_tts import TTS

    tts = TTS(models_dir=str(ROOT / "gsv_models"), use_bert=True)
    tts.load_gpt_model(args.gpt)
    tts.load_sovits_model(args.sovits)
    # 预热：模型刚加载首次推理含 CUDA 图编译，多预热几档（短/中/含英文）让 TTFT 反映稳态
    for wt in ("预热。", "今天天气不错，我们出去走走吧。", "哥哥，这个 Python 的新玩法真的很有趣哦！"):
        tts.infer(spk_audio_path=args.ref, prompt_audio_path=args.ref, prompt_audio_text=args.ref_text,
                  text=wt, text_language="zh", prompt_language="zh")

    # 生成器消费：拼接成 wav；服务端可改为逐 chunk 流式播放
    parts, sr = [], 32000
    prev_punct = ""
    ttft_ms = None
    t0 = time.perf_counter()
    for typ, content, end_punct in chunks:
        if typ == "action":
            parts.append(np.zeros(int(PAUSE_SEC["action"] * 32000), dtype=np.float32))
            prev_punct = ""
        else:
            seg_lang = "auto" if FOREIGN_RE.search(content) else args.lang
            audio = tts.infer(
                spk_audio_path=args.ref, prompt_audio_path=args.ref, prompt_audio_text=args.ref_text,
                text=content, text_language=seg_lang, prompt_language="zh",
            )
            if ttft_ms is None:  # 首个语音段就绪时间 = 首字延迟
                ttft_ms = (time.perf_counter() - t0) * 1000
                print(f"首字延迟(TTFT): {ttft_ms:.0f} ms", flush=True)
            sr = audio.samplerate
            if parts and prev_punct:
                parts.append(np.zeros(int(PAUSE_SEC.get(prev_punct, 0.3) * sr), dtype=np.float32))
            parts.append(audio.audio_data)
            prev_punct = end_punct

    full = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), full, sr)
    print(f"合成完成: {len(full) / sr:.2f}s → {out}")


if __name__ == "__main__":
    main()
