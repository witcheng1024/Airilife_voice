#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 语种策略验证：大部分中文 + 日文仅可爱词 + 英文仅专有名词。

测 text_language=zh vs auto；组合 s1e15×s2e40/e50（用户偏好系列）。
输出：output/v3_style/<组合>/<情绪>_<样本>_<zh|auto>.wav
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUT = ROOT / "output" / "v3_style"

WSL_GPT = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_weights_v2ProPlus"
WSL_SOV = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_SoVITS/SoVITS_weights_v2ProPlus"

COMBOS = {
    f"s1e{s1}_s2e{s2}": (f"{WSL_GPT}/firefly_v3-e{s1}.ckpt", f"{WSL_SOV}/firefly_v3_e{s2}_cont.pth")
    for s1 in [15, 20]
    for s2 in [40, 50]
}

# 语种策略文本：(情绪, 参考文件, 参考文本, 目标文本)
# 日文仅可爱词；英文仅专有名词
STYLE_TEXTS = {
    "cute_apology": ("撒娇甜", "撒娇甜.wav", "我最喜欢这家店的橡木蛋糕卷，每天都要吃一个。",
                     "哥哥，真的对不起嘛…ごめんね，我不是故意的，原谅我好不好嘛？"),
    "tech": ("活泼", "活泼.wav", "好啦！看上去真不错，你好上相呀。",
             "我今天在学 Python，还查了一下 wiki，那个 Transformer 模型真的好厉害啊！"),
    "comfort": ("温柔", "温柔.wav", "这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。",
                "没关系的，累了就靠着我休息一下吧。ありがとう，愿意听我说这些。"),
    "pure_zh": ("平淡", "平淡.wav", "我叫流萤，是鸢尾花家系的艺者。",
                "今天天气真好，我们出去散步吧。你开心我就开心。"),
}


def main() -> None:
    import soundfile as sf

    from gsv_tts import TTS

    for label, (gpt_src, sov_src) in COMBOS.items():
        for src in (gpt_src, sov_src):
            local = MODELS_DIR / Path(src).name
            if not local.exists() and Path(src).exists():
                print(f"拷贝 {local.name}", flush=True)
                shutil.copy(src, local)

    for label, (gpt_src, sov_src) in COMBOS.items():
        gpt_local = MODELS_DIR / Path(gpt_src).name
        sov_local = MODELS_DIR / Path(sov_src).name
        combo_dir = OUT / label
        combo_dir.mkdir(parents=True, exist_ok=True)
        if any(combo_dir.glob("*.wav")):
            print(f"[{label}] 已有输出，跳过", flush=True)
            continue
        print(f"[{label}] 加载 {gpt_local.name} + {sov_local.name}", flush=True)
        tts = TTS(models_dir=str(ROOT / "gsv_models"), use_bert=True)
        tts.load_gpt_model(str(gpt_local))
        tts.load_sovits_model(str(sov_local))
        for name, (emo, fn, ref_txt, text) in STYLE_TEXTS.items():
            ref = ROOT / "reference_audio" / fn
            for lang in ("zh", "auto"):
                audio = tts.infer(
                    spk_audio_path=str(ref), prompt_audio_path=str(ref), prompt_audio_text=ref_txt,
                    text=text, text_language=lang, prompt_language="zh",
                )
                p = combo_dir / f"{emo}_{name}_{lang}.wav"
                sf.write(str(p), audio.audio_data, audio.samplerate)
                print(f"  [{label}] {emo}_{name}_{lang}: {len(audio.audio_data)/audio.samplerate:.2f}s", flush=True)

    print(f"完成 → {OUT}", flush=True)


if __name__ == "__main__":
    main()
