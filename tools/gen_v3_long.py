#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 定版长文多语言测试：s1{e15,e20} × s2{e30,e35,e40,e45,e50}，活泼/温柔情绪，长文中英日混合。

输出：output/v3_long/<组合>/<情绪>_<样本>.wav
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUT = ROOT / "output" / "v3_long"

WSL_GPT = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_weights_v2ProPlus"
WSL_SOV = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_SoVITS/SoVITS_weights_v2ProPlus"

# 组合：s1{e15,e20} × s2{e30..e50}
COMBOS = {
    f"s1e{s1}_s2e{s2}": (f"{WSL_GPT}/firefly_v3-e{s1}.ckpt", f"{WSL_SOV}/firefly_v3_e{s2}_cont.pth")
    for s1 in [15, 20]
    for s2 in [30, 35, 40, 45, 50]
}

# 情绪：活泼 + 温柔
EMO = {
    "活泼": ("活泼.wav", "好啦！看上去真不错，你好上相呀。"),
    "温柔": ("温柔.wav", "这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。"),
}

# 长段多语言文本（中英日混合，测 code-switching 长句）
LONG_TEXTS = {
    "story": (
        "今天天气真好啊，Hi there! 我们一起去散散步吧。"
        "你知道吗，Tomodachi 在日语里就是朋友的意思。"
        "So, let's enjoy this beautiful day together! "
        "我真的很喜欢和你在一起的时光，これからもずっと一緒にいてね。"
    ),
    "phrases": (
        "华哥哥，你听过这句话吗？'The early bird catches the worm.' "
        "意思就是说，早起的鸟儿才有虫吃哦。"
        "それじゃ、今日も一緒に頑張ろうね！"
        "不过也别太累，累了就休息，一切都会好起来的。"
    ),
    "travel": (
        "明天我们出发去旅行吧！First, let's grab some breakfast. "
        "然后去车站，電車で街を巡ろう。"
        "沿途能看到很多漂亮的风景，Every corner tells a story. "
        "你说，我们是不是该拍点照片留作纪念？"
    ),
}


def main() -> None:
    import soundfile as sf

    from gsv_tts import TTS

    # 拷贝权重
    for label, (gpt_src, sov_src) in COMBOS.items():
        for src in (gpt_src, sov_src):
            local = MODELS_DIR / Path(src).name
            if not local.exists() and Path(src).exists():
                print(f"拷贝 {local.name}", flush=True)
                shutil.copy(src, local)

    for label, (gpt_src, sov_src) in COMBOS.items():
        gpt_local = MODELS_DIR / Path(gpt_src).name
        sov_local = MODELS_DIR / Path(sov_src).name
        if not gpt_local.exists() or not sov_local.exists():
            print(f"[{label}] 缺权重，跳过", flush=True)
            continue
        combo_dir = OUT / label
        combo_dir.mkdir(parents=True, exist_ok=True)
        if any(combo_dir.glob("*.wav")):
            print(f"[{label}] 已有输出，跳过", flush=True)
            continue
        print(f"[{label}] 加载 {gpt_local.name} + {sov_local.name}", flush=True)
        tts = TTS(models_dir=str(ROOT / "gsv_models"), use_bert=True)
        tts.load_gpt_model(str(gpt_local))
        tts.load_sovits_model(str(sov_local))
        for emo, (fn, ref_txt) in EMO.items():
            ref = ROOT / "reference_audio" / fn
            for name, text in LONG_TEXTS.items():
                audio = tts.infer(
                    spk_audio_path=str(ref), prompt_audio_path=str(ref), prompt_audio_text=ref_txt,
                    text=text, text_language="auto", prompt_language="zh",
                )
                sf.write(str(combo_dir / f"{emo}_{name}.wav"), audio.audio_data, audio.samplerate)
                print(f"  [{label}] {emo}_{name}: {len(audio.audio_data)/audio.samplerate:.2f}s", flush=True)

    print(f"完成 → {OUT}", flush=True)


if __name__ == "__main__":
    main()
