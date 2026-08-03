#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 定版验证：s1e15 × s2{e40,e45,e50} × 8 情绪（紧张改用长句，便于听辨）。

输出：output/v3_final/<组合>/<情绪>.wav
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUT = ROOT / "output" / "v3_final"

WSL_GPT = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_weights_v2ProPlus"
WSL_SOV = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_SoVITS/SoVITS_weights_v2ProPlus"

COMBOS = {
    f"s1e15_s2e{ep}": (f"{WSL_GPT}/firefly_v3-e15.ckpt", f"{WSL_SOV}/firefly_v3_e{ep}_cont.pth")
    for ep in [40, 45, 50]
}

# 8 情绪；紧张用长句（短句只有"小心！"两个字，不好听辨）
EMO = {
    "平淡":   ("平淡.wav", "我叫流萤，是鸢尾花家系的艺者。", "我在呢，有什么想聊的吗？"),
    "活泼":   ("活泼.wav", "好啦！看上去真不错，你好上相呀。", "太好啦！我们出去玩吧！"),
    "撒娇甜": ("撒娇甜.wav", "我最喜欢这家店的橡木蛋糕卷，每天都要吃一个。", "哥哥～陪我一会儿嘛，好不好嘛？"),
    "温柔":   ("温柔.wav", "这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。", "别担心啦，有我在呢，一切都会好好的。"),
    "悲伤":   ("悲伤.wav", "嗯，是我不好。对不起。", "呜…为什么今天这么不顺心啊…"),
    "惊讶":   ("惊讶.wav", "真的吗？看来…卡芙卡教给我的没错。", "哇！真的假的？！太让人意外啦！"),
    "生气":   ("生气.wav", "绝对不行，难道你忘了吗？作为星核猎手，我有一项不惜一切代价的「零点任务」——", "不行，绝对不行！"),
    "紧张":   ("紧张.wav", "小心，更多怪物进来了！准备开火！", "小心！前面好像有什么动静…我们得小声一点，别惊动它们。"),
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
        for emo, (fn, ref_txt, text) in EMO.items():
            ref = ROOT / "reference_audio" / fn
            audio = tts.infer(
                spk_audio_path=str(ref), prompt_audio_path=str(ref), prompt_audio_text=ref_txt,
                text=text, text_language="auto", prompt_language="zh",
            )
            sf.write(str(combo_dir / f"{emo}.wav"), audio.audio_data, audio.samplerate)
            print(f"  [{label}] {emo}: {len(audio.audio_data)/audio.samplerate:.2f}s", flush=True)

    print(f"完成 → {OUT}", flush=True)


if __name__ == "__main__":
    main()
