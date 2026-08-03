#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 多语言模型 · 全 8 情绪中文短句验证（各情绪用对应参考音频）。

输出：output/v3_demo/emotions/<情绪>.wav（与 v2 的推理/短句 同文本可对比）
"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "gsv_models"
GPT = ROOT / "models" / "gpt_firefly_v3-e45.ckpt"
SOV = ROOT / "models" / "sovits_firefly_v3_e10.pth"
REF_DIR = ROOT / "reference_audio"
OUT = ROOT / "output" / "v3_demo" / "emotions"

EMO = {
    "平淡":   ("平淡.wav", "我叫流萤，是鸢尾花家系的艺者。", "我在呢，有什么想聊的吗？"),
    "活泼":   ("活泼.wav", "好啦！看上去真不错，你好上相呀。", "太好啦！我们出去玩吧！"),
    "撒娇甜": ("撒娇甜.wav", "我最喜欢这家店的橡木蛋糕卷，每天都要吃一个。", "哥哥～陪我一会儿嘛，好不好嘛？"),
    "温柔":   ("温柔.wav", "这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。", "别担心啦，有我在呢，一切都会好好的。"),
    "悲伤":   ("悲伤.wav", "嗯，是我不好。对不起。", "呜…为什么今天这么不顺心啊…"),
    "惊讶":   ("惊讶.wav", "真的吗？看来…卡芙卡教给我的没错。", "哇！真的假的？！太让人意外啦！"),
    "生气":   ("生气.wav", "绝对不行，难道你忘了吗？作为星核猎手，我有一项不惜一切代价的「零点任务」——", "不行，绝对不行！"),
    "紧张":   ("紧张.wav", "小心，更多怪物进来了！准备开火！", "小心！"),
}


def main() -> None:
    import soundfile as sf

    from gsv_tts import TTS

    print("加载 v3 模型...", flush=True)
    tts = TTS(models_dir=str(MODELS_DIR), use_bert=True)
    tts.load_gpt_model(str(GPT))
    tts.load_sovits_model(str(SOV))
    OUT.mkdir(parents=True, exist_ok=True)

    for emo, (fn, ref_txt, short) in EMO.items():
        ref = REF_DIR / fn
        if not ref.exists():
            print(f"缺参考: {ref}", flush=True)
            continue
        audio = tts.infer(
            spk_audio_path=str(ref), prompt_audio_path=str(ref), prompt_audio_text=ref_txt,
            text=short, text_language="auto", prompt_language="zh",
        )
        p = OUT / f"{emo}.wav"
        sf.write(str(p), audio.audio_data, audio.samplerate)
        print(f"{emo}: {len(audio.audio_data)/audio.samplerate:.2f}s → {p}", flush=True)


if __name__ == "__main__":
    main()
