#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 排列组合生成：s1(e15/e45) × s2(e10/e15/e20/e25) → 各组合全 8 情绪中文短句。

输出：output/v3_combos/<s1标签>_<s2标签>/<情绪>.wav
s2 续训权重从 WSL 拷贝到 models/（不存在则跳过该组合）。
"""
from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
OUT = ROOT / "output" / "v3_combos"

# WSL 权重路径
WSL_GPT = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_weights_v2ProPlus"
WSL_SOV = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_SoVITS/SoVITS_weights_v2ProPlus"

# s1 变体：e5~e45，5epoch 间隔（9 个）
S1 = {
    f"s1e{ep}": (MODELS_DIR / f"gpt_firefly_v3-e{ep}.ckpt", f"{WSL_GPT}/firefly_v3-e{ep}.ckpt")
    for ep in [5, 10, 15, 20, 25, 30, 35, 40, 45]
}

# s2 变体：e10（原）+ e15/e20/e25/e30/e35/e40（续训），5epoch 间隔（7 个）
S2 = {
    "s2e10": (MODELS_DIR / "sovits_firefly_v3_e10.pth", None),
}
S2.update({
    f"s2e{ep}": (MODELS_DIR / f"sovits_firefly_v3_e{ep}_cont.pth", f"{WSL_SOV}/firefly_v3_e{ep}_cont.pth")
    for ep in [15, 20, 25, 30, 35, 40]
})

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


def ensure_weights() -> None:
    """把 WSL 的 s1/s2 权重拷到 models/（存在才拷；s1 每 5 epoch 一个）。"""
    for label, (local, src) in {**S1, **S2}.items():
        if src and not local.exists():
            if Path(src).exists():
                print(f"拷贝 {label}: {src} → {local}", flush=True)
                shutil.copy(src, local)
            else:
                print(f"跳过 {label}（WSL 权重未就绪: {src}）", flush=True)


def main() -> None:
    import soundfile as sf

    from gsv_tts import TTS

    ensure_weights()

    avail_s1 = {k: v[0] for k, v in S1.items() if v[0].exists()}
    avail_s2 = {k: v[0] for k, v in S2.items() if v[0].exists()}
    print(f"s1 可用: {list(avail_s1)}；s2 可用: {list(avail_s2)}", flush=True)
    if not avail_s1 or not avail_s2:
        raise SystemExit("缺 s1 或 s2 权重")

    for s1l, s1p in avail_s1.items():
        for s2l, s2p in avail_s2.items():
            label = f"{s1l}_{s2l}"
            combo_dir = OUT / label
            combo_dir.mkdir(parents=True, exist_ok=True)
            if any(combo_dir.glob("*.wav")):
                print(f"[{label}] 已有输出，跳过", flush=True)
                continue
            print(f"[{label}] 加载 {s1p.name} + {s2p.name}", flush=True)
            tts = TTS(models_dir=str(ROOT / "gsv_models"), use_bert=True)
            tts.load_gpt_model(str(s1p))
            tts.load_sovits_model(str(s2p))
            for emo, (fn, ref_txt, short) in EMO.items():
                ref = ROOT / "reference_audio" / fn
                audio = tts.infer(
                    spk_audio_path=str(ref), prompt_audio_path=str(ref), prompt_audio_text=ref_txt,
                    text=short, text_language="auto", prompt_language="zh",
                )
                sf.write(str(combo_dir / f"{emo}.wav"), audio.audio_data, audio.samplerate)
                print(f"  [{label}] {emo}: {len(audio.audio_data)/audio.samplerate:.2f}s", flush=True)

    print(f"完成 → {OUT}", flush=True)


if __name__ == "__main__":
    main()
