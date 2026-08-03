#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v3 多语言模型统一生成/测试入口（替代零散的 gen_v3_*.py）。

子命令：
  demo       基础 4 样（zh/en/ja/mix）
  emotions   8 情绪中文短句（默认组合）
  combos     s1×s2 全网格（s1 e5-45 × s2 e10-40）
  long       长文多语言（s1{e15,e20}×s2{e30..e50} × 活泼/温柔 × 3 长文）
  style      语种策略（s1{e15,e20}×s2{e40,e50} × 4 文本 × zh/auto）

默认组合（定版）：s1e20_s2e40、text_language=zh（--lang 可改）。
输出统一到 output/v3_samples/<子命令>/，已生成跳过（可断点续传）。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
GSV = ROOT / "gsv_models"
REF_DIR = ROOT / "reference_audio"
OUT = ROOT / "output" / "v3_samples"

WSL_GPT = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_weights_v2ProPlus"
WSL_SOV = "//wsl$/Ubuntu24.04/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/GPT_SoVITS/SoVITS_weights_v2ProPlus"

# 定版默认组合
DEFAULT_S1, DEFAULT_S2 = 20, 40

# 8 情绪：(参考文件, 参考文本, 短句)
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

# 基础 demo 文本
DEMO_TEXT = {
    "zh": "我叫流萤，是鸢尾花家系的艺者。",
    "en": "Hi! I'm Firefly, an artist from the Iris family. It's nice to meet you!",
    "ja": "こんにちは！私はイリス家系の芸者、ホタルです。",
    "mix": "嗨！今天天气真好，Hi there! How's your day going? 要不要一起去散步？",
}

# 长文多语言
LONG_TEXTS = {
    "story": "今天天气真好啊，Hi there! 我们一起去散散步吧。你知道吗，Tomodachi 在日语里就是朋友的意思。So, let's enjoy this beautiful day together! 我真的很喜欢和你在一起的时光，これからもずっと一緒にいてね。",
    "phrases": "华哥哥，你听过这句话吗？'The early bird catches the worm.' 意思就是说，早起的鸟儿才有虫吃哦。それじゃ、今日も一緒に頑張ろうね！不过也别太累，累了就休息，一切都会好起来的。",
    "travel": "明天我们出发去旅行吧！First, let's grab some breakfast. 然后去车站，電車で街を巡ろう。沿途能看到很多漂亮的风景，Every corner tells a story. 你说，我们是不是该拍点照片留作纪念？",
}

# 语种策略文本：(情绪, 参考文件, 参考文本, 目标文本)
STYLE_TEXTS = {
    "cute_apology": ("撒娇甜", "撒娇甜.wav", "我最喜欢这家店的橡木蛋糕卷，每天都要吃一个。", "哥哥，真的对不起嘛…ごめんね，我不是故意的，原谅我好不好嘛？"),
    "tech": ("活泼", "活泼.wav", "好啦！看上去真不错，你好上相呀。", "我今天在学 Python，还查了一下 wiki，那个 Transformer 模型真的好厉害啊！"),
    "comfort": ("温柔", "温柔.wav", "这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。", "没关系的，累了就靠着我休息一下吧。ありがとう，愿意听我说这些。"),
    "pure_zh": ("平淡", "平淡.wav", "我叫流萤，是鸢尾花家系的艺者。", "今天天气真好，我们出去散步吧。你开心我就开心。"),
}


def s1_path(ep: int) -> Path:
    return MODELS / f"gpt_firefly_v3-e{ep}.ckpt"


def s2_path(ep: int) -> Path:
    return MODELS / f"sovits_firefly_v3_e{ep}_cont.pth"


def ensure_weights(paths: list[Path], wsl: list[str]) -> None:
    for local, src in zip(paths, wsl):
        if not local.exists() and Path(src).exists():
            print(f"  拷贝 {local.name}", flush=True)
            shutil.copy(src, local)


def get_tts(gpt: Path, sovits: Path):
    from gsv_tts import TTS

    if not gpt.exists() or not sovits.exists():
        print(f"缺权重: {gpt.name} / {sovits.name}", flush=True)
        return None
    tts = TTS(models_dir=str(GSV), use_bert=True)
    tts.load_gpt_model(str(gpt))
    tts.load_sovits_model(str(sovits))
    return tts


def synth(tts, ref_fn: str, ref_txt: str, text: str, lang: str, out_path: Path) -> None:
    import soundfile as sf

    ref = REF_DIR / ref_fn
    audio = tts.infer(
        spk_audio_path=str(ref), prompt_audio_path=str(ref), prompt_audio_text=ref_txt,
        text=text, text_language=lang, prompt_language="zh",
    )
    sf.write(str(out_path), audio.audio_data, audio.samplerate)
    print(f"  {out_path.parent.name}/{out_path.name}: {len(audio.audio_data)/audio.samplerate:.2f}s", flush=True)


def cmd_demo(args) -> None:
    ensure_weights([s1_path(args.s1), s2_path(args.s2)], [f"{WSL_GPT}/firefly_v3-e{args.s1}.ckpt", f"{WSL_SOV}/firefly_v3_e{args.s2}_cont.pth"])
    tts = get_tts(s1_path(args.s1), s2_path(args.s2))
    out_dir = OUT / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in DEMO_TEXT.items():
        if (out_dir / f"{name}.wav").exists():
            continue
        synth(tts, "活泼.wav", "好啦！看上去真不错，你好上相呀。", text, args.lang, out_dir / f"{name}.wav")


def cmd_emotions(args) -> None:
    ensure_weights([s1_path(args.s1), s2_path(args.s2)], [f"{WSL_GPT}/firefly_v3-e{args.s1}.ckpt", f"{WSL_SOV}/firefly_v3_e{args.s2}_cont.pth"])
    tts = get_tts(s1_path(args.s1), s2_path(args.s2))
    out_dir = OUT / f"emotions_s1e{args.s1}_s2e{args.s2}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for emo, (fn, ref_txt, short) in EMO.items():
        if (out_dir / f"{emo}.wav").exists():
            continue
        synth(tts, fn, ref_txt, short, args.lang, out_dir / f"{emo}.wav")


def cmd_combos(args) -> None:
    s1s = args.s1s or list(range(5, 46, 5))
    s2s = args.s2s or list(range(10, 41, 5))
    for s1 in s1s:
        for s2 in s2s:
            ensure_weights([s1_path(s1), s2_path(s2)], [f"{WSL_GPT}/firefly_v3-e{s1}.ckpt", f"{WSL_SOV}/firefly_v3_e{s2}_cont.pth"])
    for s1 in s1s:
        for s2 in s2s:
            out_dir = OUT / "combos" / f"s1e{s1}_s2e{s2}"
            if any(out_dir.glob("*.wav")) if out_dir.exists() else False:
                continue
            tts = get_tts(s1_path(s1), s2_path(s2))
            if tts is None:
                continue
            out_dir.mkdir(parents=True, exist_ok=True)
            for emo, (fn, ref_txt, short) in EMO.items():
                synth(tts, fn, ref_txt, short, "auto", out_dir / f"{emo}.wav")


def cmd_long(args) -> None:
    for s1, s2 in [(s1, s2) for s1 in (args.s1s or [15, 20]) for s2 in (args.s2s or [30, 35, 40, 45, 50])]:
        ensure_weights([s1_path(s1), s2_path(s2)], [f"{WSL_GPT}/firefly_v3-e{s1}.ckpt", f"{WSL_SOV}/firefly_v3_e{s2}_cont.pth"])
    for s1, s2 in [(s1, s2) for s1 in (args.s1s or [15, 20]) for s2 in (args.s2s or [30, 35, 40, 45, 50])]:
        out_dir = OUT / "long" / f"s1e{s1}_s2e{s2}"
        if out_dir.exists() and any(out_dir.glob("*.wav")):
            continue
        tts = get_tts(s1_path(s1), s2_path(s2))
        if tts is None:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for emo, (fn, ref_txt) in (("活泼", EMO["活泼"][:2]), ("温柔", EMO["温柔"][:2])):
            for name, text in LONG_TEXTS.items():
                synth(tts, fn, ref_txt, text, "auto", out_dir / f"{emo}_{name}.wav")


def cmd_style(args) -> None:
    for s1, s2 in [(s1, s2) for s1 in (args.s1s or [15, 20]) for s2 in (args.s2s or [40, 50])]:
        ensure_weights([s1_path(s1), s2_path(s2)], [f"{WSL_GPT}/firefly_v3-e{s1}.ckpt", f"{WSL_SOV}/firefly_v3_e{s2}_cont.pth"])
    for s1, s2 in [(s1, s2) for s1 in (args.s1s or [15, 20]) for s2 in (args.s2s or [40, 50])]:
        out_dir = OUT / "style" / f"s1e{s1}_s2e{s2}"
        if out_dir.exists() and any(out_dir.glob("*.wav")):
            continue
        tts = get_tts(s1_path(s1), s2_path(s2))
        if tts is None:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, (emo, fn, ref_txt, text) in STYLE_TEXTS.items():
            for lang in (args.lang, "auto"):
                if (out_dir / f"{emo}_{name}_{lang}.wav").exists():
                    continue
                synth(tts, fn, ref_txt, text, lang, out_dir / f"{emo}_{name}_{lang}.wav")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn, help_ in [
        ("demo", cmd_demo, "基础 4 样（zh/en/ja/mix）"),
        ("emotions", cmd_emotions, "8 情绪短句"),
        ("combos", cmd_combos, "s1×s2 全网格"),
        ("long", cmd_long, "长文多语言"),
        ("style", cmd_style, "语种策略"),
    ]:
        p = sub.add_parser(name, help=help_)
        p.add_argument("--s1", type=int, default=DEFAULT_S1)
        p.add_argument("--s2", type=int, default=DEFAULT_S2)
        p.add_argument("--s1s", type=int, nargs="*")
        p.add_argument("--s2s", type=int, nargs="*")
        p.add_argument("--lang", default="zh")
        p.set_defaults(func=fn)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
