#!/usr/bin/env python
"""多情绪推理：young_full 模型必须用年轻化参考，不能用原声参考。

用法（须在 GPT-SoVITS 根目录或由本脚本 chdir）:
  python /path/to/infer_emotions.py --s1 <ckpt> --s2 <pth> --out <dir>
  python infer_emotions.py ... --ref-dir <年轻化参考目录>
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

import numpy as np
import soundfile as sf

GS = "/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS"
# 默认：直接使用 09_定稿_asetrate05 的 *_1定稿（禁止原声参考、禁止另起炉灶处理）
REF_DIR_DEFAULT = "/mnt/e/PROJECT/Airilife_voice/runs/00_参考_从09定稿"

os.chdir(GS)
sys.path.insert(0, GS)

from tools.i18n.i18n import I18nAuto  # noqa: E402
from GPT_SoVITS.inference_webui import (  # noqa: E402
    change_gpt_weights,
    change_sovits_weights,
    get_tts_wav,
)

i18n = I18nAuto()

# 情绪: (参考文件名, 参考文本, 短句, 长句)
EMO = {
    "平淡": (
        "平淡.wav",
        "我叫流萤，是鸢尾花家系的艺者。",
        "我在呢，有什么想聊的吗？",
        "今天也平安无事就好。对了，你有没有好好吃饭？别总是忙到忘记这些小事。",
    ),
    "活泼": (
        "活泼.wav",
        "好啦！看上去真不错，你好上相呀。",
        "太好啦！我们出去玩吧！",
        "欸嘿，今天天气这么好，不出去走走太可惜了！走嘛走嘛，我保证不会让你无聊的。",
    ),
    "撒娇甜": (
        "撒娇甜.wav",
        "你想合影留念吗？那就把手机交给我吧，我来拍～",
        "哥哥～陪我一会儿嘛，好不好嘛？",
        "你都不理我……人家好不容易才等到你回来的。再陪我待一会儿，就一小会儿，好不好嘛？",
    ),
    "温柔": (
        "温柔.wav",
        "这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。",
        "别担心啦，有我在呢，一切都会好好的。",
        "没关系的，慢慢来就好。累了就靠一下，我会一直在这里的。你已经做得很好了。",
    ),
    "悲伤": (
        "悲伤.wav",
        "嗯，是我不好。对不起。",
        "呜…为什么今天这么不顺心啊…",
        "有时候……会忽然觉得，有些话再也来不及说了。对不起，让你看到我这样。",
    ),
    "惊讶": (
        "惊讶.wav",
        "真的吗？看来…卡芙卡教给我的没错。",
        "哇！真的假的？！太让人意外啦！",
        "等一下，你刚才说什么？！这、这是真的吗？我完全没想到会是这样……",
    ),
    "生气": (
        "生气.wav",
        "绝对不行，难道你忘了吗？作为星核猎手，我有一项不惜一切代价的「零点任务」——",
        "不行，绝对不行！",
        "我不管你是谁，立刻给我住手！再这样下去，我可真要生气了！",
    ),
    "紧张": (
        "紧张.wav",
        "小心，更多怪物进来了！准备开火！",
        "小心！",
        "小心！前面好像有什么动静…我们得小声一点，别惊动它们。",
    ),
}


def synth(s1, s2, refwav, reftxt, text, out):
    change_gpt_weights(gpt_path=s1)
    change_sovits_weights(sovits_path=s2)
    res = get_tts_wav(
        ref_wav_path=refwav,
        prompt_text=reftxt,
        prompt_language=i18n("中文"),
        text=text,
        text_language=i18n("中文"),
        top_p=0.9,
        temperature=0.7,
    )
    parts, sr = [], None
    for sr, a in res:
        parts.append(a)
    full = np.concatenate(parts)
    sf.write(out, full, sr)
    return len(full) / sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s1", required=True)
    ap.add_argument("--s2", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--ref-dir",
        default=REF_DIR_DEFAULT,
        help="参考音频目录；young_full 必须用年轻化参考，禁止原声参考",
    )
    args = ap.parse_args()

    if "原声参考" in args.ref_dir and "年轻化" not in args.ref_dir:
        print(f"警告: ref-dir 像是原声参考 ({args.ref_dir})，年龄会被拉回原声！")

    ref_out = os.path.join(args.out, "参考_年轻化")
    inf_out = os.path.join(args.out, "推理")
    os.makedirs(ref_out, exist_ok=True)
    os.makedirs(inf_out, exist_ok=True)

    for emo, (fn, reftxt, short, longt) in EMO.items():
        src = os.path.join(args.ref_dir, fn)
        if not os.path.exists(src):
            print(f"缺参考: {src}")
            continue
        dst_ref = os.path.join(ref_out, f"{emo}.wav")
        shutil.copy2(src, dst_ref)
        d1 = synth(args.s1, args.s2, src, reftxt, short, os.path.join(inf_out, f"{emo}_短句.wav"))
        print(f"{emo}_短句: {d1:.1f}s ref={src}")
        d2 = synth(args.s1, args.s2, src, reftxt, longt, os.path.join(inf_out, f"{emo}_长句.wav"))
        print(f"{emo}_长句: {d2:.1f}s")

    with open(os.path.join(args.out, "权重与配方.txt"), "w", encoding="utf-8") as f:
        f.write(
            f"s1={args.s1}\n"
            f"s2={args.s2}\n"
            f"ref_dir={args.ref_dir}\n"
            "ref=年轻化参考(asetrate+0.5+EQ)，禁止原声参考\n"
            "train_recipe=asetrate+0.5 + EQ(80=-2,150=-1,4k=+1.5,9k=-1.5)\n"
        )
    with open(os.path.join(args.out, "听法说明.txt"), "w", encoding="utf-8") as f:
        f.write(
            "参考与训练同配方年轻化，再推理。\n"
            "先听 参考_年轻化/ 再听 推理/ 同情绪。\n"
            "不要用 00_可用_原声参考 喂本模型。\n"
        )
    print(f"完成 → {args.out}")


if __name__ == "__main__":
    main()
