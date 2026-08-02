#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""组装 v3 多语言训练数据：中文 678 + 英/日/混合 → 一份 GPT-SoVITS train.list（/mnt/e 路径）。

用法：
  python tools/assemble_multilingual.py --out data/firefly_multilingual/train.list

输入：
  data/firefly_678_orig/train.list              # 中文 678（路径转 /mnt/e）
  data/firefly_multilingual/manifest.json        # gen_multilingual.py 产物（en/ja）
  data/firefly_multilingual/mix_manifest.json    # 代码切换合成产物（mix_en/mix_ja）

train.list 行格式：wav_path|firefly|lang|text（1-get-text.py 支持 zh/en/ja）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MNT = "/mnt/e/PROJECT/Airilife_voice_v2"

ZH_LIST = ROOT / "data" / "firefly_678_orig" / "train.list"
ML_DIR = ROOT / "data" / "firefly_multilingual"


def zh_lines() -> list[str]:
    """中文 678 全部保留（含 {NICKNAME} 行，与 v2 训练数据一致）。wav 已平铺到根目录。"""
    out = []
    for line in ZH_LIST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        wav, spk, lang, text = line.split("|", 3)
        name = Path(wav).name  # 只取文件名，平铺在 firefly_multilingual/ 根目录
        out.append(f"{MNT}/data/firefly_multilingual/{name}|{spk}|zh|{text.strip()}")
    return out


def gen_lines_from_translations(lang: str) -> list[str]:
    """EN/JA：按 translations.json + 已生成的 wav 文件（<lang>_<i+1>.wav ↔ item[i][lang]）重建。"""
    out = []
    trans = json.loads((ROOT / "output" / "multilingual" / "translations.json").read_text(encoding="utf-8"))
    for i, item in enumerate(trans):
        text = item.get(lang)
        if not text:
            continue
        wav = ML_DIR / f"{lang}_{i + 1}.wav"
        if not wav.exists():
            continue
        out.append(f"{MNT}/data/firefly_multilingual/{wav.name}|firefly|{lang}|{text}")
    return out


def gen_lines_mix(mix_manifest_path: Path) -> list[str]:
    out = []
    if not mix_manifest_path.exists():
        return out
    for item in json.loads(mix_manifest_path.read_text(encoding="utf-8")).get("items", []):
        if item.get("status") != "OK":
            continue
        p = Path(item["path"]).name
        out.append(f"{MNT}/data/firefly_multilingual/{p}|firefly|zh|{item['text']}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default=str(ML_DIR / "train.list"))
    ap.add_argument("--mix-manifest", default=str(ML_DIR / "_mix_manifest.json"))
    args = ap.parse_args()

    zh = zh_lines()
    en = gen_lines_from_translations("en")
    ja = gen_lines_from_translations("ja")
    mix = gen_lines_mix(Path(args.mix_manifest))

    total = len(zh) + len(en) + len(ja) + len(mix)
    out = zh + en + ja + mix
    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"train.list: zh={len(zh)} en={len(en)} ja={len(ja)} mix={len(mix)} 合计={total} → {args.out}")
    print("示例 EN 行:")
    for l in en[:2]:
        print(" ", l)


if __name__ == "__main__":
    main()
