"""从 ModelScope 拉取 Qwen3-TTS 权重（音色设计 + 音色复现）。

只做下载，不加载模型，因此不占 GPU。默认存到 models/qwen3-tts/<repo-name>/，
该目录已被 .gitignore 排除（权重不进 git）。

用法：
    python tools/fetch_qwen3_tts_models.py                # 下载 voice-design + 2 个 base
    python tools/fetch_qwen3_tts_models.py --only voice-design
    python tools/fetch_qwen3_tts_models.py --list         # 只列出体积，不下载

三个模型各自自带 speech_tokenizer 子目录（约 651MB tokenizer 会各存一份，
换来单目录可直接 from_pretrained，省掉额外的路径配置）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = ROOT / "models" / "qwen3-tts"

# key -> (ModelScope repo id, 用途说明)
REPOS = {
    "voice-design": ("Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign", "1.7B VoiceDesign：文字描述生成原创音色（~4.3GB）"),
    "base-1.7b": ("Qwen/Qwen3-TTS-12Hz-1.7B-Base", "1.7B Base：参考音频克隆声线（~4.3GB）"),
    "base-0.6b": ("Qwen/Qwen3-TTS-12Hz-0.6B-Base", "0.6B Base：轻量克隆，实时 TTS 候选（~2.4GB）"),
}


def human(n: int) -> str:
    return f"{n / 1024 ** 3:.2f}GB"


def list_repos() -> None:
    from modelscope.hub.api import HubApi

    api = HubApi()
    total = 0
    for key, (repo, desc) in REPOS.items():
        files = api.get_model_files(model_id=repo, recursive=True)
        size = sum(f.get("Size", 0) for f in files)
        total += size
        print(f"{key:14s} {repo:44s} {human(size):>8s}  {desc}")
    print(f"{'合计':14s} {'':44s} {human(total):>8s}")


def download(keys: list[str]) -> None:
    from modelscope import snapshot_download

    for key in keys:
        repo, desc = REPOS[key]
        target = MODEL_ROOT / repo.split("/")[-1]
        target.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {key}: {repo} -> {target}\n    {desc}", flush=True)
        path = snapshot_download(repo, local_dir=str(target))
        print(f"    done: {path}", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="+", choices=sorted(REPOS), help="只下载指定模型")
    parser.add_argument("--list", action="store_true", help="只列出各仓库体积")
    parser.add_argument("--modelscope-cache", default=str(ROOT / ".cache" / "modelscope"), help="ModelScope 缓存目录")
    args = parser.parse_args()

    os.environ.setdefault("MODELSCOPE_CACHE", args.modelscope_cache)
    # 走魔搭国内源，避免 HF 直连
    os.environ.pop("HF_ENDPOINT", None)

    if args.list:
        list_repos()
        return 0

    download(args.only or list(REPOS))
    print("\n全部完成。权重目录：", MODEL_ROOT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
