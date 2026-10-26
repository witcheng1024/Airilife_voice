"""十种情绪参考的克隆确认（AiriLife M53-07，上线前最后一轮人耳验收）。

读取 <角色>/qwen_vN/emotions/emotions.json（默认梓涵 models/zelia/qwen_v2/emotions），每种情绪用自己的参考片（逐字 ref_text）
以 0.6B Base ICL 克隆两句新台词，生成参数与常驻服务一致、不加指令。只出试听页，不打分。

用法（conda env qwen3-tts-fast，先停掉常驻服务腾显存）：
    python tools/qwen/emotion_clone_check.py
    python tools/qwen/emotion_clone_check.py --emotion-dir models/airi/qwen_v1/emotions
"""

from __future__ import annotations

import argparse
import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "base" / "qwen3-tts" / "Qwen3-TTS-12Hz-0.6B-Base"
EMOTION_DIR = ROOT / "models" / "zelia" / "qwen_v2" / "emotions"
OUT_ROOT = ROOT / "output" / "emotion_clone_check"

TEMPERATURE = 0.9
TOP_K = 50
CHUNK_SIZE = 4

# 每种情绪两句：一句短、一句长，均不是参考片原台词
TEXTS = {
    "neutral": ["嗯，文件我已经帮你放到桌面上了。", "今天下午大概会有点风，出门的话记得带件外套，晚饭我等你回来一起吃。"],
    "happy": ["真的吗！太好啦！", "那周末我们就去那家新开的甜品店吧，我想吃草莓的那个，你也一起尝尝！"],
    "gentle": ["没关系的，慢慢来。", "累了就闭上眼睛休息一会儿吧，我就在旁边，小声陪你说说话。"],
    "sad": ["嗯……没事的。", "那盆花还是没能救活，我每天都有好好浇水的，可它还是枯掉了。"],
    "surprised": ["诶？真的假的！", "你什么时候回来的？我怎么一点都没听见，吓了我一跳！"],
    "nervous": ["等、等一下！", "明天就要面试了吧？资料都准备好了吗？我怎么比你还紧张啊……"],
    "shy": ["才、才没有呢……", "你一直这样看着我干嘛呀，我、我会不好意思的啦。"],
    "tired": ["唔……好困……", "已经这么晚了呀……我眼睛都快睁不开了，你也早点睡吧……"],
    "confused": ["嗯？什么意思？", "你说的是上周那个文件，还是昨天新建的那个呀？我有点分不清了。"],
    "angry": ["哼，不理你了！", "说好了今天陪我打游戏的，结果你又一直在加班，我等了你一整个晚上！"],
}


def main() -> int:
    import numpy as np
    import soundfile as sf
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--emotion-dir", default=str(EMOTION_DIR), help="含 emotions.json 的定稿目录")
    args = ap.parse_args()
    emotion_dir = Path(args.emotion_dir).resolve()

    emotions = json.loads((emotion_dir / "emotions.json").read_text(encoding="utf-8"))["emotions"]
    run_dir = OUT_ROOT / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    model = FasterQwen3TTS.from_pretrained(
        str(MODEL_DIR), device="cuda", dtype=torch.bfloat16, backend="torch", local_files_only=True
    )
    model.warmup()

    rows = []
    for key, ref in emotions.items():
        ref_wav = emotion_dir / ref["file"]
        cells = []
        for i, text in enumerate(TEXTS[key], 1):
            chunks, sr = [], 24000
            for audio, chunk_sr, _t in model.generate_voice_clone_streaming(
                text=text, language="Chinese", ref_audio=str(ref_wav), ref_text=ref["text"],
                chunk_size=CHUNK_SIZE, temperature=TEMPERATURE, top_k=TOP_K,
            ):
                sr = int(chunk_sr)
                chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            name = f"{key}_{i}.wav"
            sf.write(str(run_dir / name), np.concatenate(chunks), sr)
            cells.append(f"<td><small>{html.escape(text)}</small><audio controls preload='none' src='{name}'></audio></td>")
            print(name, flush=True)
        ref_src = f"../../../{(emotion_dir / ref['file']).relative_to(ROOT).as_posix()}"
        rows.append(
            f"<tr><td><b>{html.escape(ref['label'])}</b> <small>{key}</small>"
            f"<audio controls preload='none' src='{ref_src}'></audio></td>{''.join(cells)}</tr>"
        )

    page = f"""<!doctype html><meta charset="utf-8"><title>情绪克隆确认</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}audio{{width:220px;display:block;margin-top:4px}}</style>
<h1>梓涵 v2：十种情绪参考 → 0.6B 克隆确认</h1>
<p>第一列是参考原片；右边两列是用它克隆的新台词（服务同参数、不加指令）。听两点：还是不是梓涵；情绪是否在。</p>
<table><tr><th>情绪 / 参考</th><th>短句</th><th>长句</th></tr>{''.join(rows)}</table>"""
    (run_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"\n试听页：{run_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
