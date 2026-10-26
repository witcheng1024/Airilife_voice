"""把试听选定的 VoiceDesign 候选片定稿入库：output/ → models/<角色>/qwen_vN/emotions/。

output/ 只是草稿（不入库、可随时删）；定稿只认 models/<角色>/qwen_vN/emotions/：
  <canonical>.wav     每种情绪一条参考
  emotions.json       label / source / scene / text（text 从出片 manifest 取，保证与音频逐字一致）
  index.html          十情绪试听页（每次运行自动重建）

用法：
    # 选片入库（可分多次，已有的情绪会被覆盖，没提到的保留）
    python tools/qwen/promote_emotions.py --dest models/airi/qwen_v1 ^
        neutral=output/qwen3_voice_design/airi_v1/20261026_round1/daily_t2.wav ^
        gentle=output/qwen3_voice_design/airi_v1/20261026_round1/gentle_t1.wav

    # 同时写进服务音色清单 voices/qwen3_voices.json（按 AiriLife 协议顺序）
    python tools/qwen/promote_emotions.py --dest models/airi/qwen_v1 --register airi --label Airi

    # 只重建试听页
    python tools/qwen/promote_emotions.py --dest models/zelia/qwen_v2
"""

from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
VOICES_FILE = ROOT / "voices" / "qwen3_voices.json"

# AiriLife 输出协议的情绪顺序：canonical → 服务情绪标签
EMOTIONS = {
    "neutral": "普通",
    "happy": "高兴",
    "sad": "难过",
    "angry": "生气",
    "surprised": "惊讶",
    "shy": "害羞",
    "tired": "困倦",
    "gentle": "温柔",
    "nervous": "紧张",
    "confused": "困惑",
}


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def lookup_record(wav: Path) -> dict:
    manifest = wav.parent / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"找不到出片记录：{manifest}（只能从 qwen3_voice_design.py 的产物里选片）")
    records = json.loads(manifest.read_text(encoding="utf-8"))["records"]
    for r in records:
        if r["file"] == wav.name:
            return r
    raise SystemExit(f"{manifest} 里没有 {wav.name}")


def write_index(emotion_dir: Path, title: str, emotions: dict) -> None:
    rows = []
    for key, e in emotions.items():
        rows.append(
            f"<section><h2>{html.escape(e['label'])} <small>{key} · 来源 {html.escape(e['source'])}</small></h2>"
            f"<audio controls preload=\"none\" src=\"{e['file']}\"></audio><p>{html.escape(e['text'])}</p></section>"
        )
    missing = [f"{EMOTIONS[k]}（{k}）" for k in EMOTIONS if k not in emotions]
    note = f"<p class=\"miss\">尚缺：{'、'.join(missing)}</p>" if missing else ""
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(title)}情绪参考</title>
<style>
 body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; max-width: 760px; margin: 24px auto; padding: 0 16px; background:#faf8f6; color:#2b2b2b; }}
 section {{ border-bottom: 1px solid #e6e0da; padding: 10px 0; }}
 h2 {{ font-size: 17px; margin: 0 0 6px; }}
 small {{ color: #888; font-weight: normal; font-size: 12px; }}
 audio {{ width: 100%; }}
 p {{ margin: 6px 0; color: #444; }}
 .miss {{ color: #b0522d; }}
</style></head><body>
<h1>{html.escape(title)} · 情绪参考定稿</h1>
<p>清单见 emotions.json；服务读取 voices/qwen3_voices.json。</p>
{note}
{''.join(rows)}
</body></html>
"""
    (emotion_dir / "index.html").write_text(page, encoding="utf-8")


def register(voice_id: str, label: str, emotion_dir: Path, emotions: dict) -> None:
    data = json.loads(VOICES_FILE.read_text(encoding="utf-8"))
    voice = data["voices"].setdefault(voice_id, {"label": label, "language": "Chinese"})
    if label:
        voice["label"] = label
    voice["references"] = {
        e["label"]: {"audio": rel(emotion_dir / e["file"]), "text": e["text"]} for e in emotions.values()
    }
    VOICES_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"已写入 {rel(VOICES_FILE)} → {voice_id}（{len(emotions)} 种情绪），重启服务生效")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True, help="定稿目录，如 models/airi/qwen_v1")
    ap.add_argument("picks", nargs="*", help="canonical=候选 wav 路径，canonical 取 " + " / ".join(EMOTIONS))
    ap.add_argument("--register", metavar="VOICE_ID", help="同时写入 voices/qwen3_voices.json 的这个音色")
    ap.add_argument("--label", default=None, help="--register 时的音色显示名")
    args = ap.parse_args()

    emotion_dir = (ROOT / args.dest / "emotions").resolve()
    emotion_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = emotion_dir / "emotions.json"
    old = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    existing = old.get("emotions", {})

    for pick in args.picks:
        key, sep, src = pick.partition("=")
        if not sep or key not in EMOTIONS:
            raise SystemExit(f"格式应为 canonical=路径，canonical 取 {list(EMOTIONS)}：{pick}")
        wav = (ROOT / src).resolve()
        record = lookup_record(wav)
        shutil.copyfile(wav, emotion_dir / f"{key}.wav")
        existing[key] = {
            "label": EMOTIONS[key],
            "file": f"{key}.wav",
            "source": rel(wav),
            "scene": record["scene"],
            "text": record["text"],
        }
        print(f"{EMOTIONS[key]:<3} ← {rel(wav)}")

    emotions = {k: existing[k] for k in EMOTIONS if k in existing}
    comment = old.get("_comment") or (
        "试听选定的情绪参考。键为 AiriLife canonical 情绪，label 为服务情绪标签；text 与音频逐字一致（ICL）。"
        "由 tools/qwen/promote_emotions.py 维护。"
    )
    manifest_path.write_text(
        json.dumps({"_comment": comment, "emotions": emotions}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    title = args.label or Path(args.dest).parent.name
    write_index(emotion_dir, title, emotions)
    print(f"定稿 {len(emotions)}/{len(EMOTIONS)} → {rel(emotion_dir)}（试听页 index.html）")

    if args.register:
        register(args.register, args.label, emotion_dir, emotions)
    return 0


if __name__ == "__main__":
    sys.exit(main())
