"""锚点克隆 × 逐情绪设计 对照试听（Airi v1 起用的“先定权威锚点”流程）。

同一段 VoiceDesign 描述每次随机出的音色并不一致，所以先选一条权威锚点，再比较两种得到情绪参考的做法：
  A：0.6B Base 以锚点为 ICL 参考，直接克隆每个情绪场景的台词（音色跟锚点走，情绪靠台词）
  B：VoiceDesign 逐情绪多出几版（qwen3_voice_design.py 的产物），人耳挑和锚点同一个人的

页面按“情绪场景”排行：A 每个锚点一列，B 的全部候选放最后一列，不标分数。
A 的产物本身也可以当情绪参考：选中后同样走 promote_emotions.py 入库（本目录会写 manifest.json）。
生成参数与常驻服务一致：faster-qwen3-tts、ICL、bf16、chunk 4、temperature 0.9、top_k 50，流式。

用法（conda env qwen3-tts-fast，先停掉常驻服务腾显存）：
    python tools/qwen/anchor_emotion_clone.py --voice airi_v1 ^
        --anchors output/qwen3_voice_design/airi_v1/20261026_round1/daily_t1.wav ^
                  output/qwen3_voice_design/airi_v1/20261026_round1/explain_t2.wav ^
        --design-runs output/qwen3_voice_design/airi_v1/20261026_round1 output/qwen3_voice_design/airi_v1/20261026_round2
"""

from __future__ import annotations

import argparse
import html
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "base" / "qwen3-tts" / "Qwen3-TTS-12Hz-0.6B-Base"
PROMPTS_FILE = ROOT / "voices" / "qwen3_voice_prompts.json"
OUT_ROOT = ROOT / "output" / "anchor_clone"

TEMPERATURE = 0.9
TOP_K = 50
CHUNK_SIZE = 4


def anchor_record(wav: Path) -> dict:
    """锚点必须来自 VoiceDesign 产物：ref_text 从同目录 manifest.json 取，保证逐字一致。"""
    records = json.loads((wav.parent / "manifest.json").read_text(encoding="utf-8"))["records"]
    for r in records:
        if r["file"] == wav.name:
            return r
    raise SystemExit(f"{wav.parent / 'manifest.json'} 里没有 {wav.name}")


def design_takes(run_dirs: list[Path]) -> dict[str, list[Path]]:
    takes: dict[str, list[Path]] = {}
    for run_dir in run_dirs:
        for r in json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["records"]:
            takes.setdefault(r["scene"], []).append(run_dir / r["file"])
    return takes


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--voice", required=True, help="voices/qwen3_voice_prompts.json 里的音色 key")
    ap.add_argument("--anchors", nargs="+", required=True, help="锚点 wav（VoiceDesign 产物）")
    ap.add_argument("--scenes", nargs="+", default=None, help="只跑这些场景，默认全部")
    ap.add_argument("--design-runs", nargs="*", default=[], help="B 路线的 VoiceDesign 出片目录，放进同一页对照")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    voice = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))[args.voice]
    scenes = {k: v for k, v in voice["scenes"].items() if not args.scenes or k in args.scenes}
    anchors = [(ROOT / a).resolve() for a in args.anchors]
    anchor_recs = [anchor_record(a) for a in anchors]
    takes = design_takes([(ROOT / d).resolve() for d in args.design_runs])

    run_dir = OUT_ROOT / args.voice / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import soundfile as sf
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    model = FasterQwen3TTS.from_pretrained(
        str(MODEL_DIR), device="cuda", dtype=torch.bfloat16, backend="torch", local_files_only=True
    )
    model.warmup()

    records = []
    for anchor, rec in zip(anchors, anchor_recs):
        tag = anchor.stem
        for scene, cfg in scenes.items():
            chunks, sr = [], 24000
            for audio, chunk_sr, _t in model.generate_voice_clone_streaming(
                text=cfg["text"], language="Chinese", ref_audio=str(anchor), ref_text=rec["text"],
                chunk_size=CHUNK_SIZE, temperature=TEMPERATURE, top_k=TOP_K,
            ):
                sr = int(chunk_sr)
                chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            name = f"{scene}__{tag}.wav"
            sf.write(str(run_dir / name), np.concatenate(chunks), sr)
            records.append({"scene": scene, "anchor": anchor.relative_to(ROOT).as_posix(), "text": cfg["text"], "file": name})
            print(name, flush=True)

    (run_dir / "manifest.json").write_text(
        json.dumps({"voice": args.voice, "model": MODEL_DIR.name, "records": records}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    def player(path: Path, caption: str = "") -> str:
        src = Path(os.path.relpath(path, run_dir)).as_posix()
        cap = f"<small>{html.escape(caption)}</small>" if caption else ""
        return f"{cap}<audio controls preload='none' src='{src}'></audio>"

    head = "".join(f"<th>A：锚点 {html.escape(a.stem)}{player(a)}</th>" for a in anchors)
    rows = []
    for scene, cfg in scenes.items():
        cells = "".join(f"<td>{player(run_dir / f'{scene}__{a.stem}.wav')}</td>" for a in anchors)
        b = "".join(player(t, t.stem) for t in takes.get(scene, [])) or "<small>—</small>"
        rows.append(
            f"<tr><td><b>{html.escape(cfg['label'])}</b> <small>{scene}</small><br><small>{html.escape(cfg.get('hint', ''))}</small>"
            f"<p>{html.escape(cfg['text'])}</p></td>{cells}<td>{b}</td></tr>"
        )
    page = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><title>{html.escape(voice['label'])} 锚点对照</title>
<style>
 body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 16px; background:#faf8f6; color:#2b2b2b; }}
 table {{ border-collapse: collapse; background:#fff; }}
 th, td {{ border: 1px solid #e6e0da; padding: 8px; vertical-align: top; font-size: 14px; }}
 th {{ background: #f0ece8; text-align: left; }}
 td:first-child {{ max-width: 260px; }}
 p {{ margin: 6px 0 0; color: #555; }}
 small {{ color: #888; display: block; }}
 audio {{ width: 230px; display: block; margin: 2px 0 6px; }}
</style></head><body>
<h1>{html.escape(voice['label'])}：锚点克隆（A） × 逐情绪设计（B）</h1>
<p>A 列：0.6B Base 以表头锚点为参考克隆同一句台词（服务同参数、不加指令）。B 列：VoiceDesign 直接出的候选。
听两点：还是不是锚点那个人；情绪到不到位。</p>
<table><tr><th>场景</th>{head}<th>B：VoiceDesign 候选</th></tr>{''.join(rows)}</table>
</body></html>
"""
    (run_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"\n试听页：{run_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
