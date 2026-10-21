"""参考片 × 模型 克隆试听网格（AiriLife M53-07，纯人耳选择）。

背景：说话人余弦在负对照里分不出人（流萤对梓涵锚点也有 0.97+），不能用来选参考。
本脚本只出试听材料：同一组台词，分别以 zihan_v2 的每段 VoiceDesign 原片为 ICL 参考，
用 0.6B / 1.7B Base 克隆（服务同参数流式），页面按「参考片 × 模型」排表，不标任何分数。

用法（conda env qwen3-tts-fast，先停掉常驻服务腾显存）：
    python tools/qwen/reference_listening_grid.py
    python tools/qwen/reference_listening_grid.py --models 0.6b --refs daily comfort --run-name quick
"""

from __future__ import annotations

import argparse
import gc
import html
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QWEN_DIR = ROOT / "models" / "base" / "qwen3-tts"
MODELS = {
    "0.6b": QWEN_DIR / "Qwen3-TTS-12Hz-0.6B-Base",
    "1.7b": QWEN_DIR / "Qwen3-TTS-12Hz-1.7B-Base",
}
DESIGN_DIR = ROOT / "models" / "zelia" / "qwen_v2"
PROMPTS_FILE = ROOT / "voices" / "qwen3_voice_prompts.json"
OUT_ROOT = ROOT / "output" / "reference_grid"

# 与 tools/serve/qwen_tts_server.py 保持一致
TEMPERATURE = 0.9
TOP_K = 50
CHUNK_SIZE = 4

# 覆盖日常 / 开心 / 难过 / 生气，且都不是任何参考片的原台词
TEXTS = [
    "你回来啦？我刚泡了茶，要不要一起喝一点？",
    "真的吗！那周末我们就去那家新开的甜品店吧，我想吃草莓的那个！",
    "嗯……我知道你很忙。只是今天，有一点点想你了。",
    "你又熬到三点才睡！说了多少次了，身体是你自己的呀。",
]


def references() -> list[tuple[str, Path, str]]:
    scenes = json.loads(PROMPTS_FILE.read_text(encoding="utf-8"))["zihan_v2"]["scenes"]
    refs = []
    for key, scene in scenes.items():
        wav = DESIGN_DIR / f"{key}_t1.wav"
        if wav.exists():
            refs.append((key, wav, scene["text"]))
    return refs


def run_model(key: str, refs, texts, run_dir: Path) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"\n== {key}: 加载 {MODELS[key].name}", flush=True)
    model = FasterQwen3TTS.from_pretrained(
        str(MODELS[key]), device="cuda", dtype=torch.bfloat16, backend="torch", local_files_only=True
    )
    model.warmup()
    records = []
    for ref_key, ref_wav, ref_text in refs:
        for ti, text in enumerate(texts, 1):
            chunks, sr = [], 24000
            for audio, chunk_sr, _timing in model.generate_voice_clone_streaming(
                text=text, language="Chinese", ref_audio=str(ref_wav), ref_text=ref_text,
                chunk_size=CHUNK_SIZE, temperature=TEMPERATURE, top_k=TOP_K,
            ):
                sr = int(chunk_sr)
                chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
            name = f"{key}_{ref_key}_{ti}.wav"
            sf.write(str(run_dir / name), np.concatenate(chunks), sr)
            records.append({"model": key, "ref": ref_key, "text_idx": ti, "file": name})
            print(f"  {name}", flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return records


def write_index(run_dir: Path, refs, texts, models, records) -> Path:
    files = {(r["model"], r["ref"], r["text_idx"]): r["file"] for r in records}
    head = "".join(f"<th>{i}. {html.escape(t)}</th>" for i, t in enumerate(texts, 1))
    rows = []
    for ref_key, ref_wav, _text in refs:
        for m in models:
            cells = "".join(
                f"<td><audio controls preload='none' src='{html.escape(files[(m, ref_key, i)])}'></audio></td>"
                for i in range(1, len(texts) + 1)
            )
            orig = (
                f"<audio controls preload='none' src='../../../{ref_wav.relative_to(ROOT).as_posix()}'></audio>"
                if m == models[0] else ""
            )
            rows.append(f"<tr><td><b>{html.escape(ref_key)}</b><br>{orig}</td><td>{m}</td>{cells}</tr>")
    page = f"""<!doctype html><meta charset="utf-8"><title>参考片试听网格</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:6px;vertical-align:top;font-weight:normal;font-size:13px}}audio{{width:200px}}</style>
<h1>梓涵 v2：哪段原片做参考，克隆最像</h1>
<p>每行 = 一段 v2 原片作参考 × 一个模型；第一列可听原片。只凭听感挑，页面不给分数。</p>
<table><tr><th>参考片</th><th>模型</th>{head}</tr>{''.join(rows)}</table>"""
    index = run_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["0.6b", "1.7b"], choices=sorted(MODELS))
    ap.add_argument("--refs", nargs="+", default=None, help="只用指定场景 key 的原片")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    refs = [r for r in references() if not args.refs or r[0] in args.refs]
    run_dir = OUT_ROOT / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(refs)} 段参考 × {args.models} × {len(TEXTS)} 句；输出 {run_dir}", flush=True)

    records = []
    for key in args.models:
        records += run_model(key, refs, TEXTS, run_dir)
    (run_dir / "manifest.json").write_text(json.dumps({
        "params": {"temperature": TEMPERATURE, "top_k": TOP_K, "chunk_size": CHUNK_SIZE, "mode": "icl-streaming"},
        "texts": TEXTS, "refs": [r[0] for r in refs], "records": records}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n试听页：{write_index(run_dir, refs, TEXTS, args.models, records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
