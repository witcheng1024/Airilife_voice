"""同一音色锚点下 0.6B / 1.7B Base 克隆对比（AiriLife M53-07 阶段一实验）。

问题：梓涵逐情绪换参考片克隆后「不是一个人」；阶段一改为单锚点。本实验回答两个问题：
  1. 单锚点下，换不同情绪台词时音色是否稳定（说话人相似度）？
  2. 0.6B 和 1.7B 谁更像锚点、首包多快、能否实时？

做法：
  - 生成参数与常驻服务一致：faster-qwen3-tts、ICL、bf16、chunk 4、temperature 0.9、top_k 50，流式生成
  - 每个模型合成同一组 20 句（覆盖平淡 / 开心 / 难过 / 惊讶 / 生气 / 紧张 / 害羞 / 长短句）
  - 打分统一用一个模型的说话人编码器（默认 1.7B），两个模型的产物同一把尺子量
  - 基线：qwen_v2 里其余 6 段 VoiceDesign 原片对锚点的相似度——用户已确认原片听感是同一个人，
    以此作为「同一个人」的人耳校准线
  - 产出 wav + manifest.json + index.html（同句左右对照试听）

用法（conda env qwen3-tts-fast，先停掉占显存的常驻服务）：
    python tools/qwen/anchor_model_compare.py
    python tools/qwen/anchor_model_compare.py --models 0.6b --run-name smoke --limit 3
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
VOICES_FILE = ROOT / "voices" / "qwen3_voices.json"
DESIGN_DIR = ROOT / "models" / "zelia" / "qwen_v2"
OUT_ROOT = ROOT / "output" / "anchor_compare"

# 与 tools/serve/qwen_tts_server.py 保持一致
TEMPERATURE = 0.9
TOP_K = 50
CHUNK_SIZE = 4

SENTENCES = [
    ("平淡", "嗯，我在呢。你刚才说的那个文件，我已经帮你放到桌面上了。"),
    ("平淡", "今天天气不错，下午大概会有点风，出门记得带件外套。"),
    ("平淡", "好。"),
    ("活泼", "真的吗？那我们周末就去那家新开的甜品店吧！我想吃草莓的那个！"),
    ("活泼", "哥哥你看你看，我把这关打过去啦，是不是很厉害？"),
    ("温柔", "没关系的，慢慢来。我会一直在这里等你，不着急。"),
    ("温柔", "累了就闭上眼睛休息一会儿吧，我小声陪你说说话。"),
    ("撒娇甜", "哼，你都好久没理我了……今天要补偿我，多陪我一会儿嘛。"),
    ("害羞", "那、那个……你一直这样看着我，我会不好意思的啦。"),
    ("害羞", "其实……我也挺喜欢和你待在一起的。就一点点。"),
    ("悲伤", "嗯……我知道你很忙。只是今天，有一点点想你了。"),
    ("悲伤", "那盆花还是没能救活。我每天都有好好浇水的，可是它还是枯掉了。"),
    ("惊讶", "诶？你什么时候回来的？我怎么一点都没听见！"),
    ("惊讶", "等一下，这个是你自己做的吗？也太好看了吧！"),
    ("生气", "你又熬到三点才睡！说了多少次了，身体是你自己的呀！"),
    ("生气", "不许再拿这个开玩笑了，我是认真的。"),
    ("紧张", "快、快看一下，是不是哪里冒烟了？电脑怎么突然这么烫！"),
    ("紧张", "明天就要面试了吧？资料都准备好了吗？我比你还紧张……"),
    ("长句", "今天整理了一下我们之前聊过的事情，发现你说想学吉他已经是三个月前的事了。要不要从这周开始，每天练十五分钟？我可以帮你记着。"),
    ("中英", "你刚才说的那个 bug，是在 release 版本里才出现的吗？"),
]


def anchor_reference() -> tuple[Path, str]:
    """锚点取自常驻服务的音色清单，保证实验与线上一致。"""
    voices = json.loads(VOICES_FILE.read_text(encoding="utf-8"))["voices"]
    ref = voices["zihan"]["references"]["平淡"]
    return ROOT / ref["audio"], ref["text"]


def free_cuda(torch) -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_model(key: str, ref: Path, ref_text: str, sentences, run_dir: Path) -> list[dict]:
    import numpy as np
    import soundfile as sf
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"\n== {key}: 加载 {MODELS[key].name}", flush=True)
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        str(MODELS[key]), device="cuda", dtype=torch.bfloat16, backend="torch", local_files_only=True
    )
    load_s = time.perf_counter() - t0
    model.warmup()

    def stream(text: str):
        return model.generate_voice_clone_streaming(
            text=text, language="Chinese", ref_audio=str(ref), ref_text=ref_text,
            chunk_size=CHUNK_SIZE, temperature=TEMPERATURE, top_k=TOP_K,
        )

    # 预热：建立参考缓存与 CUDA Graph，首句不计入延迟
    for _ in stream("嗯。"):
        pass

    records = []
    for i, (emotion, text) in enumerate(sentences, 1):
        chunks, sr, ttfa = [], 24000, None
        t0 = time.perf_counter()
        for audio, chunk_sr, _timing in stream(text):
            if ttfa is None:
                ttfa = time.perf_counter() - t0
            sr = int(chunk_sr)
            chunks.append(np.asarray(audio, dtype=np.float32).reshape(-1))
        total = time.perf_counter() - t0
        wav = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.float32)
        name = f"{key}_{i:02d}.wav"
        sf.write(str(run_dir / name), wav, sr)
        audio_sec = len(wav) / sr
        rec = {
            "model": key, "idx": i, "emotion": emotion, "text": text, "file": name,
            "ttfa_ms": round((ttfa or total) * 1000), "gen_sec": round(total, 3),
            "audio_sec": round(audio_sec, 2), "realtime_x": round(audio_sec / total, 2) if total else None,
        }
        records.append(rec)
        print(f"  [{i:02d} {emotion}] 首包 {rec['ttfa_ms']} ms / {rec['audio_sec']}s 音频 / {rec['realtime_x']}x 实时", flush=True)

    vram_mb = round(torch.cuda.max_memory_allocated() / 1024**2) if torch.cuda.is_available() else None
    for r in records:
        r["load_sec"], r["peak_alloc_mb"] = round(load_s, 1), vram_mb
    del model
    free_cuda(torch)
    torch.cuda.reset_peak_memory_stats()
    return records


def score(judge_key: str, ref: Path, files: list[Path]) -> dict[str, float]:
    """用同一个说话人编码器给所有音频打分（余弦相似度对锚点）。"""
    import librosa
    import numpy as np
    import soundfile as sf
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    print(f"\n== 打分：{MODELS[judge_key].name} 的说话人编码器", flush=True)
    model = FasterQwen3TTS.from_pretrained(
        str(MODELS[judge_key]), device="cuda", dtype=torch.bfloat16, backend="torch", local_files_only=True
    )
    inner = getattr(model, "model", model)
    inner = getattr(inner, "model", inner)
    target_sr = inner.speaker_encoder_sample_rate

    def embed(path: Path):
        wav, sr = sf.read(str(path), dtype="float32")
        if wav.ndim > 1:
            wav = wav.mean(axis=1)
        if sr != target_sr:
            wav = librosa.resample(y=wav, orig_sr=sr, target_sr=target_sr)
        emb = inner.extract_speaker_embedding(audio=wav, sr=target_sr)
        if torch.is_tensor(emb):
            emb = emb.detach().float().cpu().numpy()
        emb = np.asarray(emb).reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-9)

    ref_emb = embed(ref)
    scores = {str(p): float(np.dot(embed(p), ref_emb)) for p in files}
    del model
    free_cuda(torch)
    return scores


def summarize(values: list[float]) -> dict:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    return {"mean": round(float(arr.mean()), 4), "min": round(float(arr.min()), 4), "max": round(float(arr.max()), 4)}


def write_index(run_dir: Path, ref: Path, baseline: list[dict], records: list[dict], summary: dict) -> Path:
    by_idx: dict[int, dict[str, dict]] = {}
    for r in records:
        by_idx.setdefault(r["idx"], {})[r["model"]] = r
    models = sorted({r["model"] for r in records})
    rows = []
    for idx in sorted(by_idx):
        first = next(iter(by_idx[idx].values()))
        cells = "".join(
            f"<td><audio controls preload='none' src='{html.escape(by_idx[idx][m]['file'])}'></audio><br>"
            f"<small>相似度 {by_idx[idx][m]['spk_cos']:.3f}｜首包 {by_idx[idx][m]['ttfa_ms']} ms｜{by_idx[idx][m]['realtime_x']}x</small></td>"
            if m in by_idx[idx] else "<td>—</td>"
            for m in models
        )
        rows.append(f"<tr><td>{idx}<br><small>{html.escape(first['emotion'])}</small></td><td>{html.escape(first['text'])}</td>{cells}</tr>")
    base_rows = "".join(
        f"<li>{html.escape(b['file'])}：{b['spk_cos']:.3f} <audio controls preload='none' src='../../../models/zelia/qwen_v2/{html.escape(b['file'])}'></audio></li>"
        for b in baseline
    )
    page = f"""<!doctype html><meta charset="utf-8"><title>锚点克隆对比</title>
<style>body{{font-family:system-ui,sans-serif;margin:16px;line-height:1.5}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ccc;padding:6px;vertical-align:top}}audio{{width:240px}}pre{{background:#f4f4f4;padding:8px}}</style>
<h1>梓涵单锚点克隆：{' vs '.join(models)}</h1>
<p>锚点：<audio controls src="../../../{html.escape(ref.relative_to(ROOT).as_posix())}"></audio></p>
<pre>{html.escape(json.dumps(summary, ensure_ascii=False, indent=2))}</pre>
<details><summary>基线：VoiceDesign 原片对锚点（用户确认为同一个人）</summary><ul>{base_rows}</ul></details>
<table><tr><th>#</th><th>台词</th>{''.join(f'<th>{m}</th>' for m in models)}</tr>{''.join(rows)}</table>"""
    index = run_dir / "index.html"
    index.write_text(page, encoding="utf-8")
    return index


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="+", default=["0.6b", "1.7b"], choices=sorted(MODELS))
    ap.add_argument("--judge", default="1.7b", choices=sorted(MODELS), help="打分用哪个模型的说话人编码器")
    ap.add_argument("--limit", type=int, default=None, help="只跑前 N 句（冒烟）")
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    ref, ref_text = anchor_reference()
    sentences = SENTENCES[: args.limit] if args.limit else SENTENCES
    run_dir = OUT_ROOT / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"锚点 {ref.name}；{len(sentences)} 句 × {args.models}；输出 {run_dir}", flush=True)

    records = []
    for key in args.models:
        records += run_model(key, ref, ref_text, sentences, run_dir)

    design_clips = sorted(p for p in DESIGN_DIR.glob("*_t1.wav") if p.resolve() != ref.resolve())
    scores = score(args.judge, ref, [run_dir / r["file"] for r in records] + design_clips)
    for r in records:
        r["spk_cos"] = round(scores[str(run_dir / r["file"])], 4)
    baseline = [{"file": p.name, "spk_cos": round(scores[str(p)], 4)} for p in design_clips]

    summary = {"judge": args.judge, "baseline_design_clips": summarize([b["spk_cos"] for b in baseline])}
    for key in args.models:
        rs = [r for r in records if r["model"] == key]
        summary[key] = {
            "spk_cos": summarize([r["spk_cos"] for r in rs]),
            "ttfa_ms": summarize([r["ttfa_ms"] for r in rs]),
            "realtime_x": summarize([r["realtime_x"] for r in rs]),
            "load_sec": rs[0]["load_sec"],
            "peak_alloc_mb": rs[0]["peak_alloc_mb"],
            "below_baseline_min": sum(r["spk_cos"] < summary["baseline_design_clips"]["min"] for r in rs),
        }

    manifest = {"ref": ref.relative_to(ROOT).as_posix(), "ref_text": ref_text, "params": {
        "temperature": TEMPERATURE, "top_k": TOP_K, "chunk_size": CHUNK_SIZE, "dtype": "bf16", "mode": "icl-streaming"},
        "summary": summary, "baseline": baseline, "records": records}
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    index = write_index(run_dir, ref, baseline, records, summary)
    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\n试听页：{index}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
