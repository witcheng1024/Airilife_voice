"""用 Faster Qwen3-TTS 复现设计好的音色（Base 模型克隆），并测音色一致性。

定位：VoiceDesign 是「创作器」，Base 是「复现器」。设计阶段产出大量候选 wav，
选定后用 Base 克隆，才能在日常合成里稳定拿到同一条声线。

本脚本做三件事：
  1. 用 Base 模型克隆参考音频（ICL 模式：ref_audio + ref_text，最接近原声色）、
     xvec 模式（只用说话人嵌入）各跑一遍，对比哪种更稳。
  2. 用同一句参考台词重复生成 N 次，统计音色漂移（说话人嵌入余弦相似度），
     这是「音色一致性」的可量化指标，比纯主观听感更早发现问题。
  3. 出 wav + manifest.json + index.html 试听页。

用法：
    python tools/qwen/qwen3_base_clone_check.py --ref output/qwen3_voice_design/zihan/20261020_013207/daily_t1.wav
    python tools/qwen/qwen3_base_clone_check.py --ref ... --model 0.6b --reps 3 --scenes daily bedtime
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
QWEN_DIR = ROOT / "models" / "base" / "qwen3-tts"
PROMPTS_FILE = ROOT / "voices" / "qwen3_voice_prompts.json"
OUT_ROOT = ROOT / "output" / "qwen3_base_clone"

MODELS = {
    "1.7b": QWEN_DIR / "Qwen3-TTS-12Hz-1.7B-Base",
    "0.6b": QWEN_DIR / "Qwen3-TTS-12Hz-0.6B-Base",
}


def load_scenes(voice: str) -> dict:
    with PROMPTS_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    return data[voice]["scenes"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="参考音频 wav 路径")
    ap.add_argument("--ref-text", required=True, help="参考音频对应的台词（ICL 模式必需，必须逐字一致）")
    ap.add_argument("--voice", default="zihan", help="用于取场景台词的音色 key")
    ap.add_argument("--scenes", nargs="+", default=["daily", "comfort", "bedtime"])
    ap.add_argument("--model", default="1.7b", choices=sorted(MODELS))
    ap.add_argument("--reps", type=int, default=3, help="同一句重复生成次数，用于测音色一致性")
    ap.add_argument("--modes", nargs="+", default=["icl"], choices=["icl", "xvec"])
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--run-name", default=None)
    args = ap.parse_args()

    import numpy as np
    import soundfile as sf
    import torch
    from faster_qwen3_tts import FasterQwen3TTS

    ref = Path(args.ref)
    if not ref.exists():
        raise SystemExit(f"参考音频不存在：{ref}")
    scenes = load_scenes(args.voice)
    model_dir = MODELS[args.model]

    run_dir = OUT_ROOT / args.voice / (args.run_name or time.strftime("%Y%m%d_%H%M%S"))
    run_dir.mkdir(parents=True, exist_ok=True)

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    print(f"加载 {model_dir.name} ...", flush=True)
    t0 = time.perf_counter()
    model = FasterQwen3TTS.from_pretrained(
        str(model_dir), device="cuda", dtype=dtype, backend="torch", local_files_only=True
    )
    load_s = time.perf_counter() - t0
    print(f"加载 {load_s:.1f}s", flush=True)
    model.warmup()

    # 说话人嵌入提取器：直接拿底层模型的方法算，用于一致性打分。
    # FasterQwen3TTS.model -> Qwen3TTSModel（包一层） -> .model -> Qwen3TTSForConditionalGeneration
    def speaker_embedding(wav_path: Path) -> np.ndarray:
        wav, sr = sf.read(str(wav_path), dtype="float32")
        inner = getattr(model, "model", model)
        inner = getattr(inner, "model", inner)
        if sr != inner.speaker_encoder_sample_rate:
            import librosa

            wav = librosa.resample(
                y=wav.astype(np.float32), orig_sr=sr, target_sr=inner.speaker_encoder_sample_rate
            )
            sr = inner.speaker_encoder_sample_rate
        emb = inner.extract_speaker_embedding(audio=wav, sr=sr)
        if torch.is_tensor(emb):
            emb = emb.detach().float().cpu().numpy()
        return np.asarray(emb).reshape(-1)

    ref_emb = speaker_embedding(ref)

    def cosine(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    records = []
    for mode in args.modes:
        xvec = mode == "xvec"
        for key in args.scenes:
            text = scenes[key]["text"]
            for rep in range(1, args.reps + 1):
                t0 = time.perf_counter()
                wavs, sr = model.generate_voice_clone(
                    text=text,
                    language="Chinese",
                    ref_audio=str(ref),
                    ref_text="" if xvec else args.ref_text,
                    xvec_only=xvec,
                )
                gen_s = time.perf_counter() - t0
                wav = wavs[0]
                name = f"{mode}_{key}_r{rep}.wav"
                sf.write(str(run_dir / name), wav, sr)
                emb = speaker_embedding(run_dir / name)
                audio_sec = len(wav) / sr
                rec = {
                    "mode": mode,
                    "scene": key,
                    "scene_label": f"{scenes[key]['label']}｜{scenes[key]['hint']}",
                    "rep": rep,
                    "text": text,
                    "file": name,
                    "audio_sec": round(audio_sec, 3),
                    "gen_sec": round(gen_s, 3),
                    "rtf": round(gen_s / max(audio_sec, 1e-6), 3),
                    "spk_cos_to_ref": round(cosine(emb, ref_emb), 5),
                }
                records.append(rec)
                print(
                    f"  {mode:4s} {key:8s} r{rep}  {audio_sec:5.2f}s/ {gen_s:5.2f}s  "
                    f"音色相似度 {rec['spk_cos_to_ref']:.4f}",
                    flush=True,
                )

    # 一致性统计：同一 mode+scene 内，各次生成彼此之间的相似度（越低越漂）
    summary = {}
    for mode in args.modes:
        for key in args.scenes:
            embs = [speaker_embedding(run_dir / f"{mode}_{key}_r{r}.wav") for r in range(1, args.reps + 1)]
            pairs = [
                cosine(embs[i], embs[j])
                for i in range(len(embs))
                for j in range(i + 1, len(embs))
            ]
            summary[f"{mode}/{key}"] = {
                "self_similarity_mean": round(float(np.mean(pairs)), 5) if pairs else None,
                "self_similarity_min": round(float(np.min(pairs)), 5) if pairs else None,
                "cos_to_ref_mean": round(
                    float(np.mean([cosine(e, ref_emb) for e in embs])), 5
                ),
            }

    manifest = {
        "voice": args.voice,
        "model": str(model_dir),
        "ref_audio": str(ref),
        "ref_text": args.ref_text,
        "model_load_sec": round(load_s, 3),
        "records": records,
        "consistency": summary,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    rows = "".join(
        f"<tr><td>{r['mode']}</td><td>{r['scene_label']}</td><td>{r['text']}</td>"
        f"<td>{r['audio_sec']:.2f}s</td><td>{r['rtf']:.2f}</td>"
        f"<td>{r['spk_cos_to_ref']:.4f}</td>"
        f"<td><audio controls preload='none' src='{r['file']}'></audio></td></tr>"
        for r in records
    )
    cons = "".join(
        f"<tr><td>{k}</td><td>{v['self_similarity_mean']}</td><td>{v['self_similarity_min']}</td>"
        f"<td>{v['cos_to_ref_mean']}</td></tr>"
        for k, v in summary.items()
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<title>Base 克隆一致性 · {args.voice}</title>
<style>body{{font-family:-apple-system,"Microsoft YaHei",sans-serif;margin:24px;background:#faf8f6}}
table{{border-collapse:collapse;width:100%;background:#fff;margin-bottom:24px}}
th,td{{border-bottom:1px solid #eee;padding:8px 10px;font-size:14px;text-align:left;vertical-align:top}}
th{{background:#f0ece8}} audio{{width:240px}} small{{color:#888}}</style></head><body>
<h1>Base 克隆一致性 · {args.voice}</h1>
<p><small>参考音频：{ref.name}　模型：{model_dir.name}　每句重复 {args.reps} 次</small></p>
<h2>生成结果</h2><table><thead><tr><th>模式</th><th>场景</th><th>台词</th><th>音频</th><th>RTF</th>
<th>与参考音色余弦</th><th>试听</th></tr></thead><tbody>{rows}</tbody></table>
<h2>一致性统计</h2>
<p><small>self_similarity = 同一场景多次生成彼此之间的音色余弦（越接近 1 越稳）；cos_to_ref = 与参考音频的相似度。</small></p>
<table><thead><tr><th>组合</th><th>自相似均值</th><th>自相似最低</th><th>与参考相似均值</th></tr></thead>
<tbody>{cons}</tbody></table>
</body></html>
"""
    (run_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\n完成 {len(records)} 条 → {run_dir}")
    print(f"试听页：{run_dir / 'index.html'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
