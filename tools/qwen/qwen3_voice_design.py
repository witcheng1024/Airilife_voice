"""Qwen3-TTS VoiceDesign：用自然语言描述设计原创音色，批量出片 + 出 HTML 试听页。

只做「设计声音」这一段：给一句 prompt + 若干场景台词，生成 wav 候选。
不加载 Base 模型、不做流式，因此不会碰现有 GPT-SoVITS 服务。

用法（在 qwen3-tts 环境里跑）：
    # 默认：梓涵 prompt，全部 7 个场景，落 output/qwen3_voice_design
    python tools/qwen/qwen3_voice_design.py

    # 只出某几个场景
    python tools/qwen/qwen3_voice_design.py --scenes daily bedtime

    # 只生一条自由文本
    python tools/qwen/qwen3_voice_design.py --voice zihan --text "嗯？你回来啦。" --tag free

    # 同一条 prompt 出多个随机种子，挑最像人声的那条
    python tools/qwen/qwen3_voice_design.py --scenes daily --take 3

输出：
    output/qwen3_voice_design/<voice>/<date>_<run>/<scene>_t<take>.wav
    output/qwen3_voice_design/<voice>/<date>_<run>/manifest.json    每条的 prompt/耗时/显存
    output/qwen3_voice_design/<voice>/<date>_<run>/index.html       浏览器直接听完对比
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROMPTS_FILE = ROOT / "voices" / "qwen3_voice_prompts.json"
DEFAULT_MODEL = ROOT / "models" / "base" / "qwen3-tts" / "Qwen3-TTS-12Hz-1.7B-VoiceDesign"
OUT_ROOT = ROOT / "output" / "qwen3_voice_design"


def load_prompts() -> dict:
    with PROMPTS_FILE.open(encoding="utf-8") as f:
        data = json.load(f)
    return {k: v for k, v in data.items() if not k.startswith("_")}


def load_model(model_dir: Path, dtype: str, device: str):
    import torch
    from qwen_tts import Qwen3TTSModel

    dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    if not model_dir.exists():
        raise SystemExit(
            f"模型目录不存在：{model_dir}\n先跑：python tools/fetch_qwen3_tts_models.py --only voice-design"
        )
    t0 = time.perf_counter()
    model = Qwen3TTSModel.from_pretrained(str(model_dir), device_map=device, dtype=dtypes[dtype])
    load_s = time.perf_counter() - t0
    return model, load_s


def vram_report(torch) -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {
        "torch_allocated_mb": round(torch.cuda.memory_allocated() / 1024**2, 1),
        "torch_reserved_mb": round(torch.cuda.memory_reserved() / 1024**2, 1),
        "device_used_mb": round((total - free) / 1024**2, 1),
        "device_total_mb": round(total / 1024**2, 1),
    }


def write_index(out_dir: Path, voice: str, instruct: str, records: list[dict]) -> Path:
    rows = []
    for r in records:
        rows.append(
            f"""<tr>
  <td>{r['scene']}<br><small>{r['scene_label']}</small></td>
  <td>{r['text']}</td>
  <td>{r['audio_sec']:.2f}s</td>
  <td>{r['gen_sec']:.2f}s</td>
  <td>RTF {r['rtf']:.2f}</td>
  <td><audio controls preload="none" src="{r['file']}"></audio></td>
</tr>"""
        )
    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>Qwen3-TTS VoiceDesign · {voice}</title>
<style>
 body {{ font-family: -apple-system, "Microsoft YaHei", sans-serif; margin: 24px; background:#faf8f6; color:#2b2b2b; }}
 h1 {{ font-size: 20px; }}
 table {{ border-collapse: collapse; width: 100%; background:#fff; }}
 th, td {{ border-bottom:1px solid #eee; padding:8px 10px; text-align:left; vertical-align:top; font-size:14px; }}
 th {{ background:#f0ece8; }}
 audio {{ width: 260px; }}
 details {{ margin: 12px 0 20px; }}
 pre {{ white-space: pre-wrap; background:#fff; padding:12px; border:1px solid #eee; font-size:12px; }}
 small {{ color:#888; }}
</style></head><body>
<h1>Qwen3-TTS VoiceDesign · {voice}</h1>
<details><summary>音色提示词（instruct）</summary><pre>{instruct}</pre></details>
<table>
<thead><tr><th>场景</th><th>台词</th><th>音频时长</th><th>生成耗时</th><th>速度</th><th>试听</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
<p><small>RTF = 生成耗时 / 音频时长，小于 1 表示比实时快。此处是离线整句生成，不代表 AiriLife 流式首包延迟。</small></p>
</body></html>
"""
    path = out_dir / "index.html"
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--voice", default="zihan", help="voices/qwen3_voice_prompts.json 里的音色 key")
    parser.add_argument("--scenes", nargs="+", help="只跑指定场景 key；不给则跑全部")
    parser.add_argument("--text", help="直接给一句台词（配合 --tag），忽略场景表")
    parser.add_argument("--tag", default="free", help="--text 时的场景名/文件名前缀")
    parser.add_argument("--model", default=str(DEFAULT_MODEL), help="VoiceDesign 模型目录")
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--take", type=int, default=1, help="每条台词出几个随机采样（挑最像的）")
    parser.add_argument("--seed", type=int, default=None, help="固定随机种子，便于逐日对比")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--temperature", type=float, default=None, help="覆盖模型默认采样温度")
    parser.add_argument("--top-p", dest="top_p", type=float, default=None)
    parser.add_argument("--run-name", help="输出子目录名，默认 <日期>_<序号>")
    args = parser.parse_args()

    voices = load_prompts()
    if args.voice not in voices:
        raise SystemExit(f"未知音色 {args.voice}，可选：{', '.join(voices)}")
    voice_cfg = voices[args.voice]
    instruct = voice_cfg["instruct"]

    # 组装要生成的任务
    jobs: list[dict] = []
    if args.text:
        jobs.append({"scene": args.tag, "label": "自由台词", "hint": "", "text": args.text})
    else:
        scenes = voice_cfg["scenes"]
        keys = args.scenes or list(scenes)
        for k in keys:
            if k not in scenes:
                raise SystemExit(f"未知场景 {k}，可选：{', '.join(scenes)}")
            jobs.append({"scene": k, **scenes[k]})

    import torch
    import soundfile as sf

    if args.seed is not None:
        torch.manual_seed(args.seed)

    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = OUT_ROOT / args.voice / (args.run_name or stamp)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"加载模型：{args.model} ({args.dtype})", flush=True)
    model, load_s = load_model(Path(args.model), args.dtype, args.device)
    print(f"模型加载 {load_s:.1f}s；vram={vram_report(torch)}", flush=True)

    gen_kwargs: dict = {}
    if args.temperature is not None:
        gen_kwargs["temperature"] = args.temperature
    if args.top_p is not None:
        gen_kwargs["top_p"] = args.top_p

    records: list[dict] = []
    sample_rate = None
    for job in jobs:
        for take in range(1, args.take + 1):
            t0 = time.perf_counter()
            wavs, sr = model.generate_voice_design(
                text=job["text"],
                language=args.language,
                instruct=instruct,
                non_streaming_mode=True,
                **gen_kwargs,
            )
            gen_s = time.perf_counter() - t0
            wav = wavs[0]
            audio_sec = len(wav) / sr
            name = f"{job['scene']}_t{take}.wav"
            sf.write(str(run_dir / name), wav, sr)
            rec = {
                "scene": job["scene"],
                "scene_label": f"{job['label']}｜{job['hint']}" if job.get("hint") else job["label"],
                "text": job["text"],
                "file": name,
                "sample_rate": sr,
                "audio_sec": round(audio_sec, 3),
                "gen_sec": round(gen_s, 3),
                "rtf": round(gen_s / max(audio_sec, 1e-6), 3),
                "vram": vram_report(torch),
            }
            records.append(rec)
            print(f"  [{job['scene']} t{take}] {audio_sec:.2f}s 音频 / {gen_s:.2f}s 生成  RTF {rec['rtf']}", flush=True)

    manifest = {
        "voice": args.voice,
        "model": str(args.model),
        "dtype": args.dtype,
        "instruct": instruct,
        "sample_rate": sample_rate,
        "model_load_sec": round(load_s, 3),
        "run_dir": str(run_dir.relative_to(ROOT)),
        "records": records,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    index = write_index(run_dir, voice_cfg.get("label", args.voice), instruct, records)

    print(f"\n完成 {len(records)} 条 → {run_dir}")
    print(f"试听页：{index}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
