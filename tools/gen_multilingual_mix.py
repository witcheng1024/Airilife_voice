#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 CosyVoice3 合成"代码切换"（中英/中日混合）音频 —— firefly 音色。

输入：output/multilingual/mix_sources.json（gen_codeswitch.py 产物，zh 文本内含英文/日文短语）
输出：data/firefly_multilingual/mix/mix_<n>.wav + _mix_manifest.json
     （训练时语言字段用 zh，LangSegment 自动识别嵌入的外语部分）

用法（必须用 .venv，等 EN/JA 主生成完再跑，避免显存不够）:
  E:/PROJECT/Airilife_voice/.venv/Scripts/python.exe tools/gen_multilingual_mix.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
COSYVOICE_ROOT = Path(r"E:/PROJECT/Airilife_voice/CosyVoice")
MODEL_DIR = Path(r"E:/PROJECT/Airilife_voice/pretrained_models/Fun-CosyVoice3-0.5B")
REF_AUDIO = ROOT / "reference_audio" / "活泼.wav"
SRC = ROOT / "output" / "multilingual" / "mix_sources.json"
OUT_DIR = ROOT / "data" / "firefly_multilingual" / "mix"

CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(CACHE / "numba"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
sys.path.insert(0, str(COSYVOICE_ROOT))
sys.path.insert(0, str(COSYVOICE_ROOT / "third_party" / "Matcha-TTS"))

INSTRUCT = "You are a helpful assistant. 请自然、口语化地说这句话，其中外语部分用对应语言发音。<|endofprompt|>"


def main() -> None:
    import soundfile as sf
    from cosyvoice.cli.cosyvoice import AutoModel

    items = json.loads(SRC.read_text(encoding="utf-8"))
    print(f"待合成: {len(items)} 条代码切换", flush=True)

    if torch.cuda.is_available():
        torch.cuda.init()
        torch.zeros(1, device="cuda")
        torch.cuda.synchronize()
    print("加载 CosyVoice3...", flush=True)
    cv = AutoModel(model_dir=str(MODEL_DIR), fp16=True)
    sr = cv.sample_rate
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"ref_audio": str(REF_AUDIO), "sr": sr, "items": []}
    ok = 0
    for i, item in enumerate(items):
        out_path = OUT_DIR / f"mix_{i + 1}.wav"
        if out_path.exists():
            continue
        t1 = time.time()
        try:
            chunks = []
            gen = cv.inference_instruct2(
                tts_text=item["zh"], instruct_text=INSTRUCT, prompt_wav=str(REF_AUDIO), stream=False,
            )
            for out in gen:
                torch.cuda.synchronize()
                chunks.append(out["tts_speech"])
            audio = torch.cat(chunks, dim=1)
            peak = float(torch.max(torch.abs(audio)).item())
            if peak > 0.95:
                audio = audio * (0.95 / peak)
        except Exception as e:
            print(f"[{i}] 失败 {type(e).__name__}: {e}", flush=True)
            continue
        dur = audio.shape[1] / sr
        peak = float(torch.max(torch.abs(audio)).item())
        if dur < 0.5 or peak < 0.01:
            print(f"[{i}] BAD(dur={dur:.2f}s) 跳过", flush=True)
            continue
        sf.write(str(out_path), audio.cpu().numpy().reshape(-1), sr)
        manifest["items"].append({"i": i, "type": item["type"], "text": item["zh"],
                                  "path": str(out_path), "dur_s": round(dur, 3), "status": "OK"})
        ok += 1
        print(f"[{i}] {dur:.2f}s ({time.time() - t1:.1f}s) → {out_path.name}", flush=True)

    out_manifest = ROOT / "data" / "firefly_multilingual" / "_mix_manifest.json"
    out_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成: {ok} 条 → {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
