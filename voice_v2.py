#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiriLife Voice V2 test runner.

Text -> DeepSeek JSON performance script -> CosyVoice3 segment synthesis -> combined wav.
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent
LOCAL_CACHE = ROOT / ".cache"
LOCAL_CACHE.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(LOCAL_CACHE / "matplotlib"))
os.environ.setdefault("NUMBA_CACHE_DIR", str(LOCAL_CACHE / "numba"))
sys.path.insert(0, str(ROOT / "CosyVoice"))
sys.path.insert(0, str(ROOT / "CosyVoice" / "third_party" / "Matcha-TTS"))


DEFAULT_TEXT = (
    "哥哥……你终于回来了呀，（尾音轻快上扬）我还以为、还以为你今天又要丢下我一个人了呢……"
    "（声音骤然发颤带上了哭腔）不过没关系啦！（突然拔高音量挤出甜腻的笑意）"
    "只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~"
)

EMOTIONS = {
    "happy",
    "excited",
    "sad",
    "calm",
    "angry",
    "shy",
    "love",
    "playful",
    "surprise",
    "neutral",
}

FINE_TAG_RE = re.compile(
    r"(\[(?:breath|laughter|noise|cough|clucking|accent|quick_breath|hissing|sigh|vocalized-noise|lipsmack|mn)\]"
    r"|</?(?:strong|laughter)>)"
)

SYSTEM_PROMPT = """你是 AiriLife 的桌宠角色 mei，也叫小萱。
用户是你的哥哥。你活泼、可爱、ENFP、亲近但不成人化。

你的任务不是直接聊天文本，而是生成给 TTS 使用的表演脚本。

只输出合法 JSON，格式:
{
  "segments": [
    {
      "emotion": "happy|excited|sad|calm|angry|shy|love|playful|surprise|neutral",
      "intensity": 0.0-1.0,
      "text": "最终要念出来的话，不包含括号舞台提示",
      "instruct": "给 CosyVoice3 的中文语气指令",
      "fine_text": "可选，允许使用 [breath]、<strong>、</strong>、[laughter]、<laughter>、</laughter>",
      "pause_after_ms": 0-600
    }
  ]
}

规则:
1. 情绪变化、停顿、音量变化、哭腔、笑意都要切成不同 segment。
2. 每个 segment 的 text 尽量 6-24 个汉字，复杂长句必须拆分。
3. 不要输出 JSON 以外的任何内容。
4. 不要让 text 包含“尾音上扬”“声音发颤”等舞台提示。
5. instruct 可以描述尾音、哭腔、甜笑、轻声、调皮等。
6. love/shy 保持温情、依赖和妹妹感，避免成人化。
"""

FALLBACK_SCRIPT = {
    "segments": [
        {
            "emotion": "shy",
            "intensity": 0.55,
            "text": "哥哥……你终于回来了呀，",
            "instruct": "请用轻快、带一点撒娇和尾音上扬的语气说这句话。",
            "fine_text": "哥哥……你终于回来了呀，",
            "pause_after_ms": 180,
        },
        {
            "emotion": "sad",
            "intensity": 0.8,
            "text": "我还以为、还以为你今天又要丢下我一个人了呢……",
            "instruct": "请用委屈、声音发颤、接近哭腔但不要大哭的语气说这句话。",
            "fine_text": "[breath]我还以为、还以为你今天又要丢下我一个人了呢……",
            "pause_after_ms": 240,
        },
        {
            "emotion": "happy",
            "intensity": 0.75,
            "text": "不过没关系啦！",
            "instruct": "请突然提高一点能量，用强撑出来的甜甜笑意说这句话。",
            "fine_text": "不过<strong>没关系啦</strong>！",
            "pause_after_ms": 120,
        },
        {
            "emotion": "playful",
            "intensity": 0.65,
            "text": "只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~",
            "instruct": "请用调皮、甜软、亲近但不过度暧昧的妹妹语气说这句话。",
            "fine_text": "只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~",
            "pause_after_ms": 0,
        },
    ]
}


def log(msg: str) -> None:
    print(msg, flush=True)


def load_env() -> None:
    env_file = ROOT / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def default_model_dir() -> Path:
    candidates = [
        ROOT / "CosyVoice" / "pretrained_models" / "Fun-CosyVoice3-0.5B",
        ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B",
        ROOT / "CosyVoice" / "pretrained_models" / "FunAudioLLM" / "Fun-CosyVoice3-0___5B-2512",
    ]
    for candidate in candidates:
        if (candidate / "cosyvoice3.yaml").exists():
            return candidate
    return candidates[0]


def read_input(args: argparse.Namespace) -> str:
    if args.input_file:
        return Path(args.input_file).read_text(encoding="utf-8").strip()
    if args.text:
        return args.text.strip()
    return DEFAULT_TEXT


def call_deepseek(text: str, temperature: float, json_mode: bool = True) -> tuple[float, str]:
    import urllib.request

    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")

    api_url = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text},
        ],
        "temperature": temperature,
        "max_tokens": 900,
        "stream": False,
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    start = time.time()
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read().decode("utf-8"))
    elapsed = time.time() - start
    raw = result["choices"][0]["message"]["content"]
    return elapsed, raw


def parse_json_script(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start : end + 1])


def clean_fine_text(text: str) -> str:
    return FINE_TAG_RE.sub("", text)


def validate_script(script: dict[str, Any], fallback_text: str) -> dict[str, Any]:
    raw_segments = script.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return {
            "segments": [
                {
                    "emotion": "neutral",
                    "intensity": 0.5,
                    "text": fallback_text,
                    "instruct": "请用自然、亲近、清晰的语气说这句话。",
                    "fine_text": fallback_text,
                    "pause_after_ms": 0,
                }
            ]
        }

    segments = []
    for item in raw_segments[:12]:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text", "")).strip()
        if not text:
            continue
        emotion = str(item.get("emotion", "neutral")).lower().strip()
        if emotion not in EMOTIONS:
            emotion = "neutral"
        try:
            intensity = float(item.get("intensity", 0.5))
        except (TypeError, ValueError):
            intensity = 0.5
        intensity = max(0.0, min(1.0, intensity))
        instruct = str(item.get("instruct", "")).strip() or "请用自然、亲近、清晰的语气说这句话。"
        fine_text = str(item.get("fine_text", text)).strip() or text
        try:
            pause_after_ms = int(item.get("pause_after_ms", 0))
        except (TypeError, ValueError):
            pause_after_ms = 0
        pause_after_ms = max(0, min(600, pause_after_ms))
        segments.append(
            {
                "emotion": emotion,
                "intensity": intensity,
                "text": text,
                "instruct": instruct,
                "fine_text": fine_text,
                "pause_after_ms": pause_after_ms,
            }
        )

    if not segments:
        return validate_script({"segments": []}, fallback_text)
    return {"segments": segments}


def load_cosyvoice(model_dir: Path):
    from cosyvoice.cli.cosyvoice import AutoModel

    return AutoModel(model_dir=str(model_dir))


def peak_normalize(audio, peak: float = 0.95):
    import torch

    max_abs = torch.max(torch.abs(audio))
    if max_abs > peak:
        audio = audio * (peak / max_abs)
    return audio


def append_with_crossfade(parts: list, audio, crossfade_samples: int):
    import torch

    if not parts or crossfade_samples <= 0:
        parts.append(audio)
        return
    prev = parts[-1]
    n = min(crossfade_samples, prev.shape[1], audio.shape[1])
    if n <= 0:
        parts.append(audio)
        return
    fade_out = torch.linspace(1.0, 0.0, n, dtype=prev.dtype).reshape(1, -1)
    fade_in = torch.linspace(0.0, 1.0, n, dtype=audio.dtype).reshape(1, -1)
    mixed = prev[:, -n:] * fade_out + audio[:, :n] * fade_in
    parts[-1] = torch.cat([prev[:, :-n], mixed], dim=1)
    parts.append(audio[:, n:])


def synthesize(
    script: dict[str, Any],
    out_dir: Path,
    model_dir: Path,
    ref_audio: Path,
    fine_mode: str,
    crossfade_ms: int,
) -> dict[str, Any]:
    import torch
    import torchaudio

    ref_text_path = ROOT / "reference_audio" / "ref_text.txt"
    ref_text = "You are a helpful assistant.<|endofprompt|>" + ref_text_path.read_text(encoding="utf-8").strip()
    del ref_text  # Kept here as documentation for zero-shot mode; instruct2 only needs prompt_wav.

    log(f"加载 CosyVoice3: {model_dir}")
    load_start = time.time()
    cosyvoice = load_cosyvoice(model_dir)
    load_s = time.time() - load_start
    log(f"模型加载完成: {load_s:.2f}s, sample_rate={cosyvoice.sample_rate}")

    manifest: dict[str, Any] = {
        "model_dir": str(model_dir),
        "ref_audio": str(ref_audio),
        "fine_mode": fine_mode,
        "model_load_s": round(load_s, 3),
        "segments": [],
    }
    audio_parts = []
    crossfade_samples = int(cosyvoice.sample_rate * crossfade_ms / 1000)

    for idx, segment in enumerate(script["segments"], 1):
        clean_text = segment["text"]
        fine_text = segment.get("fine_text") or clean_text
        use_text = clean_text
        if fine_mode == "instruct" and fine_text:
            use_text = fine_text
        elif fine_mode == "none":
            use_text = clean_text
        instruct_text = "You are a helpful assistant. " + segment["instruct"] + "<|endofprompt|>"

        log(f"[{idx:02d}] {segment['emotion']} {segment['intensity']:.2f}: {clean_text}")
        start = time.time()
        chunks = []
        mode_used = "instruct2"
        try:
            if fine_mode == "cross" and fine_text and fine_text != clean_text:
                mode_used = "cross_lingual"
                cross_text = "You are a helpful assistant.<|endofprompt|>" + fine_text
                gen = cosyvoice.inference_cross_lingual(cross_text, str(ref_audio), stream=False)
            else:
                gen = cosyvoice.inference_instruct2(use_text, instruct_text, str(ref_audio), stream=False)
            for output in gen:
                chunks.append(output["tts_speech"])
        except Exception as exc:
            if use_text != clean_text:
                log(f"  fine_text 失败，回退 clean text: {exc}")
                chunks = []
                mode_used = "instruct2_clean_fallback"
                for output in cosyvoice.inference_instruct2(clean_text, instruct_text, str(ref_audio), stream=False):
                    chunks.append(output["tts_speech"])
            else:
                raise

        audio = torch.concat(chunks, dim=1)
        audio = peak_normalize(audio)
        synth_s = time.time() - start
        audio_s = audio.shape[1] / cosyvoice.sample_rate
        rtf = synth_s / audio_s if audio_s > 0 else 0.0
        seg_file = out_dir / f"segment_{idx:02d}_{segment['emotion']}.wav"
        torchaudio.save(str(seg_file), audio.cpu(), cosyvoice.sample_rate)

        append_with_crossfade(audio_parts, audio, crossfade_samples if segment["pause_after_ms"] == 0 else 0)
        pause_samples = int(cosyvoice.sample_rate * segment["pause_after_ms"] / 1000)
        if pause_samples:
            audio_parts.append(torch.zeros((1, pause_samples), dtype=audio.dtype))

        log(f"  -> {audio_s:.2f}s audio, synth {synth_s:.2f}s, RTF={rtf:.2f}, mode={mode_used}")
        manifest["segments"].append(
            {
                **segment,
                "mode": mode_used,
                "file": seg_file.name,
                "audio_s": round(audio_s, 3),
                "synth_s": round(synth_s, 3),
                "rtf": round(rtf, 3),
            }
        )

    combined = torch.cat(audio_parts, dim=1) if audio_parts else torch.zeros((1, 1))
    combined = peak_normalize(combined)
    combined_file = out_dir / "combined.wav"
    torchaudio.save(str(combined_file), combined.cpu(), cosyvoice.sample_rate)
    manifest["combined_file"] = combined_file.name
    manifest["combined_audio_s"] = round(combined.shape[1] / cosyvoice.sample_rate, 3)
    manifest["total_segment_synth_s"] = round(sum(s["synth_s"] for s in manifest["segments"]), 3)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", help="Input text. Defaults to the complex multi-emotion sample.")
    parser.add_argument("--input-file", help="Read input text from a file.")
    parser.add_argument("--no-llm", action="store_true", help="Skip DeepSeek and use the built-in sample JSON script.")
    parser.add_argument("--script-only", action="store_true", help="Only generate and save the JSON script.")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--fine-mode", choices=["none", "instruct", "cross"], default="instruct")
    parser.add_argument("--crossfade-ms", type=int, default=35)
    parser.add_argument("--model-dir", type=Path, default=default_model_dir())
    parser.add_argument("--ref-audio", type=Path, default=ROOT / "reference_audio" / "xiaoxuan_neutral.wav")
    parser.add_argument("--out-root", type=Path, default=ROOT / "runs")
    args = parser.parse_args()

    load_env()
    text = read_input(args)
    run_dir = args.out_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.txt").write_text(text, encoding="utf-8")

    log(f"Run dir: {run_dir}")
    log(f"Input: {text}")

    raw_llm = ""
    if args.no_llm:
        script = FALLBACK_SCRIPT
        llm_s = 0.0
        log("DeepSeek: skipped, using built-in fallback script")
    else:
        log("DeepSeek: generating JSON script...")
        try:
            llm_s, raw_llm = call_deepseek(text, args.temperature, json_mode=True)
            (run_dir / "deepseek_raw.txt").write_text(raw_llm, encoding="utf-8")
            script = parse_json_script(raw_llm)
            log(f"DeepSeek JSON done: {llm_s:.2f}s")
        except Exception as exc:
            log(f"DeepSeek JSON mode failed: {exc}")
            try:
                llm_s, raw_llm = call_deepseek(text, args.temperature, json_mode=False)
                (run_dir / "deepseek_raw.txt").write_text(raw_llm, encoding="utf-8")
                script = parse_json_script(raw_llm)
                log(f"DeepSeek plain mode done: {llm_s:.2f}s")
            except Exception as exc2:
                if raw_llm:
                    (run_dir / "deepseek_raw.txt").write_text(raw_llm, encoding="utf-8")
                log(f"DeepSeek failed, using fallback script: {exc2}")
                script = FALLBACK_SCRIPT
                llm_s = 0.0

    script = validate_script(script, text)
    (run_dir / "script.json").write_text(json.dumps(script, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Segments: {len(script['segments'])}")
    for idx, seg in enumerate(script["segments"], 1):
        log(f"  {idx:02d}. [{seg['emotion']}:{seg['intensity']:.2f}] {seg['text']} pause={seg['pause_after_ms']}ms")

    if args.script_only:
        log("script-only mode; skip TTS")
        (run_dir / "manifest.json").write_text(
            json.dumps({"llm_s": round(llm_s, 3), "script_only": True}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return

    manifest = synthesize(
        script=script,
        out_dir=run_dir,
        model_dir=args.model_dir,
        ref_audio=args.ref_audio,
        fine_mode=args.fine_mode,
        crossfade_ms=args.crossfade_ms,
    )
    manifest["llm_s"] = round(llm_s, 3)
    manifest["input"] = text
    (run_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    log(f"Combined: {run_dir / manifest['combined_file']}")
    log(
        "Timing: "
        f"llm={manifest['llm_s']:.2f}s, "
        f"model_load={manifest['model_load_s']:.2f}s, "
        f"tts_sum={manifest['total_segment_synth_s']:.2f}s, "
        f"audio={manifest['combined_audio_s']:.2f}s"
    )


if __name__ == "__main__":
    main()
