#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Latency benchmarks for DeepSeek streaming formats and CosyVoice3 TTS."""
import argparse
import json
import os
import re
import sys
import time
import urllib.request
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


EMOTIONS = "happy, excited, sad, calm, angry, shy, love, playful, surprise, neutral"
DEFAULT_TEXT = "我好无聊啊，陪我聊聊天吧"
CLEAN_TEXT = (
    "哥哥……你终于回来了呀，"
    "我还以为、还以为你今天又要丢下我一个人了呢……"
    "不过没关系啦！"
    "只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~"
)
CLEAN_INSTRUCT = "请用先惊喜撒娇、再委屈带哭腔、然后强撑甜笑、最后调皮甜软的语气说这句话。"


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


def model_dir() -> Path:
    for candidate in [
        ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B",
        ROOT / "CosyVoice" / "pretrained_models" / "Fun-CosyVoice3-0.5B",
    ]:
        if (candidate / "cosyvoice3.yaml").exists():
            return candidate
    raise FileNotFoundError("Fun-CosyVoice3-0.5B/cosyvoice3.yaml not found")


def deepseek_stream(system_prompt: str, user_text: str):
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    api_url = os.environ.get("DEEPSEEK_API_URL", "https://api.deepseek.com/v1/chat/completions")
    model = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text},
        ],
        "temperature": 0.4,
        "max_tokens": 260,
        "stream": True,
    }
    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        for line in resp:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            try:
                delta = json.loads(data)["choices"][0].get("delta", {})
            except (json.JSONDecodeError, KeyError, IndexError):
                continue
            content = delta.get("content", "")
            if content:
                yield content


def protocol_prompt(protocol: str) -> str:
    base = (
        "你是小萱（mei），活泼可爱的年轻女生，用户是哥哥。"
        "回复要自然短句，拆成 1-3 个适合 TTS 立即播放的短片段。"
        f"可用情感: {EMOTIONS}。不要输出解释或 Markdown。"
    )
    if protocol == "inline_tags":
        return base + "只输出连续格式：[emotion:intensity]文本[emotion:intensity]文本。不要换行。"
    if protocol == "line_tags":
        return base + "每行一个片段，格式：[emotion:intensity]文本。每个片段后必须换行。"
    if protocol == "jsonl":
        return base + '每行一个 JSON：{"e":"emotion","i":0.0-1.0,"t":"文本","p":0-300}。'
    if protocol == "json_object":
        return (
            base
            + '只输出紧凑 JSON：{"segments":[{"emotion":"...","intensity":0.7,'
            + '"text":"...","pause_after_ms":120}]}。'
        )
    raise ValueError(f"unknown protocol: {protocol}")


TAG_RE = re.compile(r"\[(\w+)(?::([0-9.]+))?\]")
PUNCT_RE = re.compile(r"[。！？!?~\n]")


def first_inline_chunk(text: str) -> dict[str, Any] | None:
    tag = TAG_RE.search(text)
    if not tag:
        return None
    rest = text[tag.end() :]
    punct = PUNCT_RE.search(rest)
    next_tag = TAG_RE.search(rest)
    end_candidates = []
    if punct:
        end_candidates.append(punct.end())
    if next_tag:
        end_candidates.append(next_tag.start())
    if not end_candidates:
        return None
    end = min(end_candidates)
    chunk = rest[:end].strip()
    if not chunk:
        return None
    return {"emotion": tag.group(1), "text": chunk}


def first_line_tag_chunk(text: str) -> dict[str, Any] | None:
    line = text.splitlines()[0].strip()
    tag = TAG_RE.match(line)
    if not tag:
        return None
    rest = line[tag.end() :]
    if "\n" in text:
        chunk = rest.strip()
    else:
        punct = PUNCT_RE.search(rest)
        if not punct:
            return None
        chunk = rest[: punct.end()].strip()
    return {"emotion": tag.group(1), "text": chunk} if chunk else None


def first_jsonl_chunk(text: str) -> dict[str, Any] | None:
    line = text.splitlines()[0] if "\n" in text else text.strip()
    if not line.endswith("}"):
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    chunk = str(data.get("t", "")).strip()
    return {"emotion": data.get("e", "neutral"), "text": chunk} if chunk else None


def first_json_object_chunk(text: str) -> dict[str, Any] | None:
    marker = text.find('"segments"')
    if marker == -1:
        return None
    start = text.find("{", marker)
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    for idx in range(start, len(text)):
        char = text[idx]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    data = json.loads(text[start : idx + 1])
                except json.JSONDecodeError:
                    return None
                chunk = str(data.get("text", "")).strip()
                return {"emotion": data.get("emotion", "neutral"), "text": chunk} if chunk else None
    return None


PARSERS = {
    "inline_tags": first_inline_chunk,
    "line_tags": first_line_tag_chunk,
    "jsonl": first_jsonl_chunk,
    "json_object": first_json_object_chunk,
}


def bench_protocol(protocol: str, user_text: str) -> dict[str, Any]:
    start = time.perf_counter()
    first_token_s = None
    first_chunk_s = None
    first_chunk = None
    full = ""
    for token in deepseek_stream(protocol_prompt(protocol), user_text):
        if first_token_s is None:
            first_token_s = time.perf_counter() - start
        full += token
        if first_chunk_s is None:
            parsed = PARSERS[protocol](full)
            if parsed:
                first_chunk_s = time.perf_counter() - start
                first_chunk = parsed
    total_s = time.perf_counter() - start
    return {
        "protocol": protocol,
        "model": os.environ.get("DEEPSEEK_MODEL", ""),
        "first_token_ms": round(first_token_s * 1000) if first_token_s else None,
        "first_chunk_ms": round(first_chunk_s * 1000) if first_chunk_s else None,
        "total_ms": round(total_s * 1000),
        "chars": len(full),
        "first_chunk": first_chunk,
        "output": full,
    }


def bench_protocols(args: argparse.Namespace) -> None:
    load_env()
    protocols = args.protocols.split(",")
    results = []
    for protocol in protocols:
        for _ in range(args.rounds):
            result = bench_protocol(protocol.strip(), args.text)
            results.append(result)
            print("BENCH_PROTOCOL " + json.dumps(result, ensure_ascii=False), flush=True)
    save_results("deepseek_protocols", results)


def bench_tts(args: argparse.Namespace) -> None:
    import torch
    from cosyvoice.cli.cosyvoice import AutoModel

    load_env()
    ref_audio = str(ROOT / "reference_audio" / "xiaoxuan_neutral.wav")
    instruct = "You are a helpful assistant. " + args.instruct + "<|endofprompt|>"
    load_start = time.perf_counter()
    cosyvoice = AutoModel(model_dir=str(model_dir()), fp16=True)
    load_s = time.perf_counter() - load_start
    start = time.perf_counter()
    first_chunk_s = None
    chunks = []
    for output in cosyvoice.inference_instruct2(args.text, instruct, ref_audio, stream=args.stream):
        now = time.perf_counter()
        if first_chunk_s is None:
            first_chunk_s = now - start
        chunks.append(output["tts_speech"])
    total_s = time.perf_counter() - start
    audio = torch.concat(chunks, dim=1) if chunks else torch.zeros((1, 0))
    audio_s = audio.shape[1] / cosyvoice.sample_rate if audio.shape[1] else 0
    result = {
        "stream": args.stream,
        "load_s": round(load_s, 3),
        "first_chunk_s": round(first_chunk_s, 3) if first_chunk_s else None,
        "total_s": round(total_s, 3),
        "audio_s": round(audio_s, 3),
        "rtf": round(total_s / audio_s, 3) if audio_s else None,
        "chunk_count": len(chunks),
    }
    print("BENCH_TTS " + json.dumps(result, ensure_ascii=False), flush=True)
    save_results("tts", [result])


def save_results(kind: str, results: list[dict[str, Any]]) -> None:
    out_dir = ROOT / "runs" / ("bench_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{kind}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved: {out_dir / f'{kind}.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("deepseek-protocols")
    p.add_argument("--text", default=DEFAULT_TEXT)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--protocols", default="inline_tags,line_tags,jsonl,json_object")
    p.set_defaults(func=bench_protocols)

    p = sub.add_parser("tts")
    p.add_argument("--text", default=CLEAN_TEXT)
    p.add_argument("--instruct", default=CLEAN_INSTRUCT)
    p.add_argument("--stream", action="store_true")
    p.set_defaults(func=bench_tts)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
