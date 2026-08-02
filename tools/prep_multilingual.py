#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""多语言数据集准备：从 firefly 中文台词批量翻译成英文/日文（SJTU 模型 API）。

用法：
  python tools/prep_multilingual.py --lines 200 --out output/multilingual/translations.json
  python tools/prep_multilingual.py --resume output/multilingual/translations.json   # 断点续传

- 模型：https://models.sjtu.edu.cn/api/v1 的 qwen3.6-27b（多语言翻译最强）
- 一次调用同时返回 en+ja；并发线程池（默认 20，API 支持 ~100）；429/网络错误自动退避重试
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import re
import time
from pathlib import Path

import httpx

API = "https://models.sjtu.edu.cn/api/v1/chat/completions"
MODEL = "qwen3.6-27b"


def _load_key() -> str:
    key = os.environ.get("SJTU_API_KEY", "")
    if not key:
        env = Path(__file__).resolve().parent.parent / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("SJTU_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("请设置环境变量 SJTU_API_KEY 或在 .env 写 SJTU_API_KEY=...")
    return key


KEY = _load_key()
CONCURRENCY = 5
# API 实测限流：每 key 每分钟 10 次请求（并发声明 100 指连接数）。用全局令牌桶压到 ~9 次/分。
RATE_INTERVAL = 6.5

ROOT = Path(__file__).resolve().parent.parent
LIST = ROOT / "data" / "firefly_678_orig" / "train.list"

SYS_BOTH = (
    "你是资深游戏台词翻译。把中文游戏角色台词翻译成自然的英文和日文，"
    "保持口语化、符合角色语气、简短、不要逐字直译。"
    "只输出 JSON：{\"en\":\"英文译文\",\"ja\":\"日文译文\"}，不要任何其他文字。"
)

import threading
_rate_lock = threading.Lock()
_next_slot = 0.0


def acquire_slot():
    """全局令牌桶：所有 worker 共享 ~9 次/分的速率。"""
    global _next_slot
    with _rate_lock:
        now = time.time()
        if _next_slot < now:
            _next_slot = now
        wait = _next_slot - now
        _next_slot += RATE_INTERVAL
    if wait > 0:
        time.sleep(wait)


def translate_both(text: str) -> tuple[str, str]:
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYS_BOTH}, {"role": "user", "content": text}],
        "temperature": 0.3,
        "max_tokens": 500,
    }
    last_err = None
    for attempt in range(12):
        acquire_slot()
        try:
            r = httpx.post(API, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=120)
            if r.status_code == 429:
                ra = r.headers.get("retry-after", "60")
                last_err = f"429(等{ra}s)"
                time.sleep(min(int(ra) + 2, 120))  # 等配额重置
                continue
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"].strip()
            if out.startswith("<think>"):
                out = out.split("</think>", 1)[-1].strip()
            m = re.search(r"\{.*\}", out, re.S)
            if not m:
                last_err = f"no-json: {out[:80]}"
                time.sleep(2)
                continue
            d = json.loads(m.group(0))
            return d["en"].strip(), d["ja"].strip()
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = type(e).__name__
            time.sleep(2 + attempt)
    raise RuntimeError(f"翻译失败({last_err}): {text!r}")


def load_lines(n: int) -> list[tuple[str, str]]:
    all_lines = []
    for line in LIST.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) < 4 or "NICKNAME" in line:
            continue
        all_lines.append((Path(parts[0]).stem, parts[3]))
    step = len(all_lines) / n
    return [all_lines[int(i * step)] for i in range(n)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lines", type=int, default=200)
    ap.add_argument("--out", default=str(ROOT / "output" / "multilingual" / "translations.json"))
    ap.add_argument("--resume", action="store_true", help="断点续传")
    ap.add_argument("--workers", type=int, default=CONCURRENCY)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.resume and out_path.exists():
        items = json.loads(out_path.read_text(encoding="utf-8"))
        print(f"续传：已有 {len(items)} 条")
    else:
        items = [{"id": aid, "zh": zh, "en": None, "ja": None} for aid, zh in load_lines(args.lines)]
        print(f"抽取 {len(items)} 条中文台词")

    pending = [i for i, it in enumerate(items) if not (it["en"] and it["ja"])]
    print(f"待翻译: {len(pending)} 条，并发 {args.workers}")

    def work(i: int):
        en, ja = translate_both(items[i]["zh"])
        return i, en, ja

    done = 0
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, i) for i in pending]
        for fut in cf.as_completed(futs):
            i, en, ja = fut.result()
            items[i]["en"], items[i]["ja"] = en, ja
            done += 1
            if done % 10 == 0 or done == len(pending):
                out_path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
                rate = done / max(time.time() - t0, 1e-6)
                print(f"进度 {done}/{len(pending)} ({rate:.1f} 条/s)", flush=True)

    out_path.write_text(json.dumps(items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：{len(items)} 条 → {out_path}")


if __name__ == "__main__":
    main()
