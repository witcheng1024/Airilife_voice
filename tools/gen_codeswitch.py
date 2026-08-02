#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""生成中英/中日"代码切换"（混合语种）台词源，供 CosyVoice3 合成。

用途：多语言模型需要会"一句话里混两种语言"，比如：
  "华哥哥，请查一下wiki看一下'Hi there! How's your day going? Let's go for a walk!'这句话是什么意思吧"
这类句子叫 code-switching。这里用 SJTU API 生成一批自然的混合语种台词源。

用法：
  python tools/gen_codeswitch.py --en 60 --ja 20 --out output/multilingual/mix_sources.json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from pathlib import Path

import httpx

API = "https://models.sjtu.edu.cn/api/v1/chat/completions"
MODEL = "qwen3.6-27b"
RATE_INTERVAL = 6.5  # ~9 次/分（API 实测每 key 10 次/分）

ROOT = Path(__file__).resolve().parent.parent


def _load_key() -> str:
    key = os.environ.get("SJTU_API_KEY", "")
    if not key:
        env = ROOT / ".env"
        if env.exists():
            for line in env.read_text(encoding="utf-8").splitlines():
                if line.startswith("SJTU_API_KEY="):
                    key = line.split("=", 1)[1].strip()
    if not key:
        raise SystemExit("请设置环境变量 SJTU_API_KEY 或在 .env 写 SJTU_API_KEY=...")
    return key


KEY = _load_key()

_rate_lock = threading.Lock()
_next_slot = 0.0


def acquire_slot():
    global _next_slot
    with _rate_lock:
        now = time.time()
        if _next_slot < now:
            _next_slot = now
        wait = _next_slot - now
        _next_slot += RATE_INTERVAL
    if wait > 0:
        time.sleep(wait)


def gen_batch(n: int, lang: str) -> list[str]:
    embed = "英文短句（日常口语，5~12 词）" if lang == "en" else "日语短句（日常口语）"
    sys_p = (
        f"你是《崩坏：星穹铁道》角色流萤（Firefly）的台词编剧。流萤是活泼可爱的年轻女孩。"
        f"请生成 {n} 条自然、口语化的中文台词，每条里嵌入一句{embed}，"
        f"模拟角色说话时自然的中外语混用（如引用英文短语、对外语打招呼、念外语台词）。"
        f"例如：\"华哥哥，请查一下wiki看一下'Hi there! How's your day going? Let's go for a walk!'这句话是什么意思吧\"。"
        f"只输出 JSON 数组：[{{\"zh\":\"中文含外语句子\"}}]，不要任何其他文字。"
    )
    body = {
        "model": MODEL,
        "messages": [{"role": "system", "content": sys_p}, {"role": "user", "content": f"生成 {n} 条，混合{embed}的中文台词。"}],
        "temperature": 0.9,
        "max_tokens": 3000,
    }
    last_err = None
    for attempt in range(12):
        acquire_slot()
        try:
            r = httpx.post(API, headers={"Authorization": f"Bearer {KEY}"}, json=body, timeout=180)
            if r.status_code == 429:
                ra = r.headers.get("retry-after", "60")
                time.sleep(min(int(ra) + 2, 120))
                continue
            r.raise_for_status()
            out = r.json()["choices"][0]["message"]["content"].strip()
            m = re.search(r"\[.*\]", out, re.S)
            if not m:
                last_err = f"no-json: {out[:100]}"
                continue
            arr = json.loads(m.group(0))
            return [x["zh"] for x in arr]
        except (httpx.HTTPStatusError, httpx.TransportError) as e:
            last_err = type(e).__name__
            time.sleep(3)
    raise RuntimeError(f"生成失败({last_err})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--en", type=int, default=60, help="中英混合条数")
    ap.add_argument("--ja", type=int, default=20, help="中日混合条数")
    ap.add_argument("--out", default=str(ROOT / "output" / "multilingual" / "mix_sources.json"))
    args = ap.parse_args()

    all_items = []
    if args.en:
        print(f"生成中英混合 {args.en} 条...", flush=True)
        all_items += [{"zh": s, "type": "mix_en"} for s in gen_batch(args.en, "en")]
    if args.ja:
        print(f"生成中日混合 {args.ja} 条...", flush=True)
        all_items += [{"zh": s, "type": "mix_ja"} for s in gen_batch(args.ja, "ja")]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_items, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"完成：{len(all_items)} 条 → {out}")


if __name__ == "__main__":
    main()
