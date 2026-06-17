#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
首字延迟测试 — 文本输入 → DeepSeek SSE → CosyVoice3 → 播放

测量全链路每个环节的延迟。
用法:
  export DEEPSEEK_API_KEY="sk-..."
  python latency_test.py
"""
import sys, os, time, json, re
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "CosyVoice"))
sys.path.insert(0, str(ROOT / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import torchaudio

# ============================================================
# 配置
# ============================================================
# 加载 .env
from pathlib import Path
def _load_env():
    env_file = Path(__file__).parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
_load_env()

DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL = os.environ.get("DEEPSEEK_API_URL", "https://cpa.witcheng.de/v1/chat/completions")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash")
MODEL_DIR = ROOT / "CosyVoice" / "pretrained_models" / "Fun-CosyVoice3-0.5B"
REF_AUDIO = str(ROOT / "reference_audio" / "xiaoxuan_neutral.wav")
REF_TEXT = "You are a helpful assistant.<|endofprompt|>" + \
    (ROOT / "reference_audio" / "ref_text.txt").read_text(encoding="utf-8").strip()

EMOTION_TO_INSTRUCT = {
    "happy":    "请非常开心地说一句话。",
    "excited":  "请非常开心地说一句话。",
    "sad":      "请非常伤心地说一句话。",
    "calm":     "请用平静从容的语气说这句话。",
    "angry":    "请非常生气地说一句话。",
    "shy":      "请用害羞的语气说这句话。",
    "love":     "请用温柔深情的语气说这句话。",
    "playful":  "请用调皮可爱的语气说这句话。",
    "surprise": "请用惊讶的语气说这句话。",
}
INSTRUCT_PREFIX = "You are a helpful assistant. "
ENDOFPROMPT = "<|endofprompt|>"

SYSTEM_PROMPT = """你是小萱（mei），活泼可爱的年轻女生，ENFP 性格。用户是你的哥哥。
每句话前用方括号标注情感：[emotion:intensity] 内容
可用情感：happy, excited, sad, calm, angry, shy, love, playful, surprise
回复控制在 1-3 句话。"""

# ============================================================
# DeepSeek SSE 流式
# ============================================================
def deepseek_stream(messages):
    """SSE 流式调用 DeepSeek，逐 token yield"""
    import urllib.request

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    data = json.dumps({
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": 0.9,
        "max_tokens": 200,
        "stream": True,
    }).encode("utf-8")

    req = urllib.request.Request(DEEPSEEK_API_URL, data=data, headers=headers, method="POST")
    resp = urllib.request.urlopen(req, timeout=30)

    for line in resp:
        line = line.decode("utf-8").strip()
        if not line or not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
            delta = chunk["choices"][0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
        except (json.JSONDecodeError, KeyError, IndexError):
            continue

# ============================================================
# 流式情感 chunk 提取器
# ============================================================
def extract_chunks(token_stream):
    """
    从 SSE token 流中增量提取 (emotion, text) chunks。
    遇到完整的情感段就 yield，不等全部完成。
    """
    buffer = ""
    current_emotion = "calm"
    # 匹配 [emotion:intensity] 或 [emotion]
    tag_pattern = re.compile(r'\[(\w+)(?::\d+\.?\d*)?\]')

    for token in token_stream:
        buffer += token

        # 尝试提取情感标签
        while True:
            m = tag_pattern.search(buffer)
            if not m:
                break

            # 标签前的文本 → 属于上一个情感段
            prefix = buffer[:m.start()].strip()
            if prefix:
                yield (current_emotion, prefix)

            # 更新当前情感
            current_emotion = m.group(1).lower()
            if current_emotion not in EMOTION_TO_INSTRUCT:
                current_emotion = "calm"
            buffer = buffer[m.end():]

    # 剩余文本
    remaining = buffer.strip()
    if remaining:
        yield (current_emotion, remaining)


# ============================================================
# 延迟测试
# ============================================================
def run_latency_test(cosyvoice, user_text, round_num):
    """单轮延迟测试"""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_text},
    ]

    t_start = time.time()
    print(f"\n{'='*60}")
    print(f"📝 输入: {user_text}")
    print(f"{'='*60}")

    # Phase 1: DeepSeek SSE
    t_llm_start = time.time()
    first_token_time = None
    full_text = ""

    def timed_token_stream():
        nonlocal first_token_time, full_text
        for token in deepseek_stream(messages):
            if first_token_time is None:
                first_token_time = time.time() - t_llm_start
                print(f"  🧠 LLM 首 token: {first_token_time*1000:.0f}ms — \"{token}\"")
            full_text += token
            yield token

    # Phase 2: 流式情感 chunk 提取 + TTS
    t_tts_first = None
    all_audio = []
    chunk_idx = 0

    for emotion, text in extract_chunks(timed_token_stream()):
        chunk_idx += 1
        t_chunk = time.time()

        instruct = EMOTION_TO_INSTRUCT.get(emotion, "请用平静从容的语气说这句话。")
        instruct_text = INSTRUCT_PREFIX + instruct + ENDOFPROMPT

        print(f"  🎭 Chunk {chunk_idx}: [{emotion}] \"{text}\"")

        # CosyVoice3 TTS
        t_synth = time.time()
        audio_parts = []
        for output in cosyvoice.inference_instruct2(text, instruct_text, REF_AUDIO, stream=False):
            audio_parts.append(output["tts_speech"])

        if audio_parts:
            audio = torch.concat(audio_parts, dim=1)
            all_audio.append(audio)
            synth_time = time.time() - t_synth

            if t_tts_first is None:
                t_tts_first = time.time() - t_start
                print(f"  🔊 首段音频: 合成 {synth_time:.1f}s | 距开始 {t_tts_first:.1f}s ← 首字延迟")
            else:
                dur = audio.shape[1] / cosyvoice.sample_rate
                print(f"  🔊 Chunk {chunk_idx}: {dur:.1f}s 音频, 合成 {synth_time:.1f}s")

    t_total = time.time() - t_start

    # 汇总
    total_audio_dur = sum(a.shape[1] for a in all_audio) / cosyvoice.sample_rate if all_audio else 0
    print(f"\n  ─── 延迟汇总 (第 {round_num} 轮) ───")
    print(f"  LLM 首 token:       {first_token_time*1000:.0f}ms" if first_token_time else "  LLM 首 token:       N/A")
    print(f"  LLM 完整响应:       {(time.time()-t_llm_start)*1000:.0f}ms (估算)")
    print(f"  首段音频延迟:       {t_tts_first:.1f}s" if t_tts_first else "  首段音频延迟:       N/A")
    print(f"  总音频时长:         {total_audio_dur:.1f}s")
    print(f"  总耗时:             {t_total:.1f}s")
    print(f"  LLM 完整输出:       {full_text}")

    return {
        "round": round_num,
        "input": user_text,
        "llm_first_token_ms": round(first_token_time * 1000) if first_token_time else None,
        "first_audio_s": round(t_tts_first, 2) if t_tts_first else None,
        "total_audio_s": round(total_audio_dur, 2),
        "total_s": round(t_total, 2),
        "llm_output": full_text,
    }


# ============================================================
# 主流程
# ============================================================
def main():
    if not DEEPSEEK_API_KEY:
        print("❌ 请设置: export DEEPSEEK_API_KEY='sk-...'")
        sys.exit(1)

    print("=" * 60)
    print("⏱️  首字延迟测试 — DeepSeek SSE + CosyVoice3 (晓萱)")
    print("=" * 60)

    # 加载 CosyVoice3
    print("🔧 加载 CosyVoice3...")
    from cosyvoice.cli.cosyvoice import AutoModel
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR))
    print(f"✅ CosyVoice3 加载完成\n")

    # 预热测试
    print("🔧 预热 CosyVoice3 (首次推理较慢)...")
    t = time.time()
    for _ in cosyvoice.inference_instruct2(
        "你好", INSTRUCT_PREFIX + "请非常开心地说一句话。" + ENDOFPROMPT,
        REF_AUDIO, stream=False
    ):
        pass
    print(f"✅ 预热完成 ({time.time()-t:.1f}s)\n")

    # 测试用例
    test_cases = [
        "你今天过得怎么样？",
        "我好无聊啊，陪我聊聊天吧",
        "你最喜欢吃什么？",
    ]

    results = []
    for i, text in enumerate(test_cases, 1):
        r = run_latency_test(cosyvoice, text, i)
        results.append(r)

    # 也可交互输入
    print(f"\n{'='*60}")
    print("进入交互模式 (输入 q 退出)")
    print(f"{'='*60}")
    while True:
        try:
            text = input("\n💬 输入: ").strip()
            if text.lower() == 'q':
                break
            if not text:
                continue
            r = run_latency_test(cosyvoice, text, len(results) + 1)
            results.append(r)
        except KeyboardInterrupt:
            break

    # 汇总所有轮次
    print(f"\n{'='*60}")
    print("📊 全链路延迟汇总")
    print(f"{'='*60}")
    print(f"{'轮次':>4} | {'LLM首token':>10} | {'首音频延迟':>10} | {'总音频':>6} | {'总耗时':>6}")
    print(f"{'─'*4} | {'─'*10} | {'─'*10} | {'─'*6} | {'─'*6}")
    for r in results:
        llm_ft = f"{r['llm_first_token_ms']}ms" if r['llm_first_token_ms'] else "N/A"
        print(f"{r['round']:>4} | {llm_ft:>10} | {r['first_audio_s']:>9.1f}s | {r['total_audio_s']:>5.1f}s | {r['total_s']:>5.1f}s")

    if results:
        avg_llm = [r['llm_first_token_ms'] for r in results if r['llm_first_token_ms']]
        avg_audio = [r['first_audio_s'] for r in results if r['first_audio_s']]
        if avg_llm:
            print(f"\n  平均 LLM 首 token:  {sum(avg_llm)/len(avg_llm):.0f}ms")
        if avg_audio:
            print(f"  平均首音频延迟:   {sum(avg_audio)/len(avg_audio):.1f}s")

    # 保存结果
    out = ROOT / "latency_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存: {out}")


if __name__ == "__main__":
    main()
