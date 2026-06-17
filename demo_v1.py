#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AiriLife V1 Demo — 端到端语音对话
麦克风 → Whisper → DeepSeek API → 情感解析 → CosyVoice3 → 播放

用法:
  conda activate cosyvoice
  cd Airilife_voice
  python demo_v1.py

首次运行前需设置环境变量:
  export DEEPSEEK_API_KEY="sk-..."
"""
import sys
import os
import time
import json
import re
import io
import wave
import struct
import threading
import queue
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "CosyVoice"))
sys.path.insert(0, str(ROOT / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import numpy as np

# ============================================================
# 配置
# ============================================================
# 加载 .env
def _load_env():
    env_file = ROOT / ".env"
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
REF_AUDIO = ROOT / "reference_audio" / "xiaoxuan_neutral.wav"
REF_TEXT_RAW = (ROOT / "reference_audio" / "ref_text.txt").read_text(encoding="utf-8").strip()
REF_TEXT = "You are a helpful assistant.<|endofprompt|>" + REF_TEXT_RAW

# 采样率
ASR_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

# ============================================================
# DeepSeek 人设 Prompt
# ============================================================
SYSTEM_PROMPT = """你是小萱（mei），一个活泼可爱的年轻女生，ENFP 性格。
你和用户是兄妹关系（你是妹妹），你叫用户"哥哥"。
你性格开朗、好奇、有点撒娇、偶尔调皮。

## 输出格式要求
你必须在每句话前用方括号标注情感和强度（0-1），格式：
[emotion:intensity] 说话内容

可用情感标签：
- happy: 开心、温暖
- excited: 兴奋、期待
- sad: 难过、委屈
- calm: 平静、从容
- angry: 不满、吐槽
- shy: 害羞、撒娇
- love: 深情、温情
- playful: 调皮、俏皮
- surprise: 惊讶

可以在一段回复中切换多种情感。回复要简短自然，像真人聊天。
每次回复控制在 1-3 句话。

## 示例
[happy:0.8] 哥哥你回来啦！今天过得怎么样？
[playful:0.7] 嘿嘿，我才没有偷吃你的零食呢~
[sad:0.5] 嗯…有点想你了。[happy:0.6] 不过现在看到你就好啦！"""

# ============================================================
# 情感 → CosyVoice3 instruct 映射
# ============================================================
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

# ============================================================
# 情感解析器
# ============================================================
# 匹配 [emotion:intensity] 文本 或 [emotion] 文本
EMOTION_PATTERN = re.compile(r'\[(\w+)(?::(\d+\.?\d*))?\]\s*([^\[]+)')

def parse_emotion_text(text):
    """解析 DeepSeek 输出，返回 [(emotion, intensity, content), ...]"""
    matches = EMOTION_PATTERN.findall(text.strip())
    if not matches:
        # 没有情感标签，默认 calm
        return [("calm", 0.5, text.strip())]

    segments = []
    for emotion, intensity, content in matches:
        emotion = emotion.lower().strip()
        intensity = float(intensity) if intensity else 0.5
        content = content.strip()
        if emotion not in EMOTION_TO_INSTRUCT:
            emotion = "calm"
        if content:
            segments.append((emotion, intensity, content))

    return segments if segments else [("calm", 0.5, text.strip())]

# ============================================================
# 模块加载
# ============================================================
def load_whisper():
    """加载 Whisper ASR 模型"""
    print("🔧 加载 Whisper ASR...")
    import whisper
    model = whisper.load_model("small", device="cpu")
    print("✅ Whisper 加载完成")
    return model

def load_cosyvoice():
    """加载 CosyVoice3 TTS 模型"""
    print("🔧 加载 CosyVoice3...")
    from cosyvoice.cli.cosyvoice import AutoModel
    model = AutoModel(model_dir=str(MODEL_DIR))
    print(f"✅ CosyVoice3 加载完成 (采样率: {model.sample_rate} Hz)")
    return model

# ============================================================
# ASR: 录音 + 识别
# ============================================================
def record_audio(duration=None, sample_rate=ASR_SAMPLE_RATE):
    """录音（按键控制：按 Enter 停止，或指定 duration 秒）"""
    import sounddevice as sd

    if duration:
        print(f"🎤 录音 {duration} 秒...")
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate,
                       channels=1, dtype='float32')
        sd.wait()
        return audio.flatten()

    print("🎤 开始录音... 按 Enter 停止")
    frames = []
    recording = True

    def callback(indata, frame_count, time_info, status):
        if recording:
            frames.append(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=1,
                            dtype='float32', callback=callback)
    stream.start()
    input()  # 等待 Enter
    recording = False
    stream.stop()
    stream.close()

    audio = np.concatenate(frames).flatten()
    duration = len(audio) / sample_rate
    print(f"🔇 录音结束 ({duration:.1f}s)")
    return audio

def transcribe(whisper_model, audio):
    """Whisper 语音识别"""
    t = time.time()
    result = whisper_model.transcribe(audio, language="zh", fp16=False)
    text = result["text"].strip()
    elapsed = time.time() - t
    print(f"📝 ASR ({elapsed:.1f}s): {text}")
    return text

# ============================================================
# LLM: DeepSeek API
# ============================================================
def call_deepseek(conversation_history):
    """调用 DeepSeek API，返回完整文本"""
    import urllib.request

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
    }
    data = json.dumps({
        "model": "deepseek-chat",
        "messages": conversation_history,
        "temperature": 0.9,
        "max_tokens": 200,
    }).encode("utf-8")

    req = urllib.request.Request(
        DEEPSEEK_API_URL,
        data=data,
        headers=headers,
        method="POST"
    )

    t = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read().decode("utf-8"))

    text = result["choices"][0]["message"]["content"]
    elapsed = time.time() - t
    print(f"🧠 LLM ({elapsed:.1f}s): {text}")
    return text

# ============================================================
# TTS: CosyVoice3 + 情感控制
# ============================================================
def synthesize_with_emotion(cosyvoice_model, segments, ref_audio, ref_text):
    """用 CosyVoice3 instruct2 模式合成带情感的语音"""
    import torchaudio
    all_audio = []

    for emotion, intensity, content in segments:
        instruct = EMOTION_TO_INSTRUCT.get(emotion, "请用平静从容的语气说这句话。")
        instruct_text = INSTRUCT_PREFIX + instruct + ENDOFPROMPT

        t = time.time()
        print(f"  🔊 [{emotion}:{intensity}] {content}")

        audio_chunks = []
        for output in cosyvoice_model.inference_instruct2(
            content, instruct_text, ref_audio, stream=False
        ):
            audio_chunks.append(output["tts_speech"])

        if audio_chunks:
            audio = torch.concat(audio_chunks, dim=1)
            all_audio.append(audio)

        elapsed = time.time() - t
        dur = sum(a.shape[1] for a in audio_chunks) / cosyvoice_model.sample_rate if audio_chunks else 0
        print(f"     ✅ {dur:.1f}s, {elapsed:.1f}s")

    if all_audio:
        return torch.concat(all_audio, dim=1)
    return None

def play_audio(audio_tensor, sample_rate):
    """播放音频"""
    import sounddevice as sd

    audio_np = audio_tensor.squeeze().numpy()
    # 归一化
    audio_np = audio_np / max(abs(audio_np.max()), abs(audio_np.min()), 1e-8)
    sd.play(audio_np, sample_rate)
    sd.wait()

# ============================================================
# 主循环
# ============================================================
def main():
    print("=" * 60)
    print("🌸 AiriLife V1 Demo — 小萱语音对话")
    print("=" * 60)

    if not DEEPSEEK_API_KEY:
        print("❌ 请设置 DEEPSEEK_API_KEY 环境变量")
        print("   export DEEPSEEK_API_KEY='sk-...'")
        sys.exit(1)

    # 检查参考音频
    if not REF_AUDIO.exists():
        print(f"❌ 参考音频不存在: {REF_AUDIO}")
        sys.exit(1)
    print(f"🎵 参考音频: {REF_AUDIO.name} (晓萱)")

    # 加载模型
    whisper_model = load_whisper()
    cosyvoice_model = load_cosyvoice()

    # 对话历史
    conversation = [
        {"role": "system", "content": SYSTEM_PROMPT},
    ]

    print("\n" + "=" * 60)
    print("开始对话！按 Enter 开始录音，再按 Enter 停止。输入 'q' 退出。")
    print("=" * 60)

    while True:
        try:
            cmd = input("\n>>> (Enter=录音, q=退出): ").strip()
            if cmd.lower() == 'q':
                print("👋 再见！")
                break

            # 1. ASR
            audio = record_audio(sample_rate=ASR_SAMPLE_RATE)
            if len(audio) < ASR_SAMPLE_RATE * 0.3:
                print("⚠️ 录音太短，跳过")
                continue

            user_text = transcribe(whisper_model, audio)
            if not user_text:
                print("⚠️ 未识别到语音")
                continue

            # 2. LLM
            conversation.append({"role": "user", "content": user_text})
            llm_response = call_deepseek(conversation)
            conversation.append({"role": "assistant", "content": llm_response})

            # 3. 情感解析
            segments = parse_emotion_text(llm_response)
            print(f"🎭 情感片段: {len(segments)} 段")
            for emo, intensity, text in segments:
                print(f"   [{emo}:{intensity}] {text}")

            # 4. TTS
            print("🔊 合成中...")
            full_audio = synthesize_with_emotion(
                cosyvoice_model, segments,
                str(REF_AUDIO), REF_TEXT
            )

            # 5. 播放
            if full_audio is not None:
                play_audio(full_audio, cosyvoice_model.sample_rate)

        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
