# -*- coding: utf-8 -*-
"""
生成多音色参考音频（edge-tts）— macOS 版

与 make_refs_win.py 逻辑一致，区别：
  - 参考音频存在 reference_audio/（与 Windows 共享目录）
  - 额外生成用于 instruct2 的情感参考音频
"""
import asyncio
from pathlib import Path
import edge_tts
import librosa
import soundfile as sf

ROOT = Path(__file__).parent
OUT = ROOT / "reference_audio"
OUT.mkdir(exist_ok=True)

# 统一参考文本：音素覆盖广、语气平稳（最适合做音色提取）
REF_TEXT = "你好呀，我是你的小助手。今天天气真不错，要不要一起出去走走？无论开心还是难过，我都会一直陪在你身边。"

# edge-tts 可用女声
VOICES = {
    "xiaoyi":   "zh-CN-XiaoyiNeural",    # 当前 Airilife 默认
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",  # 经典女声
    "xiaoxuan": "zh-CN-XiaoxuanNeural",  # 温柔女声
}


async def gen(name, voice):
    mp3 = OUT / f"{name}_neutral.mp3"
    c = edge_tts.Communicate(REF_TEXT, voice=voice, rate="+0%", pitch="+0Hz")
    await c.save(str(mp3))
    # 转 24kHz mono PCM16
    y, _ = librosa.load(str(mp3), sr=24000, mono=True)
    wav = OUT / f"{name}_neutral.wav"
    sf.write(str(wav), y, 24000, subtype="PCM_16")
    print(f"  ✅ {name:10s} ({voice}) -> {wav.name}  {len(y)/24000:.2f}s @24kHz")


async def main():
    print(f"参考文本: {REF_TEXT}\n")
    for name, voice in VOICES.items():
        await gen(name, voice)
    # 保存 prompt 文本供 run_test.py 复用
    (OUT / "ref_text.txt").write_text(REF_TEXT, encoding="utf-8")
    print(f"\nprompt 文本已存到 {OUT / 'ref_text.txt'}")


if __name__ == "__main__":
    asyncio.run(main())
