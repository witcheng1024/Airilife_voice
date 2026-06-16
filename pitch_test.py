# -*- coding: utf-8 -*-
"""
晓萱音调实验：验证"晓萱音色 + 改音调"能否逼近目标声线

对晓萱的克隆音做不同半音(semitone)的 pitch shift，
保持共振峰不破坏的前提下改变音高，听哪个最接近想要的效果。
"""
from pathlib import Path
import librosa
import soundfile as sf

ROOT = Path(__file__).parent
SRC = ROOT / "test_outputs" / "晓萱" / "晓萱_克隆.wav"
OUT = ROOT / "test_outputs" / "晓萱_音调实验"
OUT.mkdir(parents=True, exist_ok=True)

# 半音偏移：负=更低沉成熟，正=更高更年轻/萝莉
# Neuro 那类音色通常在原声基础上略微上调
SHIFTS = [-3, -2, -1, +1, +2, +3, +4]

def main():
    y, sr = librosa.load(str(SRC), sr=None, mono=True)
    print(f"源: {SRC.name} ({len(y)/sr:.2f}s @ {sr}Hz)\n")
    # 原声也复制一份做基准
    sf.write(str(OUT / "晓萱_原声_0.wav"), y, sr, subtype="PCM_16")
    print("  ✅ 晓萱_原声_0.wav (基准)")
    for st in SHIFTS:
        y2 = librosa.effects.pitch_shift(y, sr=sr, n_steps=st)
        sign = f"+{st}" if st > 0 else f"{st}"
        name = f"晓萱_音调{sign}.wav"
        sf.write(str(OUT / name), y2, sr, subtype="PCM_16")
        print(f"  ✅ {name}")
    print(f"\n输出: {OUT}")

if __name__ == "__main__":
    main()
