# -*- coding: utf-8 -*-
"""
CosyVoice3 多音色情感测试

为每个音色生成：克隆(中性) + 3 种预训练情感(开心/伤心/生气)，
用于 A/B 对比哪个 edge-tts 女声克隆得最像、情感最自然。

文件命名：test_outputs/<音色中文名>/<音色中文名>_<情感>.wav
"""
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
# 官方要求的两条 import 路径
sys.path.insert(0, str(ROOT / "CosyVoice"))
sys.path.insert(0, str(ROOT / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import torchaudio

MODEL_DIR = ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"
REF_DIR = ROOT / "reference_audio"
OUT_DIR = ROOT / "test_outputs"
OUT_DIR.mkdir(exist_ok=True)

# 参考音频对应的 prompt 文本（3 个音色共用同一段，由 make_refs.py 生成）
REF_TEXT_RAW = (REF_DIR / "ref_text.txt").read_text(encoding="utf-8").strip()
REF_TEXT = "You are a helpful assistant.<|endofprompt|>" + REF_TEXT_RAW

# 待测音色：中文名 -> 参考音频文件名前缀
VOICES = {
    "晓伊": "xiaoyi",
    "晓晓": "xiaoxiao",
    "晓萱": "xiaoxuan",
}

# 每个音色测：克隆 + 3 种预训练情感
# (情感中文名, instruct 指令, 合成文本)；instruct=None 表示纯克隆
TASKS = [
    ("克隆", None, "大家好，这是用零样本克隆生成的声音，听听看像不像？"),
    ("开心", "请非常开心地说一句话。", "太好了！今天真是开心的一天，我们一起去玩吧！"),
    ("伤心", "请非常伤心地说一句话。", "唉……我今天心情有点低落，感觉有些难过。"),
    ("生气", "请非常生气地说一句话。", "你怎么又迟到了！这已经是这周第三次了！"),
]

INSTRUCT_PREFIX = "You are a helpful assistant. "
ENDOFPROMPT = "<|endofprompt|>"


def log(msg):
    print(msg, flush=True)


def main():
    log("=" * 60)
    log("CosyVoice3 多音色情感测试")
    log("=" * 60)
    log(f"音色: {', '.join(VOICES)}")
    log(f"CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    from cosyvoice.cli.cosyvoice import AutoModel
    log("\n加载模型中（首次较慢）...")
    t0 = time.time()
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR), fp16=torch.cuda.is_available())
    log(f"✅ 模型加载完成 {time.time() - t0:.1f}s，类型={type(cosyvoice).__name__}")

    results = []

    def synth(voice_cn, prompt_wav, emo_cn, tts_text, instruct):
        sub = OUT_DIR / voice_cn
        sub.mkdir(exist_ok=True)
        out_path = sub / f"{voice_cn}_{emo_cn}.wav"
        tag = f"{voice_cn}/{emo_cn}"
        t = time.time()
        try:
            if instruct is None:  # 纯克隆
                gen = cosyvoice.inference_zero_shot(
                    tts_text, REF_TEXT, prompt_wav, stream=False)
            else:                 # 克隆 + 情感
                instruct_text = INSTRUCT_PREFIX + instruct + ENDOFPROMPT
                gen = cosyvoice.inference_instruct2(
                    tts_text, instruct_text, prompt_wav, stream=False)
            audio = torch.concat([o["tts_speech"] for o in gen], dim=1)
            torchaudio.save(str(out_path), audio, cosyvoice.sample_rate)
            dur = audio.shape[1] / cosyvoice.sample_rate
            elapsed = time.time() - t
            rtf = elapsed / dur if dur > 0 else 0
            log(f"  ✅ {tag}: {dur:.2f}s, RTF={rtf:.2f}")
            results.append({"voice": voice_cn, "emotion": emo_cn, "ok": True,
                            "audio_sec": round(dur, 2), "rtf": round(rtf, 3),
                            "file": str(out_path.relative_to(OUT_DIR))})
        except Exception as e:
            log(f"  ❌ {tag}: {e}")
            results.append({"voice": voice_cn, "emotion": emo_cn, "ok": False, "error": str(e)})

    for voice_cn, prefix in VOICES.items():
        ref = REF_DIR / f"{prefix}_neutral.wav"
        log(f"\n=== 音色: {voice_cn} ({ref.name}) ===")
        if not ref.exists():
            log(f"  ⚠️ 参考音频缺失，跳过: {ref}")
            continue
        for emo_cn, instruct, text in TASKS:
            synth(voice_cn, str(ref), emo_cn, text, instruct)

    ok = [r for r in results if r.get("ok")]
    log("\n" + "=" * 60)
    log(f"完成: {len(ok)}/{len(results)} 成功")
    if ok:
        log(f"平均 RTF: {sum(r['rtf'] for r in ok) / len(ok):.3f}")
    log(f"音频目录: {OUT_DIR}（按音色分子文件夹）")
    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log("=" * 60)


if __name__ == "__main__":
    main()
