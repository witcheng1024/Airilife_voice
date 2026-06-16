# -*- coding: utf-8 -*-
"""
CosyVoice3 正式测试脚本（重写版）

修复上一轮的所有问题：
  1. 用官方 AutoModel 入口（自动识别 CosyVoice3），不再用基类 CosyVoice
  2. 正确加入 third_party/Matcha-TTS 到 sys.path
  3. 用真实 edge-tts 晓伊参考音频（reference_audio/xiaoyi_neutral.wav）
  4. 不碰 pynini/WeTextProcessing，用 wetext 做正则化
  5. 情感分两类：预训练(可靠) vs 自由指令(实验)，如实标注

输出：test_outputs/ 下的真实 wav + 一份 result.json 性能/成功率报告
"""
import os
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
# 关键：官方要求的两条路径
sys.path.insert(0, str(ROOT / "CosyVoice"))
sys.path.insert(0, str(ROOT / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import torchaudio

MODEL_DIR = ROOT / "pretrained_models" / "Fun-CosyVoice3-0.5B"
REF_WAV = ROOT / "reference_audio" / "xiaoyi_neutral.wav"
OUT_DIR = ROOT / "test_outputs"
OUT_DIR.mkdir(exist_ok=True)

# zero-shot 时给参考音频配的 "prompt 文本"（必须与参考音频内容一致）
# 注意：CosyVoice3 要求 prompt_text 必须带 <|endofprompt|> 标记
REF_TEXT = ("You are a helpful assistant.<|endofprompt|>"
            "你好呀，我是你的小助手。今天天气真不错，要不要一起出去走走？"
            "无论开心还是难过，我都会一直陪在你身边。")

# A 类：CosyVoice3 预训练支持的情感（可靠）
PRETRAINED_EMOTIONS = [
    ("happy", "请非常开心地说一句话。", "太好了！今天真是开心的一天，我们一起去玩吧！"),
    ("sad",   "请非常伤心地说一句话。", "唉……我今天心情有点低落，感觉有些难过。"),
    ("angry", "请非常生气地说一句话。", "你怎么又迟到了！这已经是这周第三次了！"),
]

# B 类：自由文本指令（实验性，非预训练，效果不保证）
EXPERIMENTAL_EMOTIONS = [
    ("shy",      "请用害羞撒娇的语气说一句话。", "哎呀……人家才不是那个意思啦，你别这样看着我嘛。"),
    ("gentle",   "请用温柔体贴的语气说一句话。", "辛苦啦，先休息一下吧，我给你倒杯热水。"),
    ("playful",  "请用俏皮可爱的语气说一句话。", "嘿嘿，被我抓到啦，你是不是又偷吃零食了？"),
    ("surprise", "请用非常惊讶的语气说一句话。", "什么？！这居然是真的，我简直不敢相信！"),
]

INSTRUCT_PREFIX = "You are a helpful assistant. "
ENDOFPROMPT = "<|endofprompt|>"


def log(msg):
    print(msg, flush=True)


def main():
    log("=" * 60)
    log("CosyVoice3 正式测试")
    log("=" * 60)
    log(f"模型: {MODEL_DIR}")
    log(f"参考音频: {REF_WAV} (存在={REF_WAV.exists()})")
    log(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---- 加载模型 ----
    from cosyvoice.cli.cosyvoice import AutoModel
    log("\n[1/4] 加载模型中（首次较慢）...")
    t0 = time.time()
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR), fp16=torch.cuda.is_available())
    log(f"✅ 模型加载完成，耗时 {time.time() - t0:.1f}s，类型={type(cosyvoice).__name__}，采样率={cosyvoice.sample_rate}")

    # 参考音频：接口需要的是「文件路径字符串」，内部会自己 load_wav
    prompt_wav = str(REF_WAV)

    results = []

    def synth(tag, tts_text, mode, instruct=None):
        """生成一条并保存，记录耗时/RTF"""
        out_path = OUT_DIR / f"{tag}.wav"
        t = time.time()
        try:
            if mode == "zero_shot":
                gen = cosyvoice.inference_zero_shot(
                    tts_text, REF_TEXT, prompt_wav, stream=False)
            else:  # instruct2
                instruct_text = INSTRUCT_PREFIX + instruct + ENDOFPROMPT
                gen = cosyvoice.inference_instruct2(
                    tts_text, instruct_text, prompt_wav, stream=False)
            speeches = [o["tts_speech"] for o in gen]
            audio = torch.concat(speeches, dim=1)
            torchaudio.save(str(out_path), audio, cosyvoice.sample_rate)
            dur = audio.shape[1] / cosyvoice.sample_rate
            elapsed = time.time() - t
            rtf = elapsed / dur if dur > 0 else 0
            log(f"  ✅ {tag}: {dur:.2f}s 音频, 耗时 {elapsed:.2f}s, RTF={rtf:.3f}")
            results.append({"tag": tag, "mode": mode, "ok": True,
                            "audio_sec": round(dur, 2), "elapsed_sec": round(elapsed, 2),
                            "rtf": round(rtf, 3), "file": out_path.name})
        except Exception as e:
            log(f"  ❌ {tag}: {e}")
            results.append({"tag": tag, "mode": mode, "ok": False, "error": str(e)})

    # ---- [2/4] zero-shot 克隆（晓伊音色，中性）----
    log("\n[2/4] zero-shot 晓伊音色克隆...")
    synth("clone_neutral", "大家好，我是用零样本克隆生成的声音，听听看像不像晓伊？",
          "zero_shot")

    # ---- [3/4] 预训练情感（可靠）----
    log("\n[3/4] 预训练情感 (instruct2, 可靠)...")
    for name, instruct, text in PRETRAINED_EMOTIONS:
        synth(f"emotion_{name}", text, "instruct2", instruct)

    # ---- [4/4] 实验性自由指令情感 ----
    log("\n[4/4] 实验性自由指令情感 (非预训练, 效果不保证)...")
    for name, instruct, text in EXPERIMENTAL_EMOTIONS:
        synth(f"exp_{name}", text, "instruct2", instruct)

    # ---- 报告 ----
    ok = [r for r in results if r.get("ok")]
    log("\n" + "=" * 60)
    log(f"完成: {len(ok)}/{len(results)} 成功")
    if ok:
        avg_rtf = sum(r["rtf"] for r in ok) / len(ok)
        log(f"平均 RTF: {avg_rtf:.3f} (越小越快, <1 即快于实时)")
    log(f"音频输出目录: {OUT_DIR}")
    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log("=" * 60)


if __name__ == "__main__":
    main()
