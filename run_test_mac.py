# -*- coding: utf-8 -*-
"""
CosyVoice3 完整能力测试 — macOS 版

覆盖：
  - 多音色 zero-shot 克隆（晓伊/晓晓/晓萱）
  - 3 种预训练情感（开心/伤心/生气）
  - 6 种自由指令情感（撒娇/温柔/害羞/调皮/惊讶/平静）
  - 3 种 fine-grained 标签（[breath]/<strong>/[laughter]）
  - 4 种语速/音量控制
"""
import sys
import json
import time
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "CosyVoice"))
sys.path.insert(0, str(ROOT / "CosyVoice" / "third_party" / "Matcha-TTS"))

import torch
import torchaudio

MODEL_DIR = ROOT / "CosyVoice" / "pretrained_models" / "Fun-CosyVoice3-0.5B"
REF_DIR = ROOT / "reference_audio"
OUT_DIR = ROOT / "test_outputs_mac"
OUT_DIR.mkdir(exist_ok=True)

# 参考音频 prompt 文本
REF_TEXT_RAW = (REF_DIR / "ref_text.txt").read_text(encoding="utf-8").strip()
REF_TEXT = "You are a helpful assistant.<|endofprompt|>" + REF_TEXT_RAW

# 待测音色
VOICES = {
    "晓伊": "xiaoyi",
    "晓晓": "xiaoxiao",
    "晓萱": "xiaoxuan",
}

INSTRUCT_PREFIX = "You are a helpful assistant. "
ENDOFPROMPT = "<|endofprompt|>"

# ============================================================
# 测试定义
# ============================================================

# 测试 A: 克隆 + 预训练情感（与 Windows 对齐）
TEST_A = [
    ("克隆", None, "大家好，这是用零样本克隆生成的声音，听听看像不像？"),
    ("开心", "请非常开心地说一句话。", "太好了！今天真是开心的一天，我们一起去玩吧！"),
    ("伤心", "请非常伤心地说一句话。", "唉……我今天心情有点低落，感觉有些难过。"),
    ("生气", "请非常生气地说一句话。", "你怎么又迟到了！这已经是这周第三次了！"),
]

# 测试 B: 自由指令情感（macOS 独有）
TEST_B = [
    ("撒娇", "请用撒娇的语气说这句话。", "哥哥你就陪我玩一会儿嘛~"),
    ("温柔", "请用温柔深情的语气说这句话。", "有哥哥在身边，我就什么都不怕。"),
    ("害羞", "请用害羞的语气说这句话。", "你、你别老盯着我看啦..."),
    ("调皮", "请用调皮可爱的语气说这句话。", "嘿嘿，我偷偷把你的手机藏起来啦！"),
    ("惊讶", "请用惊讶的语气说这句话。", "哇！这个蛋糕好大好漂亮！"),
    ("平静", "请用平静从容的语气说这句话。", "没关系的，我们慢慢来就好。"),
]

# 测试 C: Fine-grained 标签（cross_lingual 模式）
TEST_C = [
    ("breath",   "[breath]今天好累啊[breath]，但是和哥哥在一起就很开心[breath]"),
    ("strong",   "我最<strong>最喜欢</strong>哥哥了！"),
    ("laughter", "哈哈哈哈[laughter]你太搞笑了[laughter]，我笑得肚子都疼了。"),
]

# 测试 D: 语速/音量
TEST_D = [
    ("慢速", "请用尽可能慢地语速说一句话。", "时间过得好慢啊，我在等你回来。"),
    ("快速", "请用尽可能快地语速说一句话。", "快点快点，我们要迟到了！"),
    ("大声", "Please say a sentence as loudly as possible.", "哥哥！！我在这里！！"),
    ("轻声", "Please say a sentence in a very soft voice.", "嘘...小声点，别吵醒别人..."),
]


def log(msg):
    print(msg, flush=True)


def synth_zero_shot(cosyvoice, voice_cn, prompt_wav, emo_cn, tts_text):
    """Zero-shot 克隆"""
    sub = OUT_DIR / voice_cn
    sub.mkdir(exist_ok=True)
    out_path = sub / f"{voice_cn}_{emo_cn}.wav"
    tag = f"{voice_cn}/{emo_cn}"
    t = time.time()
    try:
        gen = cosyvoice.inference_zero_shot(tts_text, REF_TEXT, prompt_wav, stream=False)
        audio = torch.concat([o["tts_speech"] for o in gen], dim=1)
        torchaudio.save(str(out_path), audio, cosyvoice.sample_rate)
        dur = audio.shape[1] / cosyvoice.sample_rate
        elapsed = time.time() - t
        rtf = elapsed / dur if dur > 0 else 0
        log(f"  ✅ {tag}: {dur:.2f}s, RTF={rtf:.2f}")
        return {"voice": voice_cn, "emotion": emo_cn, "category": "pretrained",
                "ok": True, "audio_sec": round(dur, 2), "rtf": round(rtf, 3),
                "file": str(out_path.relative_to(OUT_DIR))}
    except Exception as e:
        log(f"  ❌ {tag}: {e}")
        return {"voice": voice_cn, "emotion": emo_cn, "category": "pretrained",
                "ok": False, "error": str(e)}


def synth_instruct(cosyvoice, voice_cn, prompt_wav, emo_cn, instruct, tts_text, category="custom"):
    """Instruct2 情感控制"""
    sub = OUT_DIR / voice_cn
    sub.mkdir(exist_ok=True)
    out_path = sub / f"{voice_cn}_{emo_cn}.wav"
    tag = f"{voice_cn}/{emo_cn}"
    t = time.time()
    try:
        instruct_text = INSTRUCT_PREFIX + instruct + ENDOFPROMPT
        gen = cosyvoice.inference_instruct2(tts_text, instruct_text, prompt_wav, stream=False)
        audio = torch.concat([o["tts_speech"] for o in gen], dim=1)
        torchaudio.save(str(out_path), audio, cosyvoice.sample_rate)
        dur = audio.shape[1] / cosyvoice.sample_rate
        elapsed = time.time() - t
        rtf = elapsed / dur if dur > 0 else 0
        log(f"  ✅ {tag}: {dur:.2f}s, RTF={rtf:.2f}")
        return {"voice": voice_cn, "emotion": emo_cn, "category": category,
                "ok": True, "audio_sec": round(dur, 2), "rtf": round(rtf, 3),
                "file": str(out_path.relative_to(OUT_DIR))}
    except Exception as e:
        log(f"  ❌ {tag}: {e}")
        return {"voice": voice_cn, "emotion": emo_cn, "category": category,
                "ok": False, "error": str(e)}


def synth_fine_grained(cosyvoice, voice_cn, prompt_wav, tag_name, text):
    """Fine-grained 标签控制 (cross_lingual 模式)"""
    sub = OUT_DIR / f"{voice_cn}_fine"
    sub.mkdir(exist_ok=True)
    out_path = sub / f"{tag_name}.wav"
    t = time.time()
    try:
        full_text = INSTRUCT_PREFIX + ENDOFPROMPT + text
        gen = cosyvoice.inference_cross_lingual(full_text, prompt_wav, stream=False)
        audio = torch.concat([o["tts_speech"] for o in gen], dim=1)
        torchaudio.save(str(out_path), audio, cosyvoice.sample_rate)
        dur = audio.shape[1] / cosyvoice.sample_rate
        elapsed = time.time() - t
        rtf = elapsed / dur if dur > 0 else 0
        log(f"  ✅ {tag_name}: {dur:.2f}s, RTF={rtf:.2f}")
        return {"voice": voice_cn, "emotion": tag_name, "category": "fine_grained",
                "ok": True, "audio_sec": round(dur, 2), "rtf": round(rtf, 3),
                "file": str(out_path.relative_to(OUT_DIR))}
    except Exception as e:
        log(f"  ❌ {tag_name}: {e}")
        return {"voice": voice_cn, "emotion": tag_name, "category": "fine_grained",
                "ok": False, "error": str(e)}


def main():
    log("=" * 60)
    log("CosyVoice3 macOS 完整能力测试")
    log("=" * 60)
    log(f"音色: {', '.join(VOICES)}")
    log(f"MPS: {torch.backends.mps.is_available()}")
    log(f"torch: {torch.__version__}")

    from cosyvoice.cli.cosyvoice import AutoModel
    log("\n加载模型中（首次较慢）...")
    t0 = time.time()
    cosyvoice = AutoModel(model_dir=str(MODEL_DIR))
    log(f"✅ 模型加载完成 {time.time() - t0:.1f}s")

    results = []

    # ---- 测试 A & B: 克隆 + 预训练情感 + 自由指令情感 ----
    for voice_cn, prefix in VOICES.items():
        ref = REF_DIR / f"{prefix}_neutral.wav"
        log(f"\n=== 音色: {voice_cn} ({ref.name}) ===")
        if not ref.exists():
            log(f"  ⚠️ 参考音频缺失，跳过: {ref}")
            continue

        prompt_wav = str(ref)

        # 测试 A: 克隆 + 预训练情感
        log(f"\n  📝 测试 A: 克隆 + 预训练情感")
        for emo_cn, instruct, text in TEST_A:
            if instruct is None:
                results.append(synth_zero_shot(cosyvoice, voice_cn, prompt_wav, emo_cn, text))
            else:
                results.append(synth_instruct(cosyvoice, voice_cn, prompt_wav, emo_cn, instruct, text, "pretrained"))

        # 测试 B: 自由指令情感
        log(f"\n  📝 测试 B: 自由指令情感（mei 映射）")
        for emo_cn, instruct, text in TEST_B:
            results.append(synth_instruct(cosyvoice, voice_cn, prompt_wav, emo_cn, instruct, text, "custom"))

        # 测试 C: Fine-grained 标签
        log(f"\n  📝 测试 C: Fine-grained 控制标签")
        for tag_name, text in TEST_C:
            results.append(synth_fine_grained(cosyvoice, voice_cn, prompt_wav, tag_name, text))

    # ---- 测试 D: 语速/音量（用晓伊）----
    log(f"\n=== 测试 D: 语速/音量控制 ===")
    xiaoyi_ref = str(REF_DIR / "xiaoyi_neutral.wav")
    if Path(xiaoyi_ref).exists():
        for name, instruct, text in TEST_D:
            results.append(synth_instruct(cosyvoice, "晓伊", xiaoyi_ref, name, instruct, text, "speed_volume"))

    # ---- 汇总 ----
    ok = [r for r in results if r.get("ok")]
    log("\n" + "=" * 60)
    log(f"完成: {len(ok)}/{len(results)} 成功")
    if ok:
        log(f"平均 RTF: {sum(r['rtf'] for r in ok) / len(ok):.3f}")

    # 按类别汇总
    for cat in ["pretrained", "custom", "fine_grained", "speed_volume"]:
        cat_results = [r for r in ok if r.get("category") == cat]
        if cat_results:
            avg_rtf = sum(r['rtf'] for r in cat_results) / len(cat_results)
            log(f"  {cat}: {len(cat_results)} 条, 平均 RTF={avg_rtf:.3f}")

    log(f"音频目录: {OUT_DIR}")
    with open(OUT_DIR / "result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    log("=" * 60)


if __name__ == "__main__":
    main()
