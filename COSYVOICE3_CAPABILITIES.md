# CosyVoice3 能力图谱（源码级调研）

> 日期: 2026-06-16
> 来源: FunAudioLLM/CosyVoice GitHub 源码 + CosyVoice3 Technical Report
> 模型: Fun-CosyVoice3-0.5B-2512

---

## 1. 重要结论

### Ollama 兼容性
**Ollama 不能直接运行 CosyVoice3。** Ollama 是 LLM 推理引擎（支持 Llama/Mistral/Qwen 等 GGUF 格式），不支持 TTS 模型。CosyVoice3 的模型格式是 PyTorch .pt 文件，不是 Ollama 支持的格式。

### CosyVoice3 vs CosyVoice2 情感能力对比

| 能力维度 | CosyVoice2 | CosyVoice3 | 评价 |
|----------|-----------|-----------|------|
| Instruct 情感 | **8 种** (高兴/悲伤/惊讶/愤怒/恐惧/厌恶/冷静/严肃) | **3 种** (开心/伤心/生气) | **CosyVoice3 反而更少！** |
| 方言控制 | 10+ | **17 种** | CosyVoice3 更多 |
| 零样本克隆 | 好 | **更好** (CER↓, SIM↑) | CosyVoice3 质量更高 |
| 情感克隆 | 一般 | **强** (happy 81.8%, sad 96.4%) | 从参考音频迁移情感 |
| 流式延迟 | 150ms | 150ms | 相同 |
| 模型大小 | 0.5B | 0.5B | 相同 |

**关键发现**: CosyVoice3 的 `instruct_list`（预训练指令集）只有 3 种情感，比 CosyVoice2 的 8 种少！但它的 **zero-shot 情感克隆能力**（从参考音频迁移情感）显著更强。

### 对 AiriLife 的影响

对于 mei 的 10 种情感标签，推荐混合策略：
1. **音色定义**: 用 zero-shot 找一段活泼可爱女声参考音频
2. **情感控制**: 尝试 `inference_instruct2` + 自由文本指令（如"请用撒娇的语气说"）
3. **备选方案**: 如果 instruct2 对非预训练情感效果差，考虑用 CosyVoice2（8 种预训练情感）

---

## 2. Instruct Mode (inference_instruct2)

### 预训练指令集 (`common.py` instruct_list)

#### 方言 (17种)
```
广东话, 东北话, 甘肃话, 贵州话, 河南话, 湖北话, 湖南话,
江西话, 闽南话, 宁夏话, 山西话, 陕西话, 山东话, 上海话,
四川话, 天津话, 云南话
```

#### 音量 (2种)
```
Please say a sentence as loudly as possible.
Please say a sentence in a very soft voice.
```

#### 语速 (2种)
```
请用尽可能慢地语速说一句话。
请用尽可能快地语速说一句话。
```

#### 情感 (3种) ← 注意：只有3种！
```
请非常开心地说一句话。
请非常伤心地说一句话。
请非常生气地说一句话。
```

#### 风格 (2种)
```
我想体验一下小猪佩奇风格，可以吗？
你可以尝试用机器人的方式解答吗？
```

### 使用方式
```python
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')

# instruct2 接受自由文本指令，理论上可以尝试非预训练情感
for i, j in enumerate(cosyvoice.inference_instruct2(
    '今天天气真好啊，我们出去玩吧！',
    'You are a helpful assistant. 请用撒娇的语气说这句话。<|endofprompt|>',  # 自由指令
    './reference_audio.wav',  # 音色参考
    stream=False
)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

### 指令格式
```
You are a helpful assistant. {具体指令}。<|endofprompt|>
```

---

## 3. Fine-Grained Control Tags

### CosyVoice3Tokenizer 支持的标签 (`tokenizer.py` L274-285)

| 标签 | 功能 | 用法 |
|------|------|------|
| `[breath]` | 呼吸声 | 文本中插入 |
| `<strong>文本</strong>` | 强调/重音 | 包裹文本 |
| `[noise]` | 背景噪声 | 文本中插入 |
| `[laughter]` | 笑声 | 文本中插入 |
| `<laughter>文本</laughter>` | 带笑说 | 包裹文本 |
| `[cough]` | 咳嗽 | 文本中插入 |
| `[clucking]` | 啧啧声 | 文本中插入 |
| `[accent]` | 口音标记 | 文本中插入 |
| `[quick_breath]` | 快速呼吸 | 文本中插入 |
| `[hissing]` | 嘶嘶声 | 文本中插入 |
| `[sigh]` | 叹息 | 文本中插入 |
| `[vocalized-noise]` | 声音噪声 | 文本中插入 |
| `[lipsmack]` | 咂嘴声 | 文本中插入 |
| `[mn]` | 嗯声 | 文本中插入 |

### 使用方式 (cross_lingual mode)
```python
# 在 cross_lingual 模式下使用 fine-grained 标签
for i, j in enumerate(cosyvoice.inference_cross_lingual(
    'You are a helpful assistant.<|endofprompt|>[breath]今天天气真好啊[breath]，我们出去玩吧！',
    './reference_audio.wav',
    stream=False
)):
    torchaudio.save(f'output_{i}.wav', j['tts_speech'], cosyvoice.sample_rate)
```

---

## 4. EMOTION 字典 (Whisper Tokenizer)

`tokenizer.py` L150-156:
```python
EMOTION = {
    "HAPPY": "HAPPY",
    "SAD": "SAD",
    "ANGRY": "ANGRY",
    "NEUTRAL": "NEUTRAL",
}
```

这是 Whisper tokenizer 内部使用的 emotion 标记（用于 ASR/情感识别），不是 TTS 情感控制。

---

## 5. AUDIO_EVENT 字典

`tokenizer.py` L136-149:
```python
AUDIO_EVENT = {
    "ASR": "ASR",
    "AED": "AED",
    "SER": "SER",
    "Speech": "Speech", "/Speech": "/Speech",
    "BGM": "BGM", "/BGM": "/BGM",
    "Laughter": "Laughter", "/Laughter": "/Laughter",
    "Applause": "Applause", "/Applause": "/Applause",
}
```

用于音频事件标注（ASR/AED/SER），不是 TTS 控制。

---

### 音色能力（已验证）

| 模型 | 内置音色 | 说明 |
|------|---------|------|
| CosyVoice v1 (300M-SFT) | ✅ 7 个（中文女/男、英文女/男、日语男、粤语女、韩语女） | SFT 训练，有固定音色 |
| CosyVoice2 (0.5B) | ✅ 有 spk2info.pt | 有预训练音色 |
| **CosyVoice3 (0.5B)** | **❌ 0 个** | **纯 zero-shot 克隆，必须提供参考音频** |

**验证方式**: `cosyvoice.list_available_spks()` 返回空列表，`spk2info.pt` 文件不存在。

`TTS_Vocal_Token`（`tokenizer.py` 中定义的 `TTS/B`, `TTS/O`, `TTS/Q` 等 20 个 token）只是 tokenizer 词汇表中的特殊 token，**不对应任何可用的说话人 embedding**。

---

## 7. 推理 API 完整列表

| 方法 | 模型 | 用途 | 关键参数 |
|------|------|------|---------|
| `inference_sft` | CosyVoice (300M-SFT) | 预设说话人合成 | `spk_id` |
| `inference_zero_shot` | CosyVoice/2/3 | 零样本声音克隆 | `prompt_text`, `prompt_wav` |
| `inference_cross_lingual` | CosyVoice/2/3 | 跨语言合成+细粒度控制 | `prompt_wav` |
| `inference_instruct` | CosyVoice (300M-Instruct) | 指令控制 (v1) | `spk_id`, `instruct_text` |
| `inference_instruct2` | CosyVoice2/3 | 指令控制 (v2) | `instruct_text`, `prompt_wav` |
| `inference_vc` | CosyVoice | 语音转换 | `source_wav`, `prompt_wav` |

### 通用参数
- `stream=False` / `True`: 是否流式输出
- `speed=1.0`: 语速倍率
- `text_frontend=True`: 是否启用文本前端处理
- `zero_shot_spk_id=''`: 预注册的零样本说话人 ID

---

## 8. 模型架构

```
CosyVoice3 = CosyVoice2 的子类

模型组件:
├── llm.pt          # LLM (Qwen-based) → 文本→语音 token
├── flow.pt         # Flow matching model → token→mel频谱
├── hift.pt         # HiFi-GAN vocoder → mel→波形
├── speech_tokenizer_v3.onnx  # 语音 tokenizer
├── campplus.onnx   # 说话人 embedding (CAM++)
├── spk2info.pt     # 说话人信息
└── CosyVoice-BlankEN/  # Qwen 预训练路径

配置文件: cosyvoice3.yaml
```

### AutoModel 自动选择
```python
cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B')
# 自动检测 cosyvoice.yaml / cosyvoice2.yaml / cosyvoice3.yaml
# 返回对应的 CosyVoice / CosyVoice2 / CosyVoice3 实例
```

---

## 9. macOS 兼容性评估

### 代码层面
```python
# cosyvoice.py: 无 CUDA 时自动降级
if torch.cuda.is_available() is False:
    load_trt, fp16 = False, False  # 禁用 TensorRT 和 FP16
```

### 依赖层面
| 依赖 | macOS 状态 |
|------|-----------|
| torch/torchaudio | ✅ MPS 加速可用 |
| onnxruntime | ✅ `sys_platform == 'darwin'` 用 CPU 版 |
| onnxruntime-gpu | ❌ 跳过 (仅 linux) |
| deepspeed | ❌ 跳过 (仅 linux) |
| tensorrt | ❌ 跳过 (仅 linux) |

### 预估
- **可以运行**，但速度会慢（CPU/MPS 推理）
- 模型加载 ~1.5-2GB 内存
- 单句推理可能 10-60 秒（vs GPU 上 1-3 秒）
- 足以验证功能和听感，不适合实时应用

---

## 10. CosyVoice3 论文关键数据

### 情感声音克隆成功率 (PilotTTS 论文 Table 4)
| 情感 | CosyVoice3 成功率 |
|------|------------------|
| Happy | 81.8% |
| Sad | **96.4%** |
| Fear | 80.0% |
| Angry | 80.1% |
| Contempt | 88.2% |
| Serious | **90.9%** |
| Surprise | 69.1% |
| Blue (忧郁) | 86.4% |
| Concern (关切) | 83.6% |
| Disgust (厌恶) | 52.7% |
| Psychology (内心独白) | **98.2%** |
| **平均 (基础)** | **83.8%** |

### 与 CosyVoice2 对比
| 指标 | CosyVoice2 | CosyVoice3-0.5B |
|------|-----------|----------------|
| 中文 CER ↓ | 4.08% | 3.89% |
| 英文 WER ↓ | 6.32% | 5.24% |
| 情感 Happy (text-related) | 84% | **92%** |
| 情感 Sad (text-related) | 72% | 70% |
| 情感 Angry (text-related) | 58% | **72%** |

---

## 11. AiriLife 情感映射建议

### mei 的 10 情感 → CosyVoice3 控制方案

| mei 标签 | 方案 A: instruct2 | 方案 B: zero-shot 情感参考音频 |
|----------|-------------------|-------------------------------|
| happy | "请非常开心地说" | ✅ 开心参考音频 |
| excited | "请用兴奋的语气说" (未验证) | ✅ 兴奋参考音频 |
| sad | "请非常伤心地说" | ✅ 伤心参考音频 |
| calm | "请用平静的语气说" (未验证) | ✅ 平静参考音频 |
| angry | "请非常生气地说" | ✅ 生气参考音频 |
| shy | "请用害羞的语气说" (未验证) | ✅ 害羞参考音频 |
| love | "请用温柔深情的语气说" (未验证) | ✅ 深情参考音频 |
| playful | "请用调皮可爱的语气说" (未验证) | ✅ 调皮参考音频 |
| surprise | "请用惊讶的语气说" (未验证) | ✅ 惊讶参考音频 |
| pause | 不送入 TTS，转成静音时长 | — |

### 推荐测试顺序
1. 先测 instruct2 预训练情感 (happy/sad/angry) → 确认基本功能
2. 测 instruct2 自由指令 (shy/playful/love) → 看泛化能力
3. 测 zero-shot 情感克隆 → 准备不同情感的参考音频
4. 对比 CosyVoice2 的 8 种预训练情感 → 决定是否回退
