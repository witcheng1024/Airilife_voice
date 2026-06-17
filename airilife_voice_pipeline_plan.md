# AiriLife 桌宠语音管线方案

> 目标: 给 AiriLife 桌宠模式加实时语音交互能力
> 日期: 2026-06-11 创建 / 2026-06-12 GPT 5.5 调研整合
> 状态: V2 方案确定，待启动 Phase 0 技术判定

---

## 0. 总体判断（GPT 5.5 调研结论）

**方向可行，但要把风险拆开看。**

"本地语音 I/O + DeepSeek API 大脑 + 情感标签 + 本地发声" 产品路线可行；
"从 0 训练 150M mini-Talker 并在 7-8 天内达到自然稳定流式情感可控" 风险偏高。

**分两条线推进：**

- **V1 产品可用线**（3-5 天）: 用成熟本地 TTS（CosyVoice2 优先），先验证桌宠体验、情感、打断、显存、延迟
- **V2 自研 Talker 线**（2-6 周）: 用成熟 TTS 做 teacher 蒸馏/微调 mini-Talker，不从零训练

### 可行性评分

| 模块 | 可行性 | 说明 |
|------|--------|------|
| 本地 ASR: faster-whisper | **高** | 成熟，支持 int8 量化 |
| DeepSeek API 大脑 | **高** | 支持 `stream=true` SSE，有 logprobs，无 hidden states |
| 情感标签控制文本 | **高** | special token 方案可行 |
| 本地 TTS 直接发声 | **中高** | CosyVoice2/Spark/F5/Fish 均可尝试 |
| 从零训练 mini-Talker | **中低** | 5000-10000 条数据不够，需 3-10 万条 |
| 4070 8GB 部署 | **中高** | 够跑，但显存预算要保守估 |
| 7-8 天完成完整方案 | **偏乐观** | V1 POC 3-5 天可行，含自研 Talker 不止 1 周 |

---

## 1. 需求与约束

### 1.1 核心需求

- AiriLife 桌宠（mei）需要实时语音对话能力
- 用户说话 → 桌宠用 mei 的声音回复，且带有情感表达
- mei 人设: 活泼可爱的年轻女声，ENFP 性格，兄妹关系（非主仆）

### 1.2 硬件约束

| 组件 | 硬件 | 显存 | 用途 |
|------|------|------|------|
| 用户本地 | RTX 4070 | **8GB** | 语音 I/O + Live2D |
| 训练服务器 | 8 卡 4090 + 8 卡 B200 | 充裕 | 训练和数据生成 |

### 1.3 修正后的显存预算（GPT 5.5 纠正）

原方案按"裸权重"估 2.1GB 偏乐观。实际运行需要考虑:
- PyTorch/CUDA runtime 额外占用
- TTS 模型的 tokenizer/flow/vocoder/KV cache/mel buffer
- Live2D + Electron/渲染管线显存波动
- 多进程 CUDA 初始化不复用
- 流式播放的音频 chunk queue、ASR buffer

**保守峰值估计:**

| 组件 | 峰值显存 |
|------|---------|
| Live2D 桌宠 | 0.8-1.5GB |
| faster-whisper small (int8_float16) | 0.5-1.2GB |
| 本地 0.5B TTS (FP16/mixed) | 1.5-3.0GB |
| 小 Talker 150-300M + codec decoder | 0.8-1.8GB |
| CUDA/PyTorch/碎片/缓冲 | 1.0-2.0GB |
| **合计** | **4.5-7.5GB** |

**结论: 峰值 6GB 可控、7GB 警戒、8GB OOM。设计时按 6GB 内规划。**

工程建议:
- ASR 用 `faster-whisper` 的 `int8_float16` 或 `int8`，不上 medium/large
- TTS 和 ASR 放同一个 Python/Torch 服务，避免多进程重复 CUDA 初始化
- VAD、回声消除、音频播放、tag parser 放 CPU
- 本地 TTS 优先 0.5B 级，不上 1.5B+
- 先 FP16；INT8/ONNX/TensorRT 是优化项，不是第一天必做
- **INT4 不建议一开始做**，TTS 比 LLM 更容易因量化损失出音质问题

### 1.4 架构约束

- **大脑 = DeepSeek API**（云端），不可替换，提供推理能力
- **语音 I/O = 本地 4070**，低延迟实时交互
- 先做 **standalone POC**，验证效果后再集成到 AiriLife/PeroCore

---

## 2. 整体架构

### 2.1 V1 架构（产品 POC，不自训 Talker）

```
┌─── 用户 4070 8GB ─────────────────────────────────────────────┐
│                                                                │
│  麦克风                                                         │
│    │                                                           │
│    ▼                                                           │
│  VAD (CPU) + AEC (CPU)                                         │
│    │                                                           │
│    ▼                                                           │
│  ① faster-whisper small (int8, GPU) → 文本                      │
│                                       │                        │
│  ② ───────────────→ DeepSeek API (SSE 流式) ←────────────────│
│                        │                                       │
│                        │ system prompt: mei 人设 + 情感协议      │
│                        │                                       │
│  ③ ←── text token stream ─────────────────────────────────────│
│           │                                                    │
│           ▼                                                    │
│     Emotion/Chunk Parser (CPU)                                 │
│     解析 [emotion:intensity] + phrase chunk                    │
│           │                                                    │
│           ▼                                                    │
│     CosyVoice2-0.5B (GPU, 流式)                                │
│           │                                                    │
│           ▼                                                    │
│     Audio Queue + Barge-in Controller                          │
│           │                                                    │
│           ▼                                                    │
│     扬声器 + Live2D 口型/表情同步                               │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 2.2 V2 架构（自研 mini-Talker 替换）

```
V1 管线不变，仅替换 CosyVoice2 → mini-Talker:

  Emotion/Chunk Parser
        │
        ▼
  mini-Talker (~200-350M, GPU)
  输入: text chunk + emotion + intensity + style
  输出: audio codec tokens
        │
        ▼
  Codec Decoder (流式) → PCM
```

### 2.3 三个环节

| 环节 | 位置 | V1 选型 | V2 选型 | 状态 |
|------|------|---------|---------|------|
| ① 语音输入 | 本地 4070 | faster-whisper small | 同 V1 | 成熟 |
| ② 大脑推理 | 云端 DeepSeek | SSE + 情感标签协议 | 同 V1 | 成熟 |
| ③ 语音输出 | 本地 4070 | CosyVoice2-0.5B | mini-Talker (蒸馏) | V1 待验证 |

---

## 3. 情感控制方案

### 3.1 DeepSeek 输出协议（GPT 5.5 改进: 流式友好格式）

**不用 XML**（`<happy>...</happy>`），原因: 流式 token 中可能出现半个标签，parser 复杂。

**改用 chunk 协议:**

```
[happy:0.8] 今天也太棒啦！
[pause:200]
[playful:0.7] 不过哥哥你是不是又熬夜了？
```

解析规则:
- `[emotion:intensity]` 设置当前情感状态 + 强度 (0-1)
- `[pause:ms]` 立即 flush 当前 chunk 并插入静音
- 文本按短语/标点自然切 chunk
- 未闭合 tag 用状态机处理，不等完整解析

### 3.2 情感标签体系（10 类 + 强度 + 说话参数）

不仅传 emotion id，而是传完整控制参数:

```json
{
  "emotion": "playful",
  "intensity": 0.7,
  "speed": 1.05,
  "pitch": 0.1,
  "pause_after_ms": 180,
  "style_prompt": "俏皮、明亮、像妹妹一样调皮，但不要太夸张"
}
```

10 类情感标签:

| 标签 | 含义 | TTS 控制建议 |
|------|------|-------------|
| `happy` | 开心/温暖 | 明亮、温暖、轻快，笑意明显但不过嗲 |
| `excited` | 热情兴奋 | 语速略快，音高略上扬，能量高 |
| `sad` | 难过伤心 | 音量稍低，语速变慢，尾音下沉 |
| `calm` | 平静从容 | 平稳、清晰、少起伏 |
| `angry` | 不满愤怒 | 轻微不满/吐槽，不要真的吼 |
| `shy` | 紧张羞涩 | 语速略慢，轻声，尾音短促，少量停顿 |
| `love` | 深情陶醉 | 温柔、贴近、轻声，**避免成人化/过度暧昧** |
| `playful` | 调皮可爱 | 俏皮、上扬、带一点"哼哼"的感觉 |
| `surprise` | 惊讶 | 起句音高上扬，短促惊讶 |
| `pause` | 停顿 | 转成 pause duration，不送进 TTS |

**注意: love/shy 要约束表达边界**，保持温情、依赖、撒娇，不走向过度暧昧（兄妹人设）。

### 3.3 DeepSeek system prompt 要点

1. mei 的 ENFP 兄妹人设
2. 可用的情感标签列表 + 强度含义
3. 输出格式协议（chunk 协议）
4. 一条回复中可以切换多种情感
5. 短句优先（适合 barge-in），避免过长回答

---

## 4. V1: 本地 TTS 选型

### 4.1 首选: CosyVoice2-0.5B

- 参数 0.5B，4070 8GB 可行
- 中文能力强
- **支持 bidirectional streaming，首包 150ms**
- 支持 zero-shot voice clone + instruct 风格控制
- 8 种情感: 高兴、悲伤、惊讶、愤怒、恐惧、厌恶、冷静、严肃
- 支持自然语言情感指令: `"你能用高兴的情感说吗？<|endofprompt|>今天真是太开心了！"`
- FSQ tokenizer（finite-scalar quantization），codebook 利用率高
- License: Apache-2.0

**风险:** 情感颗粒度不等于自定义 10 类；撒娇/傲娇/害羞/love 靠 prompt + ref audio 调

### 4.2 备选: Spark-TTS (0.5B)

- LLM-based TTS，BiCodec，Apache-2.0
- 支持中英 + zero-shot voice cloning
- 低显存候选，适合和 CosyVoice2 横向对比
- 公开资料对细粒度情感控制确定性不足

### 4.3 备选: F5-TTS (0.3B)

- 100K 小时多语数据，zero-shot 能力强
- RTF ~0.15，速度快
- 原版情感控制弱；社区 Emotional-CFG 版本只支持 5 类（Neutral/Happy/Sad/Angry/Surprised）
- 覆盖不了完整 10 标签

### 4.4 备选: Fish Speech S2

- 10-30s 短参考样本快速 voice clone
- 能捕捉音色、风格、情感倾向
- 精确情感控制靠 reference/style，无稳定 10 类 API
- 商业使用条款需逐项核查

### 4.5 验证线: IndexTTS2

- **核心优势: 情感表达与 speaker identity 解耦**
- 可用 A 的音色 + B 的情感，天然适合 "mei 音色 + 多情感"
- 风险: 模型规模更大、流式不如 CosyVoice2 直接、license 复杂（非纯 Apache-2.0）
- **放在音色/情感质量验证线，不作为 V1 默认**

### 4.6 V1 推荐验证流程

```
Phase 0 第 3 步:
  CosyVoice2 / Spark / F5 / Fish 各合成 30 条 mei 文本
  → 人工听感排名
  → 选 V1 TTS
```

---

## 5. V2: mini-Talker 自研方案

### 5.1 为什么不能直接用 Qwen3-Omni Talker

调研确认:
- Qwen3-Omni Talker **需要 Thinker 的 hidden_states**（layer 18 投影）
- Thinker 是 30B MoE，8GB 装不下
- DeepSeek API 不返回 hidden states，只返回 text tokens + logprobs
- Qwen3-Omni **没有原生情感控制**（无 emotion embedding/prosody conditioning）
- **结论: Qwen Talker 权重无法脱离 Thinker 独立使用，不要走这条路**

### 5.2 原方案 mini-Talker 的问题（GPT 5.5 纠正）

原架构 `concat(emotion_embed, text_embed, audio_tokens) → Transformer → next_audio_token` 存在问题:

1. **文本到语音不是纯 next-token 问题**，还要学对齐和时长（每个字对应多少 audio token？哪里停顿？）
2. **多码本 flatten 后序列很长**，流式 KV cache/训练稳定性/首包延迟都恶化
3. **DeepSeek 流式 token 太细**，不能一来 token 就发音，需要短语级上下文才有好韵律
4. **情感不是全局 id 就够**，需要 `emotion + intensity + pause + speaking_rate`

### 5.3 改进后的 mini-Talker 架构

```
text chunk + emotion + intensity + style
      │
      ▼
  Text Encoder / Phoneme Encoder
      │
      ▼
  Duration / Alignment Predictor      ← 预测每字对应多少 audio token
      │
      ▼
  Semantic/Acoustic Token Predictor   ← 用 cross-attention 而非 concat
      │
      ▼
  Multi-Codebook Parallel Heads       ← 多码本并行预测，不 flatten
  (或 DepFormer / delay pattern)
      │
      ▼
  Streaming Codec Decoder → PCM
```

关键改进:
- **文本条件用 cross-attention 或 prefix-encoder**，不用 concat
- **增加 duration/pause predictor**
- **多码本并行预测**（Moshi/Mimi 风格 depformer 或 SNAC 多尺度），不 flatten 后纯 AR
- **emotion = emotion_id + intensity + style_prompt embedding**
- **训练目标: teacher distillation**，不从零训练

模型规模: 150M 第一版 → 250-350M 更稳。

### 5.4 Audio Tokenizer 选择（GPT 5.5 推荐）

| 优先级 | Tokenizer | 理由 |
|--------|-----------|------|
| **1st** | **Mimi** (Kyutai/Moshi) | 12.5Hz 极低 token rate，全因果，专为 LLM 工作负载设计 |
| **2nd** | **CosyVoice2 FSQ** | 和 CosyVoice2 teacher 一致，蒸馏更自然，省 codec 适配风险 |
| **3rd** | **SNAC** | 多尺度结构，粗粒度 tokens 采样频率低，Mini-Omni 也用 |
| 不推荐 | DAC | 高保真但 token rate 高，AR Talker 训练和流式推理更重 |

### 5.5 训练数据需求（GPT 5.5 纠正: 原估不足）

5200 条 GRPO 文本 ≈ 5.8-8.7 小时音频，**不够从零训练 codec LM**。

对比参考:
- Mini-Omni 为 speech output 用 VoiceAssistant-400K 数据集
- VALL-E 预训练用 60K 小时语音

**修正后的数据规模:**

| 阶段 | 数据量 | 目标 |
|------|--------|------|
| POC 听感验证 | 500-1000 条 | 选 TTS、选音色、验证情感映射 |
| V1 产品调参 | 5200 条 | persona + emotion prompt + 回归测试 |
| mini-Talker 蒸馏起步 | **3万-5万条** | 单音色、短句、10 情感基本可训 |
| mini-Talker 稳定版本 | **10万-30万条** | 更稳的韵律、长句、打断、口语化 |

数据增强策略:
- 每条文本生成 2-4 个情感强度版本
- 增加 pause 版本: 短停顿、长停顿、句尾上扬、轻笑、叹气
- 用 DeepSeek 扩写 mei 口语数据（去重）
- 加入短反馈句: "嗯嗯""诶？""哥哥你等一下""才不是呢"
- 加入 barge-in 友好短句
- 质量过滤: Whisper WER < 5% + speaker similarity + emotion classifier 粗筛 + 人工抽检

### 5.6 训练路线: 蒸馏，不从零

```
teacher TTS (CosyVoice2/豆包/IndexTTS2)
      │ 生成高质量音频
      ▼
chosen audio tokenizer (Mimi/CosyVoice FSQ)
      │ 编码
      ▼
audio tokens
      │
student mini-Talker 学习:
  (text, emotion, intensity, pause) → audio tokens
```

**优势: teacher 已经学会发音/对齐/韵律，student 只需压缩到更小模型 + 学习情感条件映射。**

---

## 6. 流式策略（GPT 5.5 纠正: 不要 token-level）

### 6.1 原方案问题

原设想 "DeepSeek 出几个 token → Talker 就生成 audio" 对端到端 omni 模型成立，但对 "DeepSeek 文本 API + 本地 TTS/Talker" 不成立。中文需要短语级上下文才有好韵律、连读、停顿。

### 6.2 改进: Phrase/Chunk-level

```
DeepSeek SSE token stream
      │
      ▼
Tag Parser: 识别 [emotion:intensity]
      │
      ▼
Incremental Sentence Segmenter
      │
      ▼
按短语/子句切 chunk（8-16 汉字或遇标点）
      │
      ▼
TTS Queue (最多 2-3 个 chunk)
      │
      ▼
边合成边播放 + Barge-in
```

Chunk 规则:
- 拿到第一个 emotion tag 后才启动该 chunk
- 中文至少等 8-16 个汉字，或遇 `，。！？～…`
- 遇 `[pause]` 立即 flush
- 单 chunk 目标 0.8-2.5 秒音频
- 不超过 20-30 汉字/ chunk
- 每个 chunk 保留 `emotion + intensity + speaking_rate + pause_after_ms`

---

## 7. 打断机制（Barge-in，一级功能）

桌宠语音对话最影响体验的功能。

```
播放 TTS 时
      │
      ▼
麦克风仍然开着
      │
      ▼
AEC 抑制扬声器回声（关键！否则 mei 声音触发 ASR）
      │
      ▼
VAD 检测用户真实说话
      │
      ▼
连续 200-300ms 置信语音
      │
      ▼
立即 stop audio output
清空 TTS chunk queue
cancel 当前 DeepSeek SSE request
      │
      ▼
ASR 新 utterance
      │
      ▼
"用户打断了你上一句话" 作为上下文发给 DeepSeek
```

关键工程点:
- **不要等 ASR 完整识别后才打断**；VAD 检测到用户说话就先停播
- 播放队列支持 `flush()`，不只 pause
- DeepSeek SSE 支持取消 HTTP request
- TTS 推理线程支持 cancel flag
- 对已播放/未播放/已生成未播放的 chunk 分开管理

**POC 指标: 用户开始说话后 150ms 内停止播放，最晚 300ms。**

---

## 8. 通信协议

### 8.1 本地: 不用 WebSocket/WebRTC

全部在本机运行时，最佳方案是:
- 本地进程 / localhost gRPC / named pipe / ZeroMQ / ring buffer

### 8.2 远程 fallback（如果 Talker 放服务器）

| 用途 | 协议 |
|------|------|
| 麦克风/扬声器音频流 | **WebRTC**（为实时媒体设计，AEC/jitter buffer/NAT traversal） |
| 控制消息/text token/emotion/状态事件 | **WebSocket**（双向交互通信） |

---

## 9. 延迟预估

### V1 (CosyVoice2)

| 环节 | 延迟 |
|------|------|
| VAD + AEC | ~20ms |
| faster-whisper small (int8, 4070) | ~150ms |
| DeepSeek API 首 token | ~500ms-1.5s (**主瓶颈**) |
| DeepSeek 后续 token streaming | ~30-50ms/token |
| Chunk parser | ~1ms |
| CosyVoice2 首 audio chunk (streaming) | ~150ms |
| Audio playback buffer | ~50ms |
| **总首字延迟** | **~900ms-1.9s** |

### V2 (mini-Talker + Mimi codec, 12.5Hz)

| 环节 | 延迟 |
|------|------|
| 前段同 V1 | ~700ms-1.7s |
| mini-Talker 首 audio chunk | ~50-80ms |
| Mimi decoder | ~20-30ms |
| **总首字延迟** | **~800ms-1.8s** |

对比: GPT-4o ~300-500ms，传统级联 TTS ~2-4s。

---

## 10. 落地路线

### Phase 0: 技术判定（1 天）

4 个硬测试:

1. **显存测试**: 4070 8GB 同时跑 Live2D + faster-whisper + 一个 TTS，测显存峰值
2. **DeepSeek SSE 测试**: 输出情感标签，测首 token 和首个完整 chunk 延迟
3. **TTS 横评**: CosyVoice2 / Spark / F5 / Fish 各合成 30 条 mei 文本，人工听感排名
4. **Barge-in demo**: TTS 播放时说话，能不能立即停

通过标准:
- GPU 峰值 < 6.5GB
- 首个可播放 audio chunk < 2 秒
- 情感标签能被 parser 稳定捕获
- 打断 < 300ms

### Phase 1: V1 standalone POC（3-5 天）

```
faster-whisper + DeepSeek SSE + emotion parser + CosyVoice2 流式 + barge-in
```

目标:
- mei 能说话，带基础情感
- 能打断
- 显存稳定
- 角色风格稳定
- 延迟可接受

### Phase 2: 数据生成和评测集（1-2 周）

用 5200 条 GRPO 文本 × 多 TTS 引擎生成:
- CosyVoice2 版本
- 豆包/火山 teacher 版本
- IndexTTS2 版本
- F5/Fish 参考版本

每条保留:
```json
{
  "text": "...",
  "emotion": "happy",
  "intensity": 0.8,
  "speaker": "mei_v1",
  "tts_engine": "cosyvoice2",
  "audio_path": "...",
  "wer": 0.02,
  "speaker_score": 0.81,
  "emotion_score": 0.74,
  "human_rating": 4
}
```

选出最像 mei 的 teacher。

### Phase 3: mini-Talker 蒸馏（2-6 周）

- Teacher: Phase 2 选出的最佳 TTS
- 数据: 3万-10万条（teacher 批量生成）
- 模型: 200-350M
- Codec: Mimi 或 CosyVoice FSQ
- 训练: 8 卡 4090 或 B200

### Phase 4: 4070 部署优化

- FP16 首版 → 必要时 INT8
- ONNX/TensorRT 只优化 decoder/TTS 热点
- 预热模型
- 固定最大 chunk 长度
- TTS queue 不超过 2-3 个 chunk
- 用户说话即 cancel

---

## 11. 音色确定方案

### 11.1 候选方案

| 方案 | 描述 | 适合阶段 |
|------|------|---------|
| **豆包灿灿 2.0** | 预设音色，22 种情感/风格，emotion_scale 1-5 | Phase 0-1 听感验证 + teacher 数据 |
| **豆包声音复刻** | 上传 30-60s 音频 → `S_` 音色 ID | Phase 0 验证情感控制能力 |
| **CosyVoice2 clone** | 开源零样本克隆 | Phase 1-2 本地 TTS + 数据生成 |
| **IndexTTS2 解耦** | A 音色 + B 情感 | Phase 2 质量验证线 |

### 11.2 豆包情感映射

| mei 标签 | 豆包 emotion | 备注 |
|----------|-------------|------|
| happy | `comfort` 或 `pleased` | |
| excited | `happy` + scale=5 | |
| sad | `sad` | |
| calm | 通用 / `narrator` | |
| angry | `angry` | |
| shy | `lovey-dovey` (撒娇) | |
| love | `charming` (娇媚) | |
| playful | `tsundere` (傲娇) | |
| surprise | `surprise` | |

### 11.3 豆包复刻音色情感验证（Phase 0 必做）

`S_` 开头复刻音色是否支持 emotion 参数控制**未确认**。Gating test:
1. 同一个 `S_` 音色
2. 同一句文本
3. 10 个 emotion × 3 个 scale = 30 条音频
4. 人工听感 + emotion classifier + speaker similarity
5. 如果差异不明显，就不用它做情感 teacher

### 11.4 豆包定位

**适合**: 早期 benchmark、teacher 数据生成、音色方向验证、情感参考音频
**不适合**: 运行时依赖（用户要求本地运行）

---

## 12. 原方案修改总结

### 保留

- DeepSeek API 作为大脑
- 情感标签由 LLM 输出
- 本地 ASR/TTS
- standalone POC 先行
- 10 类情感标签体系
- 5200 条 GRPO 数据做 persona/emotion 基础

### 修改

- ~~V1 训练 mini-Talker~~ → V1 用 CosyVoice2 本地 TTS
- ~~豆包做运行时~~ → 豆包只做 teacher 数据和音色验证
- ~~数据 5000-10000 条~~ → mini-Talker 至少 3万-10万条
- ~~Audio tokenizer DAC 优先~~ → Mimi / CosyVoice FSQ 优先
- ~~DeepSeek token-level 流式~~ → phrase/chunk-level 流式
- ~~XML 情感格式~~ → `[emotion:intensity]` chunk 协议
- ~~7-8 天计划~~ → V1 POC 3-5 天 + V2 自研 2-6 周
- ~~简单 concat decoder~~ → cross-attention + duration predictor + multi-codebook parallel
- ~~情感只传 id~~ → emotion + intensity + speed + style_prompt

---

## 13. 相关参考

### 13.1 已有项目文件

- Qwen3-Omni GRPO 训练工作报告: `/PROJECT/qwen3omni_v1_to_v10工作报告.md`
- CTC adapter 方案: `/PROJECT/speech_llm/CTC_ADAPTER_PLAN.md`
- CTC 训练方案: `/PROJECT/speech_llm/CTC_REPLICATED_TRAINING.md`

### 13.2 Zotero 文献

- Mini-Omni: `Xie和Wu - 2024 - Mini-Omni Language Models Can Hear, Talk While Thinking in Streaming`
- Qwen2.5-Omni TR: `Xu 等 - 2025 - Qwen2.5-Omni Technical Report`
- Speech LLMs CTC: `Deng 等 - 2026 - Speech LLMs are Contextual Reasoning Transcribers`
- MOSS-Audio: `Yang 等 - 2026 - MOSS-Audio Technical Report`

### 13.3 开源模型 / 工具

- CosyVoice2: `FunAudioLLM/CosyVoice2-0.5B` (Apache-2.0)
- Spark-TTS: `SparkAudio/Spark-TTS` (Apache-2.0)
- F5-TTS: `swivid/F5-TTS` (0.3B, 100K 小时)
- Fish Speech: `fishaudio/fish-speech`
- IndexTTS2: 音色-情感解耦 TTS (license 复杂)
- faster-whisper: `SYSTRAN/faster-whisper` (int8 量化)
- Mimi codec: Kyutai/Moshi 流式神经音频 codec (12.5Hz)
- SNAC: `hubertsiuzdak/snac` (多尺度 neural audio codec)

### 13.4 API / 服务

- DeepSeek API: `stream=true` SSE, logprobs, 无 hidden states
- 豆包 TTS API: https://www.volcengine.com/docs/6561/97465
  - 灿灿 2.0: 22 种情感/风格, emotion_scale 1-5, SSE 流式
  - 声音复刻: `S_` 音色 ID

### 13.5 关键技术参考

- Qwen3-Omni 源码: `transformers/models/qwen3_omni_moe/modeling_qwen3_omni_moe.py`
  - Talker: `Qwen3OmniMoeTalkerForConditionalGeneration` (MoE Transformer + CodePredictor)
  - Code2Wav: `Qwen3OmniMoeCode2Wav` (因果卷积, chunked_decode)
  - 32 codebook RVQ, codebook_size=2048
- EmoSteer-TTS: training-free 情感控制 (activation steering)
- EmoKnob: few-shot 情感控制框架
- WeSCon (NeurIPS 2025 Spotlight): word-level 情感+语速控制, 基于 CosyVoice2
