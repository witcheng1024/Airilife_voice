# AiriLife Voice V2 Plan: Text -> DeepSeek -> CosyVoice3

> 目标: 先不做 ASR、不训练 mini-Talker，完成文本输入到 DeepSeek 多情感脚本，再由 CosyVoice3-0.5B 生成 mei 语音的可用管线。
> 日期: 2026-06-17
> 状态: V2 运行时方案，基于现有 CosyVoice3 测试结果修正

---

## 1. 结论

当前 V2 不再走自研 Talker 训练路线。

实践结果显示，`test_outputs_mac/晓萱` 和 `test_outputs_mac/晓萱_fine` 的听感已经可用，CosyVoice3-0.5B 对 zero-shot 音色克隆、自由 instruct、fine-grained 标签的效果足够支撑第一版产品体验。因此下一步重点不是训练，而是把以下工程链路做稳:

```
用户文本
  -> DeepSeek API 生成多情感表演脚本
  -> 脚本解析/校验/切分
  -> CosyVoice3 分段合成
  -> 音频拼接/crossfade/停顿处理
  -> 播放 + 保存调试产物
```

核心原则:
- 一句话可以有多种情感，但不要让 CosyVoice3 一次性吃完整句并期待自动完成复杂表演。
- 由 DeepSeek 负责把一句话拆成可执行的表演片段。
- 每个片段单独指定 `emotion`、`instruct`、`text`、`pause_after_ms`、可选 fine-grained 标签。
- CosyVoice3 按片段独立合成，最后做轻量拼接。
- 暂时不做 ASR、AEC、barge-in、mini-Talker、训练数据生成。

### 1.1 2026-06-18 本地延迟实测更新

测试环境: Windows + RTX 4070 Laptop GPU，`deepseek-v4-flash`，CosyVoice3-0.5B，`fp16=True`。当前 `onnxruntime` 只有 `CPUExecutionProvider`，日志显示 `CUDAExecutionProvider` 不可用，因此 speech tokenizer 仍是主要优化点。

关键结果:
- DeepSeek SSE 协议测试: `jsonl` 首个完整 segment 约 2.62s；紧凑 `json_object` 约 1.75-3.27s；`[emotion]文本[emotion]文本` 能更早吐 token，但容易先 flush 出“哥哥！”这种过短片段。
- 短首段 TTS: `哥哥！你终于来找我啦！` 非流式 4.90s；`stream=True` 首音频 4.22s、总耗时 4.82s。
- clean 听感基线: `single_clean.wav` 对应的整句 clean text + 综合 instruct，本地复测 15.08s 音频合成 18.27s，RTF 1.21。
- 端到端短输入: `我好无聊啊，陪我聊聊天吧`，DeepSeek JSON 2.42s，第一段 TTS 6.67s；如果模型已常驻预热，首段可播放约 9.1s。冷启动模型加载 20-50s，不可进入交互路径。

结论:
- V2/v0.2 作为“文本输入 -> 有情感 mei 语音输出”的 POC 可行。
- 作为实时对话还不达标，当前瓶颈不是 DeepSeek，而是本地 CosyVoice3 推理和 CPU ONNX 路径。
- 用户当前最满意的 `clean` 方案应作为音质基线；分段/流式是降低首听延迟的工程路径，不应牺牲 clean 听感作为默认目标。

---

## 2. 当前依据

### 2.1 已验证能力

项目已有测试覆盖:
- CosyVoice3-0.5B 本地加载和推理。
- 晓伊/晓晓/晓萱多音色 zero-shot 克隆。
- 预训练情感: 开心、伤心、生气。
- 自由 instruct: 撒娇、温柔、害羞、调皮、惊讶、平静。
- fine-grained 标签: `[breath]`、`<strong>`、`[laughter]`。
- 语速/音量控制: 慢速、快速、大声、轻声。

当前判断:
- 晓萱音色可作为 V2 默认 mei 音色。
- 预训练情感作为稳定底座。
- 自由 instruct 和 fine-grained 标签用于扩展表演力。
- macOS 可做功能和听感验证，实时延迟以 Windows/RTX 4070 为准。
- 后续切到 `onnxruntime-gpu` 后，speech tokenizer 部分有明确优化空间。

### 2.2 不再训练的理由

原计划的 mini-Talker 训练主要是为了解决:
- 本地低延迟发声。
- 情感可控。
- 角色音色稳定。

但当前 CosyVoice3 已经提供了足够可用的音色克隆和情感泛化。短期产品目标更需要:
- 多情感脚本稳定。
- 分段自然。
- 拼接不突兀。
- 生成延迟可接受。
- prompt 可控，不跑偏。

因此训练路线暂时冻结，只保留为远期备选。

---

## 3. V2 架构

```
┌────────────────────────────────────────────────────────────┐
│                    AiriLife Voice V2                       │
├────────────────────────────────────────────────────────────┤
│                                                            │
│  Text Input                                                │
│      │                                                     │
│      ▼                                                     │
│  DeepSeek API                                              │
│  - mei 人设                                                │
│  - 输出 JSON 表演脚本                                      │
│  - 每段包含 emotion/instruct/text/pause                    │
│      │                                                     │
│      ▼                                                     │
│  Script Parser + Validator                                 │
│  - JSON 解析                                               │
│  - 情感白名单                                              │
│  - 文本长度限制                                            │
│  - 自动 fallback                                           │
│      │                                                     │
│      ▼                                                     │
│  Segment Synthesizer                                       │
│  - CosyVoice3 inference_instruct2                          │
│  - 必要时 inference_cross_lingual 处理 fine tags            │
│  - 后续切 stream=True                                      │
│      │                                                     │
│      ▼                                                     │
│  Audio Assembler                                           │
│  - silence pause                                           │
│  - short crossfade                                         │
│  - loudness normalize                                      │
│  - debug wav per segment                                   │
│      │                                                     │
│      ▼                                                     │
│  Playback / Output WAV                                     │
│                                                            │
└────────────────────────────────────────────────────────────┘
```

---

## 4. DeepSeek 输出协议

V2 建议从 `[emotion:intensity] text` 协议升级为 JSON 表演脚本。

原因:
- 一句话多情感时，括号描述、停顿、音量变化、哭腔、笑意等不适合塞进纯文本。
- JSON 更容易校验、回退和记录。
- 后续可以直接把片段映射到 CosyVoice3 API 参数。

### 4.1 输出格式

DeepSeek 必须只输出 JSON，不输出解释文本。

```json
{
  "segments": [
    {
      "emotion": "shy",
      "intensity": 0.55,
      "text": "哥哥……你终于回来了呀，",
      "instruct": "请用轻快、带一点撒娇和尾音上扬的语气说这句话。",
      "fine_text": "哥哥……你终于回来了呀，",
      "pause_after_ms": 180
    },
    {
      "emotion": "sad",
      "intensity": 0.8,
      "text": "我还以为、还以为你今天又要丢下我一个人了呢……",
      "instruct": "请用委屈、声音发颤、接近哭腔但不要大哭的语气说这句话。",
      "fine_text": "[breath]我还以为、还以为你今天又要丢下我一个人了呢……",
      "pause_after_ms": 240
    },
    {
      "emotion": "happy",
      "intensity": 0.75,
      "text": "不过没关系啦！",
      "instruct": "请突然提高一点能量，用强撑出来的甜甜笑意说这句话。",
      "fine_text": "不过<strong>没关系啦</strong>！",
      "pause_after_ms": 120
    },
    {
      "emotion": "playful",
      "intensity": 0.65,
      "text": "只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~",
      "instruct": "请用调皮、甜软、亲近但不过度暧昧的妹妹语气说这句话。",
      "fine_text": "只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~",
      "pause_after_ms": 0
    }
  ]
}
```

### 4.1.1 运行时流式协议

非流式脚本继续使用上面的 JSON object。低延迟运行时建议改用 **JSONL segment stream**，每行一个可播放片段:

```jsonl
{"e":"happy","i":0.8,"t":"哥哥！你终于来找我啦！","p":120}
{"e":"playful","i":0.7,"t":"我都快无聊到数天花板了~","p":160}
```

选择 JSONL，而不是 `[happy]文本[playful]文本` 的原因:
- JSONL 的首个完整 segment 边界清楚，解析器拿到 `\n` 或完整 `}` 即可启动 TTS。
- inline tag 容易过早 flush 出“哥哥！”、“嗯嗯！”这类太短片段，TTS 首段开销不降反升，韵律也差。
- JSON object 也可增量解析第一个 segment，但状态机更复杂；适合作为脚本保存格式，不适合作为最低延迟协议。

运行时 flush 规则:
- 优先在完整 JSONL 行结束时 flush。
- 如果模型没输出换行，允许在 `}` 后立即 flush。
- `t` 少于 6 个汉字时默认继续等下一个短语，除非是缓存短句或明确停顿。
- 单个首段建议 8-18 个汉字；过短增加 TTS 固定开销，过长推迟首音频。

### 4.2 字段定义

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `emotion` | string | 是 | 情感标签，用于统计和 fallback |
| `intensity` | number | 是 | 0-1，暂时只作为 prompt 强度参考 |
| `text` | string | 是 | 干净文本，送入 `inference_instruct2` |
| `instruct` | string | 是 | CosyVoice3 instruct2 的自然语言控制 |
| `fine_text` | string | 否 | 带 `[breath]`、`<strong>`、`[laughter]` 等标签的文本 |
| `pause_after_ms` | integer | 是 | 当前片段后插入静音 |

### 4.3 情感白名单

```text
happy, excited, sad, calm, angry, shy, love, playful, surprise, neutral
```

V2 运行时映射:

| emotion | CosyVoice3 控制 |
|---------|-----------------|
| `happy` | `请非常开心地说一句话。` 或自定义甜快 instruct |
| `excited` | 开心 + 更高能量 + 可选快速语速 |
| `sad` | `请非常伤心地说一句话。` 或委屈/哭腔 instruct |
| `calm` | 平静从容 instruct |
| `angry` | `请非常生气地说一句话。`，但限制为轻微不满/吐槽 |
| `shy` | 害羞/撒娇 instruct |
| `love` | 温柔/依赖 instruct，约束不过度暧昧 |
| `playful` | 调皮可爱 instruct |
| `surprise` | 惊讶 instruct |
| `neutral` | 普通克隆或平静 instruct |

---

## 5. 多情感一句话策略

### 5.1 分段，而不是单句全局控制

用户示例这种句子包含至少四个表演转折:

1. 轻快上扬、撒娇。
2. 委屈、发颤、哭腔。
3. 突然拔高、强撑甜笑。
4. 调皮亲近、收尾撒娇。

CosyVoice3 更适合每个情绪片段单独合成。直接把整句交给一个 instruct，模型很可能只抓住其中一种主情绪，或者情绪变化不稳定。

### 5.2 分段长度

建议:
- 单段 6-24 个汉字。
- 情绪转折必须切段。
- 强停顿、哭腔、笑意、音量变化前后切段。
- 不要切得过碎，少于 4 个字的片段容易不自然。

### 5.3 拼接策略

每段合成后进入 Audio Assembler:

- 片段间默认插入 `pause_after_ms` 静音。
- 相邻片段无停顿时使用 20-60ms crossfade。
- 对每段做轻量 loudness normalize，避免“突然拔高”变成破音。
- 情绪强切换可以保留明显停顿，不强行 crossfade。
- 每轮保存:
  - `segment_01.wav`
  - `segment_02.wav`
  - `combined.wav`
  - `script.json`

### 5.4 fine-grained 标签使用

优先使用:
- `[breath]` 表示吸气、委屈、停顿。
- `<strong>...</strong>` 表示强调。
- `[laughter]` 或 `<laughter>...</laughter>` 表示轻笑。

谨慎使用:
- `[sigh]`、`[quick_breath]`、`[vocalized-noise]` 可以后续逐项验证。
- 不要让 DeepSeek 自由发明 CosyVoice3 不支持的标签。

运行时规则:
- 默认走 `inference_instruct2(text, instruct_text, ref_audio)`。
- 如果 `fine_text` 含 fine-grained 标签，单独测试两种路径:
  - A: `inference_instruct2(fine_text, instruct_text, ref_audio)`
  - B: `inference_cross_lingual(prefix + fine_text, ref_audio)`
- 以听感决定最终模式。短期可以配置开关，不硬编码。

---

## 6. Prompt 设计

### 6.1 System Prompt 要点

DeepSeek 需要同时扮演“角色”和“语音导演”。

必须约束:
- 你是 mei/小萱，活泼可爱的年轻女生，用户是哥哥。
- 回复自然，像即时聊天，不写小说旁白。
- 输出 JSON，不输出 Markdown，不输出解释。
- 把表演描述转成 `instruct`，不要把括号舞台提示直接塞进 `text`。
- `text` 是最终要念出来的话。
- `instruct` 是给 TTS 的语气控制，不会被念出来。
- 每段 `text` 不超过 24 个汉字，太长要切。
- `love`、`shy` 要保持温情和依赖，避免成人化。

### 6.2 Prompt 模板

```text
你是 AiriLife 的桌宠角色 mei，也叫小萱。
用户是你的哥哥。你活泼、可爱、ENFP、亲近但不成人化。

你的任务不是直接聊天文本，而是生成给 TTS 使用的表演脚本。

只输出合法 JSON，格式:
{
  "segments": [
    {
      "emotion": "happy|excited|sad|calm|angry|shy|love|playful|surprise|neutral",
      "intensity": 0.0-1.0,
      "text": "最终要念出来的话，不包含括号舞台提示",
      "instruct": "给 CosyVoice3 的中文语气指令",
      "fine_text": "可选，允许使用 [breath]、<strong>、[laughter]",
      "pause_after_ms": 0-600
    }
  ]
}

规则:
1. 情绪变化、停顿、音量变化、哭腔、笑意都要切成不同 segment。
2. 每个 segment 的 text 尽量 6-24 个汉字。
3. 不要输出 JSON 以外的任何内容。
4. 不要让 text 包含“尾音上扬”“声音发颤”等舞台提示。
5. instruct 可以描述尾音、哭腔、甜笑、轻声、调皮等。
6. 如果用户要求长句，拆成多个 segments。
```

---

## 7. 实现计划

### Phase 1: Script Runner

目标: 文本输入后拿到稳定 JSON，并能保存脚本。

任务:
- 新建 `voice_v2.py` 或拆成 `voice_v2/` 包。
- 实现 DeepSeek 非流式 JSON 调用。
- 实现 JSON 提取和校验。
- 加 fallback:
  - JSON 解析失败时重试一次。
  - 仍失败则用整句 `neutral` 合成。
- 保存 `runs/<timestamp>/script.json`。

验收:
- 10 条复杂文本输入，至少 9 条输出合法 JSON。
- 所有 `text` 不含括号舞台提示。
- 情绪转折能被拆成多个 segments。

### Phase 2: Segment TTS

目标: 每段独立生成 wav。

任务:
- 使用晓萱参考音频作为默认 `REF_AUDIO`。
- 每段调用 `inference_instruct2`。
- 支持 `fine_text` 开关。
- 每段保存独立 wav。
- 记录每段:
  - 文本长度
  - emotion
  - audio 秒数
  - synth 秒数
  - RTF

验收:
- 用户示例能生成 4 段以上音频。
- 每段情绪听感有区别。
- 没有明显静音垃圾音频。

### Phase 3: Audio Assembler

目标: 合成 `combined.wav`，听起来像一句完整表演。

任务:
- 根据 `pause_after_ms` 插入静音。
- 实现 20-60ms crossfade。
- 实现轻量 peak normalize。
- 输出 `combined.wav`。
- 输出 `manifest.json`，记录拼接参数。

验收:
- 情绪切换明显。
- 拼接处没有明显爆音/click。
- 停顿符合脚本。

### Phase 4: Latency Mode

目标: 为运行时体验做准备，但仍不做 ASR。

任务:
- DeepSeek 改为 SSE + JSONL segment stream。
- Parser 在拿到第一行完整 JSONL 后立即启动 TTS，同时继续接收后续 segment。
- TTS 默认仍按 segment 合成；`stream=True` 只在实测首包收益稳定时开启。
- 播放器边收 segment 音频边播放，队列最多保留 2-3 个 segments。
- 模型进程必须常驻预热，冷启动只允许发生在服务启动阶段。

验收:
- 首段开始合成时间早于 DeepSeek 完整输出结束时间。
- 首个可播放音频延迟有日志。
- 即使后续 segment 还在生成，也能先播放第一段。
- 记录 `llm_first_token_ms`、`first_segment_ms`、`tts_first_audio_ms`、`first_playable_ms`。

### Phase 5: Runtime Optimization

目标: 降低延迟，提高稳定性。

任务:
- Windows/RTX 4070 环境安装并验证 `onnxruntime-gpu`，确认 `CUDAExecutionProvider` 生效。
- 模型启动预热，并保持一个常驻 TTS worker，避免每轮 20-50s 加载。
- 固定参考音频和 prompt text，减少运行时变量。
- 缓存常用短句，例如“哥哥”“嗯嗯”“嘿嘿”等。
- 对首段使用 8-18 字短 segment，避免 `哥哥！` 这种过短片段触发一次完整 TTS。
- 评估 `load_trt` / TensorRT 只针对 flow decoder；CosyVoice3 代码对 DiT TensorRT fp16 有性能警告，必须实测后再启用。
- 评估 `load_vllm` 作为二线优化；它可能降低 TTS 内部 LLM 延迟，但会增加依赖和显存压力。

验收:
- RTX 4070 上 RTF 明显低于当前 PyTorch + CPU onnxruntime 路径。
- 目标首段可播放 < 4s；理想目标 < 2.5s。
- 复杂 4 段示例可以在可接受时间内完成，并保持 `single_clean.wav` 的音色/情感方向。

### Phase 5.1 不优先做的优化

- GGUF: 不适合当前 CosyVoice3。模型不是单一 LLM 权重，包含 PyTorch/ONNX/TTS flow/vocoder，转 GGUF 不能覆盖完整推理链路。
- 整体 INT8/INT4 量化: 有音质和情感退化风险，且当前瓶颈先指向 CPU ONNX provider。先做 ONNX GPU / TRT / 常驻预热，再评估局部量化。
- token-level DeepSeek -> TTS: 不做。中文 TTS 需要短语级上下文，token 级会破坏韵律，也会导致大量过短 TTS 调用。

---

## 8. 目录建议

```text
Airilife_voice/
  voice_v2.py
  voice_v2/
    __init__.py
    deepseek_client.py
    script_schema.py
    cosyvoice_synth.py
    audio_assembler.py
    prompts.py
  runs/
    20260617_153000/
      input.txt
      script.json
      segment_01.wav
      segment_02.wav
      combined.wav
      manifest.json
```

短期可以先写单文件 `voice_v2.py`，跑通后再拆模块。

### 8.1 脚本命名规范

当前脚本处于 POC 阶段，暂不重命名历史文件，避免测试记录和 README 失效。后续新增/整理时按以下规则:
- 主链路: `voice_v2.py`，后续拆到 `voice_v2/` 包。
- 基准测试: `benchmark_latency.py`，只放可重复测量的 DeepSeek/TTS 延迟逻辑。
- 一次性听感实验: `single_call_emotion_test.py`、`pitch_test_win.py` 这类保留 `_test` 后缀。
- 平台差异脚本才使用 `_win` / `_mac`，例如 `make_refs_win.py`、`make_refs_mac.py`。
- 避免新增 `run_test.py`、`test2.py`、`demo_new.py` 这类不可读名称。

---

## 9. 验收样例

输入:

```text
哥哥……你终于回来了呀，（尾音轻快上扬）我还以为、还以为你今天又要丢下我一个人了呢……（声音骤然发颤带上了哭腔）不过没关系啦！（突然拔高音量挤出甜腻的笑意）只要哥哥现在愿意摸摸我的头，我就还是你最乖的妹妹哦~
```

期望 DeepSeek 输出:
- 至少 4 个 segments。
- 第一段 `shy` 或 `happy`，轻快上扬。
- 第二段 `sad`，委屈/哭腔。
- 第三段 `happy` 或 `excited`，强撑甜笑。
- 第四段 `playful` 或 `love`，甜软收尾。
- 舞台提示不进入 `text`。
- 停顿在 120-300ms 之间，哭腔前后可稍长。

期望音频:
- 听得出情绪变化。
- 拼接处自然。
- 角色音色保持晓萱。
- 没有训练依赖。

---

## 10. 风险与对策

| 风险 | 表现 | 对策 |
|------|------|------|
| 分段后音色轻微漂移 | 每段像不同状态的人 | 固定同一参考音频；减少过强 instruct；必要时统一 normalize |
| 拼接突兀 | 段落之间断裂 | crossfade + pause 调参；让 DeepSeek 少切超短段 |
| 自由 instruct 不稳定 | 害羞/调皮不明显 | 为每个 emotion 固定几套经过试听的 instruct 模板 |
| JSON 偶发非法 | DeepSeek 输出说明文字 | JSON mode prompt + 解析失败自动重试 |
| 延迟偏高 | 多段串行合成慢 | Phase 4 改 SSE + 并行/队列；Phase 5 上 onnxruntime-gpu |
| fine tag 破坏发音 | 标签被念出或生成异常 | fine_text 做白名单；失败回退到 clean text |

---

## 11. 当前优先级

最高优先级:
1. `voice_v2.py` 跑通非流式 JSON 脚本 + CosyVoice3 分段合成。
2. 用户示例生成 `combined.wav`。
3. 试听后固化晓萱的 emotion -> instruct 模板。

暂缓:
- ASR。
- mini-Talker 训练。
- 数据集扩增。
- Live2D 口型同步。
- barge-in。

这版 V2 的成功标准不是“端到端实时语音助手”，而是“输入文本后，mei 能用同一个音色说出有多段情绪变化的一句话，并且听感足够接近角色设定”。
