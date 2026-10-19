# Qwen3-TTS 本地部署与音色设计（`qwen3-tts` 分支）

本分支记录 AiriLife 的第二条语音路线：**用 Qwen3-TTS 做原创音色设计 + 参考音频复现**，
与 `v2` 分支的 GPT-SoVITS 方案并存，不替换、不干扰现有 9880 服务。

> 边界：训练在 WSL2（见 `v2`）。**本分支全部在 Windows 原生跑**，只做推理/设计，不涉及训练。

## 一、为什么是这套

| 环节 | 工具 | 角色 |
|---|---|---|
| 设计声音 | `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 「创作器」：一段自然语言描述 → 一条原创声线 |
| 复现声音 | `Qwen3-TTS-12Hz-1.7B-Base` / `0.6B-Base` | 「复现器」：参考 wav + 台词 → 日常合成都用这条声线 |
| 加速推理 | [`faster-qwen3-tts`](https://github.com/andimarafioti/faster-qwen3-tts)（CUDA Graph / 流式） | 让 Base 进入实时可用区间 |

官方 `qwen-tts` 只做离线整句生成，没有真流式；`faster-qwen3-tts` 补上了 CUDA Graph 和
`generate_voice_clone_streaming(chunk_size=...)`，这是 Qwen3-TTS 有可能进 AiriLife 的前提。

## 二、环境（Windows，两套 conda 环境，互不污染）

```powershell
# ① 官方包：做音色设计（离线整句，质量优先）
conda create -n qwen3-tts python=3.12 -y
conda activate qwen3-tts
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install qwen-tts
pip install -U modelscope

# ② 加速包：做实时/流式基准（transformers 必须 5.x）
conda create -n qwen3-tts-fast python=3.12 -y
conda activate qwen3-tts-fast
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install faster-qwen3-tts
pip install "transformers==5.15.1"     # ⚠️ 见下方踩坑
```

**为什么要两个环境**：`faster-qwen3-tts` 依赖 `qwen-tts-hf` + `transformers>=5.15`，
官方 `qwen-tts` 钉死 `transformers==4.57.3`，同一个环境装不了。

### 踩坑记录

1. **`transformers` 版本**：`faster-qwen3-tts` 装完会带 `transformers 5.20.x`，加载 Base 模型时报
   `AttributeError: 'MimiConfig' object has no attribute 'rope_theta'`。降到 README 验证过的
   `5.15.1` 才能跑。README 里 AMD 那段的提示同样适用于 Windows。
2. **`from_pretrained(dtype='bf16')` 会炸**：`faster_qwen3_tts/model.py` 走 `getattr(torch, dtype)`，
   传字符串 `bf16` 报 `torch has no attribute 'bf16'`。必须传 `torch.bfloat16` 对象。
3. **`SoX could not be found`**：`sox` 包在 import 时会探一次系统 `sox` 可执行文件，找不到只打警告，
   不影响生成。要消掉警告就单独装 SoX。
4. **CUDA Graph 捕获**：warmup 阶段会打印 `Capturing CUDA graph ...`。**首次几轮是捕获，不能算进指标**，
   基准脚本为此先 warmup 再丢弃第一轮。

## 三、模型下载（ModelScope，国内快）

```powershell
conda activate qwen3-tts
python tools/fetch_qwen3_tts_models.py --list          # 先看体积
python tools/fetch_qwen3_tts_models.py                 # 三个都下
python tools/fetch_qwen3_tts_models.py --only voice-design
```

| 模型 | 体积 | 落到 |
|---|---|---|
| `Qwen3-TTS-12Hz-1.7B-VoiceDesign` | 4.21 GB | `models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-VoiceDesign/` |
| `Qwen3-TTS-12Hz-1.7B-Base` | 4.23 GB | `models/qwen3-tts/Qwen3-TTS-12Hz-1.7B-Base/` |
| `Qwen3-TTS-12Hz-0.6B-Base` | 2.34 GB | `models/qwen3-tts/Qwen3-TTS-12Hz-0.6B-Base/` |

合计 10.8 GB，已加入 `.gitignore`（`models/qwen3-tts/`），权重不进 git。
每个仓库自带 `speech_tokenizer/`（651MB），三个都存一份换来「单目录直接 from_pretrained」。

## 四、音色设计工作流

提示词和场景台词都在 [`voices/qwen3_voice_prompts.json`](../voices/qwen3_voice_prompts.json)，
**改这个文件即可，不用改代码**。当前有两条梓涵候选：

| key | 定位 |
|---|---|
| `zihan` | v1：带一点点极细颗粒的丝绒质感（subtle fine-grained huskiness） |
| `zihan_v2` | v2：去掉微哑，改成轻盈通透（subtle airy quality），明确排除「幼稚」和「成熟/戏剧化」 |

```powershell
conda activate qwen3-tts
# 全部场景出片
python tools/qwen3_voice_design.py --voice zihan_v2
# 只出几句话 / 出 3 个随机采样挑最好的
python tools/qwen3_voice_design.py --voice zihan --scenes daily bedtime --take 3
# 自由台词
python tools/qwen3_voice_design.py --voice zihan --text "嗯？你回来啦。" --tag free
```

产物：`output/qwen3_voice_design/<voice>/<日期_时间>/`
`*.wav` + `manifest.json`（每条 prompt/耗时/显存）+ `index.html`（浏览器直接听完对比）。

选定的 wav 就是**角色声音母版**，下一步交给 Base 复现。

## 五、复现与音色一致性

```powershell
conda activate qwen3-tts-fast
python tools/qwen3_base_clone_check.py ^
  --ref output\qwen3_voice_design\zihan_v2\<run>\daily_t1.wav ^
  --ref-text "嗯？你回来啦。今天好像有点晚呢……我刚刚还在想，你是不是又忙着写代码，忘记时间了。" ^
  --voice zihan_v2 --model 0.6b --reps 3 --modes icl xvec
```

- `icl`：参考音频 + 对应台词，最贴原声（`--ref-text` 必须与参考音频逐字一致）。
- `xvec`：只用说话人嵌入，对参考台词不敏感，适合台词拿不准时。
- 指标 `spk_cos_to_ref` 用模型自带 speaker encoder 算余弦；同一场景多次生成的**自相似度**
  就是音色漂移的量化值，比纯听感更早发现问题。

**实测（0.6B Base，梓涵 v1 参考）**：

| 组合 | 自相似均值 | 与参考相似均值 |
|---|---|---|
| icl/daily | 0.9963 | 0.9937 |
| icl/bedtime | 0.9952 | 0.9916 |
| xvec/daily | 0.9949 | 0.9949 |
| xvec/bedtime | 0.9967 | 0.9946 |

结论：**0.6B Base 的音色一致性够用**，ICL 和 xvec 差距在噪声范围内；xvec 更省（不需要 ref_text）。

## 六、性能实测（RTX 4070 Laptop 8GB，bf16，`faster-qwen3-tts` 0.5.4）

```powershell
conda activate qwen3-tts-fast
python tools/bench_faster_qwen3.py --model 0.6b-base --chunks 8 --reps 3
python tools/bench_faster_qwen3.py --model 1.7b-base --chunks 8 4 1 --reps 3
```

| 模型 | chunk | 文本 | TTFA | 音频/耗时 | 倍速 | 整卡显存 |
|---|---|---|---|---|---|---|
| 0.6B | 8 | 7 字 | **465 ms** | 1.36s / 1.03s | 1.32× | 3.37 GB |
| 0.6B | 8 | 34 字 | **439 ms** | 7.28s / 3.69s | 1.98× | 3.47 GB |
| 1.7B | 8 | 7 字 | 512 ms | 1.36s / 1.13s | 1.20× | 5.16 GB |
| 1.7B | 4 | 34 字 | 352 ms | 6.80s / 4.80s | 1.42× | 5.35 GB |
| 1.7B | 1 | 34 字 | **234 ms** | 6.32s / 8.04s | 0.79× | 5.35 GB |

读法：**`chunk_size` 越小首包越快、吞吐越差**。chunk=1 时 1.7B 已经跟不上实时（0.79×），
chunk=4 是首包/吞吐的平衡点，chunk=8 吞吐最好。

显存口径：**整卡 used** 含桌宠/浏览器等常驻约 1.1GB；本进程 torch reserved 在 1.7B 时约 4.35GB。
8GB 卡上 1.7B 与 GPT-SoVITS 常驻服务**不建议同时常开**。

## 七、与 GPT-SoVITS v2 的关系

| 维度 | GPT-SoVITS v2（`v2` 分支） | Qwen3-TTS（本分支） |
|---|---|---|
| 音色来源 | 678 条原声微调，专属模型 | 自然语言设计 + 参考音频克隆 |
| 情绪 | 8 条参考音频切换 | 自然语言 instruct / 风格描述 |
| 双角色切换 | 换 GPT+SoVITS 权重（秒级） | 同一 Base 模型换参考，**理论上更省** |
| 流式 | token 级流式，首块 ~450ms | `faster-qwen3-tts` 流式，首包 234–465ms |
| 显存 | 实测常驻 1.5–1.9GB | 1.7B 约 4.4GB / 0.6B 约 2.2GB |

**当前结论**：GPT-SoVITS 仍是默认引擎。Qwen3-TTS 的价值在「原创声线 + 双角色低切换成本」，
要进 AiriLife 还需要先改音频格式契约（见第八节）。

## 八、接入 AiriLife 需要改的地方（已调研，未改动）

AiriLife 现有 TTS 链路：`chatSessionCoordinator` → `AssistantSpeechStream.pump()`
→ `TtsService.synthesizeStream()` → `/api/tts`（裸 PCM）→ `gatewayHub.pushAudioChunk()`
→ 桌宠 `LipSyncPlayer`。

三个必须先解决的问题：

1. **采样率写死 32000**。`packages/shared/src/tts.ts` 的 `TTS_PCM_FORMAT` 是 32k，
   `packages/pet-window-core/src/api/proactiveEvents.ts:200` 用 `isTtsPcmFormat` 严格校验，
   Qwen3-TTS 输出 **24000Hz** 会被**静默丢帧、无日志**。必须先改成按 chunk 携带真实采样率。
2. **流式白名单**。`ttsService.ts:621` 现在是 `if (provider !== 'gptsovits') yield await this.synthesize()`，
   新 provider 漏改这里会**静默退化成整句合成**，测试也不会报错。
3. **配置白名单**。`ttsService.ts` 的 `agentVoicesSchema.strict()` 与 `agentManager.parseAgentVoice()`
   两处都是硬编码字段名，新音色字段漏加会被丢弃或直接抛错。

推荐做法：把已有的 `GptSoVitsJob` 泛化成 provider-neutral 的 `TtsJob`，
引入 `TtsProviderAdapter`（`synthesize / synthesizeStream / getStatus / isAvailable`），
让 `synthesize()` / `synthesizeStream()` 各自只剩一次查表。
