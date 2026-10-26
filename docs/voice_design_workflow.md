# 角色声线工作流：VoiceDesign 设计 → 情绪参考 → Base 克隆

梓涵（`zihan`）的 Qwen3-TTS 声线已于 2026-10-22 上线，这份文档把整个过程整理成可以复用的流程，下一个角色 Airi 照着做。
环境安装、模型下载和踩坑记录见 [`qwen3_tts_local.md`](qwen3_tts_local.md)。

## 一、核心思路

```
一段固定的英文声线描述（instruct）
   │  1.7B VoiceDesign，每个情绪场景一句台词，每句出多个 take
   ▼
候选片（output/qwen3_voice_design/<voice>/<run>/）
   │  人耳试听挑选，每种 AiriLife 情绪选 1 条
   ▼
情绪参考母版（models/<角色>/qwen_vN/emotions/*.wav + emotions.json）
   │  0.6B Base，ICL（参考音频 + 逐字台词），不加 instruct
   ▼
常驻服务（voices/qwen3_voices.json → tools/serve/qwen_tts_server.py）
```

## 目录约定：草稿在 output，定稿在 models

| 阶段 | 位置 | 入库 | 说明 |
| --- | --- | --- | --- |
| 草稿：候选片 | `output/qwen3_voice_design/<voice>/<日期_轮次>/` | 否 | 每轮一个 `index.html`，可随时删 |
| 草稿：克隆验收 | `output/emotion_clone_check/<run>/` | 否 | 同上 |
| **定稿** | `models/<角色>/qwen_vN/emotions/` | 是 | `<canonical>.wav` + `emotions.json` + `index.html`，**线上只认这里** |
| 服务清单 | `voices/qwen3_voices.json` | 是 | 指向定稿目录 |

`<voice>` 和 `qwen_vN` 的版本号保持一致：`airi_v1` 对应 `models/airi/qwen_v1/`，`zihan_v2` 对应 `models/zelia/qwen_v2/`。
从草稿到定稿**只走 `tools/qwen/promote_emotions.py`**，不要手动复制。这个脚本会从出片 manifest 里取逐字台词，记下来源，并重建定稿试听页。想听现在线上用的声音，打开 `models/<角色>/qwen_vN/emotions/index.html` 就行。

几条要点：

1. **instruct 定稿后不再改动，情绪全靠台词带出来。** 所有情绪场景用同一段描述，只换台词。`hint` 字段只在试听页上显示，不会送进模型。所以台词本身要写得情绪很足，比如生气就要全程在生气，中间不能转折。
2. **梓涵最终是每个情绪单独一条参考，每条都是 VoiceDesign 设计出来的原片，不是从锚点克隆的。** 走过的路：先试过所有情绪共用 daily_t1 一个锚点（64da813，40 分钟后由 e6b36ba 回退，提交里没有记录回退原因）；又试过 bedtime 单参考加情绪指令（instruct_emotion_grid，没采用）；最后才回到“描述不变、只换台词”的逐情绪设计。
   **风险**：同一段描述每次随机出来的音色并不一致。梓涵靠每句出 3 个版本、人耳挑出和母版同一个人的那条来控制。Airi 第一轮的 daily_t1 和 explain 就是两种音色，所以先定一条权威锚点，再让各情绪向它对齐（见第三节）。
3. **克隆时不加 instruct。** 试过单参考配情绪指令（`instruct_emotion_grid.py`，两轮对照），没有采用。Base 模型没有专门训练过听指令，效果不稳定。
4. **只靠人耳挑。** 说话人余弦相似度在负对照里分不出人（流萤对梓涵锚点也有 0.97 以上），只能拿来发现明显的漂移，不能拿来选参考。
5. **请求路径里不调用 VoiceDesign。** 同一段描述每次重新生成都会漂移，线上声线以定稿母版为准。

## 二、步骤

### 1. 写声线描述和场景台词

在 [`voices/qwen3_voice_prompts.json`](../voices/qwen3_voice_prompts.json) 新增一个 key（例如 `airi_v1`）：

- `instruct`：英文声线描述，按 `zihan_v2` 的写法：以 `A natural Mandarin Chinese voice of …` 开头，写音高、音色质感、性格和说话节奏，最后用 avoiding … 列出不想要的方向（梓涵排除了“幼稚”和“成熟/戏剧化”）。**基调必须是放松、可收的**：VoiceDesign 每句都会照描述的能量去演，描述里反复写 bright / cheerful / smiling 会让难过、困倦、温柔全部带笑（Airi v1 的教训）。活泼的角色写“性格活泼，但基调是放松的对话，不是每句都开心”。
- `scenes`：AiriLife 的十种情绪各至少一个场景，容易出问题的情绪（生气、普通）多准备一两句备选。每句 30～45 字，口语化，情绪贯穿整句。

AiriLife 的十种情绪标签（顺序与输出协议一致）：

| canonical | 标签 | 梓涵最终用的场景 |
| --- | --- | --- |
| neutral | 普通 | happy（书店，平稳叙述） |
| happy | 高兴 | daily（你回来啦） |
| sad | 难过 | sad |
| angry | 生气 | angry（可爱嘟嘴版，第三轮才定） |
| surprised | 惊讶 | surprised |
| shy | 害羞 | shy2 |
| tired | 困倦 | sleepy |
| gentle | 温柔 | bedtime |
| nervous | 紧张 | nervous |
| confused | 困惑 | confused |

> 流程规则：所有角色的场景台词和情绪提示完全相同，只有声线描述（`instruct`）按角色单独写。注：梓涵的普通 / 高兴是试听后对调入库的个例，不属于流程；Airi 按场景正常对应（普通 = daily，高兴 = happy）。

### 2. 出候选片

```powershell
conda activate qwen3-tts
python tools/qwen/qwen3_voice_design.py --voice airi_v1 --take 3 --run-name 20261026_round1
# 只重做某几个情绪
python tools/qwen/qwen3_voice_design.py --voice airi_v1 --scenes angry angry2 --take 3 --run-name 20261026_angry
```

打开 `output/qwen3_voice_design/airi_v1/<run>/index.html` 试听。

### 3. 先定声线，再定情绪

- **第一轮先只看是不是这个人。** 先出 daily / explain / gentle 这几个温和场景。如果整体声线不对（太尖、太幼、太成熟），回去改 instruct 再出，**这一步不要急着挑情绪片**。
- instruct 一旦定稿就冻结。之后哪个情绪不满意，只改那句台词或者多出几个 take，不再动 instruct，否则前面已经选好的片会和新片对不上。
- 梓涵的经验：生气最难。直接写“发火”会出戏，改成“气鼓鼓的可爱抱怨、全程不转折”，又重做了两轮才定。

### 4. 定稿入库

```powershell
python tools/qwen/promote_emotions.py --dest models/airi/qwen_v1 --label Airi ^
    neutral=output/qwen3_voice_design/airi_v1/20261026_round1/daily_t2.wav ^
    gentle=output/qwen3_voice_design/airi_v1/20261026_round1/gentle_t1.wav
```

可以分多次执行：每次只覆盖本次提到的情绪，其余保留。试听页会标出还缺哪些情绪。canonical 取值为 neutral / happy / sad / angry / surprised / shy / tired / gentle / nervous / confused。

### 5. 克隆验收

```powershell
conda activate qwen3-tts-fast   # 先停掉常驻服务腾出显存
python tools/qwen/emotion_clone_check.py --emotion-dir models/airi/qwen_v1/emotions   # 每种情绪用自己的参考克隆新台词（默认梓涵）
python tools/qwen/reference_listening_grid.py --models 0.6b   # 需要时：参考片 × 模型对照
```

听两件事：新台词有没有保留住情绪；不同情绪之间切换时，听起来还是不是同一个人。哪条不过就回到第 3 步，只换那一条。

### 6. 上线

1. `python tools/qwen/promote_emotions.py --dest models/airi/qwen_v1 --register airi --label Airi`：把定稿按协议顺序写进 `voices/qwen3_voices.json`（会整块替换该音色的 `references`）。
2. 重启服务，检查 `GET /api/voices`。未知情绪会回落到「普通」或「平淡」。
3. AiriLife 角色包的 `voice.speaker` 绑定这个音色 ID。
4. 提交：母版 wav、`emotions.json`、`qwen3_voices.json`。`output/` 不入库。

## 三、Airi v1 声线描述

2026-10-26 用户给定，原文写进了 `voices/qwen3_voice_prompts.json` 的 `airi_v1.instruct`，没有改动：

> A bright, lively teenage girl's voice, around 15–16 years old, with a naturally youthful mid-to-high pitch and a clear, light, pleasantly resonant timbre.
>
> Her voice is fresh, warm, and slightly sweet, with a gentle sparkle and a natural smiling quality. The vocal texture is smooth and agile, retaining enough softness and warmth to avoid sounding sharp, thin, or shrill.
>
> She has a cheerful, affectionate, and slightly mischievous personality. Her speech feels spontaneous and conversational, with lively but natural pitch variations, expressive intonation, and a lightly bouncy rhythm.
>
> She can shift naturally between playful excitement, gentle teasing, sincere warmth, and softer, intimate moments without losing her recognizable vocal identity.
>
> The overall impression is youthful, charming, energetic, and emotionally vivid, avoiding exaggerated anime cuteness, childish squeaking, artificial sweetness, excessive nasality, or theatrical overacting.

和梓涵的对比：梓涵沉静内敛、能量收着；Airi 明亮外向、节奏有弹性。所以 Airi 的「普通」本身就带笑意，「高兴」和「普通」的差距要靠台词拉开。

### 场景与执行

**场景台词与 `zihan_v2` 完全相同**（16 个场景，原样复制），只换声线描述。新角色默认照搬这套已验证的台词，不要自己另写。
2026-10-26 的教训：第一次给 Airi 新写了一套台词，又同时试了“锚点克隆”，结果变量太多，难过、困倦、温柔全部失败，也判断不出问题出在台词还是描述上。

1. `python tools/qwen/qwen3_voice_design.py --voice airi_v1 --take 3`：一次出全部 16 个场景，一个试听页。
2. 按梓涵的方式挑：每个 AiriLife 情绪挑一条，要求“是同一个人、情绪到位”，可以跨场景归类。
3. 没有满意的情绪：同一句台词再出几版。如果多轮都偏离（例如难过总带笑），再考虑改描述，改描述另起 `airi_v2`。
   实际结果：v1 用梓涵台词后，难过、困倦、温柔（bedtime）、害羞（shy2）仍然不过，原因是描述的基调太亮。`airi_v2` 按梓涵描述的写法重写（Mandarin 开头、基调放松），只补这四个场景，只有 sleepy_t1 通过。

**Airi 最终定稿（2026-10-26）：9 种情绪，没有「难过」。** 普通 daily_t1、高兴 happy_t2、生气 angry_t1、惊讶 surprised_t3、害羞 shy_t1、温柔 comfort_t1、紧张 nervous_t3、困惑 confused_t2 来自 v1；困倦 sleepy_t1 来自 v2。`models/airi/qwen_v1/` 根目录的 tease_t1、explain_t2 是备选，不在十情绪里。
用户决定不再补「难过」：AiriLife 的提示词只列出当前音色有的情绪，前端只显示音色有的情绪，遇到音色没有的情绪按「普通」合成。配方在 AiriLife 的 `voice/recipes/airi/qwen3_tts.json`（M69-03）。
4. 定稿用 `promote_emotions.py` 放进 `models/airi/qwen_v1/`，克隆验收后替换 `firefly` 的临时参考，或新增 `airi` 音色。
