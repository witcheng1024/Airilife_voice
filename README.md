# AiriLife Voice — 角色语音服务

AiriLife 的本机语音服务与角色声音资产。AiriLife 通过 HTTP 调用这里的服务，把角色的回复读出来（桌宠流式播放、对话框朗读、QQ 语音）。

当前默认引擎是 **Qwen3-TTS**（2026-10-21 起），GPT-SoVITS 保留为备用。两者 HTTP 协议相同，同一时间只在 9880 端口跑其中一个。

| 角色 | 服务内音色 ID | Qwen3-TTS 声线 | GPT-SoVITS 声线 |
| --- | --- | --- | --- |
| 梓涵（Zelia） | `zihan` | **v2 原创声线**（VoiceDesign 设计的定稿母版，Base 克隆） | 中文微调模型 |
| Airi | `firefly` | 流萤（用 GSV 参考音频临时克隆，待设计 Airi 专属声线） | 流萤微调模型 |

> 音色 ID 是 AiriLife 角色包里 `voice.speaker` 绑定的值，改名需要同步 AiriLife 侧。界面上显示的是角色名，不显示 ID 与版本号。

## 目录

```
models/
  airi/          Airi
    gsv/           GPT-SoVITS 权重（LFS）
    references/    8 种情绪参考音频（GSV 推理参考，也是 Qwen 克隆参考）
  zelia/         梓涵
    qwen_v2/       ★ Qwen 定稿母版：7 个场景 wav + manifest.json + index.html 试听页
    references/    GSV 情绪参考音频
    voice.json     GSV 音色清单（模型路径、情绪参考）
    gsv/           GSV 微调权重（不入库）
    qwen_v1/       Qwen v1 设计稿（已被 v2 取代，不入库）
  base/          共用底座（不入库）
    qwen3-tts/     Qwen3-TTS 0.6B / 1.7B Base、1.7B VoiceDesign
    gsv/           GSV-TTS-Lite 预训练模型（首次运行自动下载）
voices/
  qwen3_voices.json         ★ Qwen 服务音色清单：每个情绪 = 一条母版 wav + 逐字台词
  qwen3_voice_prompts.json  VoiceDesign 声线描述与场景台词（设计新声线用）
tools/
  serve/          常驻服务：qwen_tts_server.py、tts_server.py（GSV）、voice_registry.py
  qwen/           Qwen 工具：声线设计、克隆一致性检查、基准、模型下载
  gsv_training/   GPT-SoVITS 训练与评估脚本（WSL）
docs/             qwen3_tts_local.md（Qwen 部署细节）、gsv_v2_history.md（GSV 训练历程）等
archive/          不再使用但保留：未选用的训练轮次、训练语料（不入库）
```

整理约定：每个角色一个目录，放这个角色专属的权重、参考音频和定稿母版；多个角色共用的模型放 `models/base/`；试听、基准等临时产物写到 `output/`（不入库，可随时删）；不再使用但还想留着的放 `archive/`。

## 启动服务

正常由 AiriLife 托管启动：设置 → 外部服务 → 语音服务。工作目录填本仓库根目录，并勾选随应用启动。

| 引擎 | Python 环境 | 脚本 |
| --- | --- | --- |
| Qwen3-TTS（默认） | `D:\miniforge3\envs\qwen3-tts-fast\python.exe` | `tools\serve\qwen_tts_server.py` |
| GPT-SoVITS | `D:\miniforge3\envs\gsv\python.exe` | `tools\serve\tts_server.py` |

手动调试：

```powershell
D:\miniforge3\envs\qwen3-tts-fast\python.exe tools\serve\qwen_tts_server.py --host 127.0.0.1 --port 9880
D:\miniforge3\envs\qwen3-tts-fast\python.exe tools\serve\qwen_tts_server.py --bench   # 不起服务，测首包与实时倍率
curl.exe -s http://127.0.0.1:9880/health
```

两套环境的安装方法与踩坑记录见 `docs/qwen3_tts_local.md`。

## HTTP 协议（两套服务相同）

- `POST /api/tts`：请求体 `{"text": "...", "emotion": "温柔", "speaker": "zihan", "stream_mode": "token"}`，返回裸 int16 PCM 流。`Content-Type: audio/L16; rate=<采样率>; channels=1`，Qwen 是 24000，GSV 是 32000。AiriLife 按这个头播放。
- `GET /health`：`{ok, engine: "qwen3-tts" | "gsv-tts-lite", status: ready | loading | unavailable, voices, ...}`
- `GET /api/voices`：音色和各自支持的情绪。AiriLife 据此做情绪适配，服务不支持的情绪会回落到「平淡」。

## Qwen3-TTS：延迟与参数

RTX 4070 Laptop 8 GB 上，0.6B Base + ICL，chunk 4：

| 测量层 | 首包 | 实时倍率 |
| --- | --- | --- |
| 服务内（`--bench`） | 梓涵约 305 ms，流萤约 240 ms | 1.2–2.0× |
| AiriLife 后端收到首块 | 275–340 ms | 1.2–1.9× |

- `--chunk-size`：越小首包越快、吞吐越低。4 是默认值；8 的首包在 390–480 ms；1 跟不上实时。
- 服务启动时会把每条参考音频都预热一遍，包括参考特征缓存和 CUDA Graph，所以首个真实请求不会更慢。加载加预热大约 25–100 秒。
- 显存：0.6B 进程约占 3.4 GB。8 GB 卡上不要和 GSV 同时常驻。
- 不在请求路径里调用 VoiceDesign，声线以定稿母版为准。同一段描述每次重新设计出来的声音都会漂移。

## 常见维护

- **给某个情绪换参考 / 新增情绪**：编辑 `voices/qwen3_voices.json`。情绪键要用 AiriLife 发的标签（平淡、活泼、温柔、撒娇甜、害羞、悲伤、惊讶、生气、紧张），`text` 必须和音频内容逐字一致，改完重启服务。
- **为 Airi 设计专属声线**：
  1. 在 `voices/qwen3_voice_prompts.json` 里写声线描述和场景台词；
  2. 用 `tools/qwen/qwen3_voice_design.py` 出片，在 `index.html` 里试听挑选；
  3. 把选定的一组移到 `models/airi/qwen_v1/`；
  4. 在 `qwen3_voices.json` 里把 `firefly` 的参考换掉，或者新增一个音色并在 AiriLife 角色包里绑定。
- **音色一致性检查**：`tools/qwen/qwen3_base_clone_check.py`，用 speaker encoder 的余弦相似度量化克隆漂移。
- **重训 GSV**：训练脚本在 `tools/gsv_training/`，语料已移到 `archive/data/`，重训前先移回 `data/` 或改脚本路径。历程见 `docs/gsv_v2_history.md`。
