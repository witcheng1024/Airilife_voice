# AiriLife 流萤语音 v2 — GPT-SoVITS v2ProPlus + GSV-TTS-Lite

Live2D 桌宠「流萤」的语音方案 v2。目标是 **15 岁年龄感 + 多情绪表达**的本地 TTS，满足桌宠实时性要求。

> 本分支（`v2`）是 GPT-SoVITS 方案。旧 CosyVoice3 方案在 `master` 分支，已弃用。
>
> 训练在 WSL2，**推理/部署在 Windows**（GPU：RTX 4070 Laptop）。

---

## 一、技术选型

| 环节 | 方案 | 说明 |
|---|---|---|
| 训练 | GPT-SoVITS **v2ProPlus** | 678 条原声训练，s1=e15 / s2=e10 |
| 推理引擎 | **GSV-TTS-Lite 0.4.7** | 纯 PyTorch GPU 推理（CUDA Graph / Nested KV Cache / fp16），**质量 = 官方 torch** |
| ~~genie-tts~~ | ~~ONNX 引擎~~ | ~~已弃用：ONNX 有音色退化 + 尾字截断（genie issue #31）~~ |
| 声线方案 | 原声数据训练 + 参考音频带情绪/年龄感 | EQ/asetrate 变调无效，年轻感靠推理时换参考 |

**为什么弃 genie-tts 换 GSV-TTS-Lite**：genie 走 ONNX 转换，v2ProPlus 中文在断句/语调/音色上有系统性退化（尾字截断、音色不到位），且是上游 open issue。GSV-TTS-Lite 不转 ONNX、不改权重，只优化执行方式（CUDA Graph/KV Cache/fp16/batching），**质量与官方 PyTorch 一致**，速度 3~4x、显存减半。

---

## 二、数据

- **训练数据**：`data/firefly_678_orig` — 678 条**原声**（无 asetrate/EQ 处理）。
- **情绪参考**（推理时用，本仓库 `reference_audio/`）：8 情绪原声样本

| 情绪 | 参考样本 | 参考文本 |
|---|---|---|
| 平淡 | `chapter3_2_firefly_223` | 我叫流萤，是鸢尾花家系的艺者。 |
| 活泼 | `chapter3_2_firefly_164` | 好啦！看上去真不错，你好上相呀。 |
| 撒娇甜 | `chapter3_2_firefly_135`（蛋糕卷桥段） | 我最喜欢这家店的橡木蛋糕卷，每天都要吃一个。 |
| 温柔 | `chapter3_3_firefly_177` | 这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。 |
| 悲伤 | `chapter3_30_firefly_125` | 嗯，是我不好。对不起。 |
| 惊讶 | `chapter3_30_firefly_123` | 真的吗？看来…卡芙卡教给我的没错。 |
| 生气 | `chapterfinality1_4_firefly_171`（7.09s） | 绝对不行，难道你忘了吗？作为星核猎手，我有一项不惜一切代价的「零点任务」—— |
| 紧张 | `chapter3_20_firefly_162` | 小心，更多怪物进来了！准备开火！ |

> 参考必须与音频文本匹配（`prompt_text`）。曾踩坑：生气提交过 2.6s 的 169_f 剪辑（<3s，genie 直接警告、合成全乱）；撒娇甜换过 163 合影句，不够甜 → 换成蛋糕卷桥段。

---

## 三、训练管线

一键脚本：`tools/run_overnight_pipeline.sh`（已适配 v2ProPlus）

```
流程：数据准备 → 特征提取 → s2(SoVITS 10ep) → s1(GPT 15ep) → 多情绪推理
用法：
  bash tools/run_overnight_pipeline.sh                          # 全流程
  bash tools/run_overnight_pipeline.sh --skip-process --skip-features   # 复用已有数据/特征
```

配套脚本（放在 GPT-SoVITS 目录）：`auto_s2train.sh` / `monitor_train.sh`（崩溃自动续训）、`run_young_features.sh`、`tools/export_ckpt.py`。

### ⚠️ 重大教训：预训练路径必须绝对路径
`tmp_s2.json` 的 `pretrained_s2G/s2D` 若用相对路径 → 预训练被静默跳过、模型随机初始化 → genie 转换后输出高频噪声/空音频。验证：`log_s2_auto.txt` 应出现 `loaded pretrained ... All keys matched successfully` 且无 `kmeans start`。

---

## 四、模型文件（Git LFS）

| 文件 | 大小 | 说明 |
|---|---|---|
| `models/gpt_firefly_678orig-e15.ckpt` | ~150MB | s1 GPT（文本→语义 token），v2 中文 |
| `models/sovits_firefly_678orig_e10.pth` | ~165MB | s2 SoVITS（语义→音频），v2 中文，与成功 torch 基准逐字节一致 |
| `models/gpt_firefly_v3-e15.ckpt` | ~150MB | s1 GPT，**v3 多语言**（e15，发音相对更清晰） |
| `models/gpt_firefly_v3-e45.ckpt` | ~150MB | s1 GPT，**v3 多语言**（续训到 e45，acc↑ 但发音更糊，待排查） |
| `models/sovits_firefly_v3_e10.pth` | ~165MB | s2 SoVITS，**v3 多语言**（未续训） |
| `models/genie/firefly_678orig/` | ~321MB | ~~旧 genie ONNX 产物~~（已弃用，保留参考） |

> 模型文件与 `runs/Airi_678orig_v2ProPlus_v2`（WSL torch 推理基准）所用权重**逐字节一致**（1MB 头哈希已核对）。

---

## 五、推理与实时性（实测数据表）

实测环境：**Windows / RTX 4070 Laptop GPU / torch 2.11.0+cu128 / gsv-tts-lite 0.4.7 / 参考音频预缓存**

| 指标 | 实测值 | 说明 |
|---|---|---|
| **首字延迟（TTFT）** | **~120 ms**（warm HTTP 首音频字节） | token 级流式，比 genie ONNX(1.2~2.4s) 快一个数量级 |
| TTFT（冷启动/首次） | 417–494 ms | bench 8 情绪均值 ~466ms |
| 实时率 RTF | ~0.1 | GPU fp16，合成 3.6s 音频仅需 ~0.5s |
| 流式粒度 | **token 级** | `stream_mode="token"`，25 token 一触发 vits |
| 显存占用 | <1GB | Nested KV Cache + fp16 |

**对比 genie-tts（已弃用）**：TTFT 1.2–2.4s（尾字被截断，音色/语调退化，open issue #31 未修）。

### 语言支持
- **引擎支持中日英**（G2P/BERT 前端 + `text_language` 参数）。
- ⚠️ **firefly 模型只训了中文**：英/日会念不准甚至乱。要英日好需训多语数据或换语种现成角色模型。

---

## 五·五、v3 多语言（2026-08-03）

**目标**：在 v2 中文基础上，让流萤会中/英/日 + 中英混说（code-switching）。

**多语言数据集**（`data/firefly_multilingual/`，1158 条）：
| 语种 | 条数 | 来源 |
|---|---|---|
| 中文 | 678 | 原 v2 数据（复制到统一目录） |
| 英文 | 200 | firefly 台词翻译（SJTU qwen3.6-27b）+ **CosyVoice3 音色克隆** |
| 日文 | 200 | 同上 |
| 中英/中日混合 | 80 | SJTU 生成混合台词 + CosyVoice3 克隆 |

生成管线（Windows）：
- `tools/prep_multilingual.py` —— 翻译（qwen3.6-27b，每 key 10 次/分限速，断点续传）
- `tools/gen_codeswitch.py` —— 生成混合语种台词源
- `tools/gen_multilingual.py` / `gen_multilingual_mix.py` —— CosyVoice3 instruct2 克隆合成（firefly 音色）
- `tools/assemble_multilingual.py` —— 组装 train.list（/mnt/e 路径，2-get-hubert 要求 wav 平铺）

**v3 训练**（WSL，`tools/run_v3_pipeline.sh`）：`firefly_v3`，v2ProPlus，s2 10ep + s1 15ep。
权重：`models/gpt_firefly_v3-e15.ckpt` + `models/sovits_firefly_v3_e10.pth`。

**验证**：`tools/gen_v3_demo.py` → `output/v3_demo/`（zh/en/ja/mix 各一句，试听确认多语言效果）。

---

## 六、流式 TTS 服务（`tools/tts_server.py`）

引擎：GSV-TTS-Lite（torch GPU 推理，质量=torch）+ token 级流式。

```bash
# 启动服务（Windows）
D:/miniforge3/envs/gsv/python.exe tools/tts_server.py          # 0.0.0.0:9880
# 离线测各情绪首字延迟 + 存 wav
D:/miniforge3/envs/gsv/python.exe tools/tts_server.py --bench
# 生成与 torch 基准同文本音频（A/B 用）
D:/miniforge3/envs/gsv/python.exe tools/tts_server.py --compare
```

### API

```
POST /api/tts
  {"text": "今天天气真好呢，我们去散步吧", "emotion": "温柔", "speaker": "firefly",
   "stream_mode": "token"}   # token=首字最低(~120ms) | sentence=句子级更稳(demo 用)
  → 200, Content-Type: audio/L16; rate=32000; channels=1
    响应体 = 原始 int16 PCM 逐块流，首块到达即"首字延迟"
GET /health      → 引擎/模型/GPU/情绪列表
GET /demo        → 浏览器流式试听页（Web Audio 实时播放，验证流式用耳朵）
```

### 试听验证（耳朵最精确）
1. 启动服务后浏览器打开 **`http://localhost:9880/demo`**
2. 选情绪、输入文本 → 点「生成并播放」
3. 首字约 100–500ms 出声，之后连续播放；页面实时显示首块延迟/块数/音频总长

### 输出目录
- `output/bench/`    —— `--bench` 各情绪 wav（基准句）
- `output/compare/`  —— `--compare` 各情绪 短句/长句 wav，与 `runs/Airi_678orig_v2ProPlus_v2/推理` 同名文件 A/B
- `output/`、`GenieData/`、`gsv_models/` 均已 gitignore

---

## 七、部署（Windows/WSL 多端）

| 端 | 位置 | 环境 | 用途 |
|---|---|---|---|
| 训练/开发端 | WSL2 Ubuntu 24.04 | conda `tts` env（Python 3.11、torch 2.5.1+cu121、GPT-SoVITS 完整目录） | 训练、导出权重 |
| **部署/推理端** | **Windows** | conda `gsv` env（Python 3.11 + torch cu128 + gsv-tts-lite，见 `requirements.txt`） | FastAPI 流式 TTS 服务 |

**Windows 部署步骤**：
```bash
conda create -n gsv python=3.11 -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python tools/tts_server.py        # 首次自动下载预训练模型到 gsv_models/（~1.3GB）
```

**多端注意事项**：
- 路径映射：WSL2 `/mnt/e/PROJECT/Airilife_voice_v2` ≡ Windows `E:\PROJECT\Airilife_voice_v2`。
- ⚠️ 本分支是 WSL 建的 linked worktree：`.git` 指向 `/mnt/e/...`，**Windows git 打不开本目录**；提交/推送去主仓库 `E:\PROJECT\Airilife_voice`（共享对象库，`git push origin v2`）。
- GPU：Windows 用 CUDA 12.8；WSL2 内 CUDA 镜像内建。

---

## 八、验证进度

- [x] genie-tts 2.0.2 装于 `tts` conda 环境，`GENIE_DATA_DIR` 指向 GenieData
- [x] v2Pro **不行**（ONNX 骨架不兼容）；v2ProPlus **可行**
- [x] genie 噪声根因 = 预训练未加载（codebook 近零），重训修复
- [x] 重训后转 genie 频谱正常，但**音色/尾字仍有退化 → 弃用 genie**
- [x] **GSV-TTS-Lite 替换**：Windows GPU 部署，质量=torch，TTFT ~120ms，token 级流式
- [x] 流式 TTS 服务 `tools/tts_server.py`：`/api/tts` 流式 + `/demo` 试听 + `--bench`/`--compare`
- [x] 参考音频修正：撒娇甜（蛋糕卷桥段）、生气（7.09s 171）
- [x] **v3 多语言**：翻译 + CosyVoice3 克隆生成 480 条多语言数据 → 训练 `firefly_v3`（中/英/日/混合）→ demo 验证
- [ ] ⚠️ **v3 中文发音模糊（待排查）**：s1 e15→e45 续训后更糊（acc↑ 反而更差），疑 s2 未续训/过拟合/多语言数据稀释。日志在 `docs/v3_training_logs/`，排查说明见其 README
- [ ] AiriLife 端对接（流式消费 / QQ 语音）
- [ ] v3 换用英语/日语参考音频优化英日音色

---

## 九、目录结构（本分支）

```
├── README.md              # 本文档
├── requirements.txt       # Windows 推理端依赖（torch cu128 + gsv-tts-lite + fastapi）
├── tools/                 # 核心脚本
│   ├── tts_server.py      # 流式 TTS 服务（GSV 后端）：server / --bench / --compare / /demo
│   ├── run_overnight_pipeline.sh
│   ├── run_v3_pipeline.sh # v3 多语言训练管线（WSL）
│   ├── prep_multilingual.py / gen_codeswitch.py / gen_multilingual.py / gen_multilingual_mix.py / assemble_multilingual.py
│   ├── gen_v3_demo.py     # v3 多语言验证
│   ├── infer_emotions.py
│   └── export_ckpt.py
├── models/                # Git LFS
│   ├── gpt_firefly_678orig-e15.ckpt
│   ├── sovits_firefly_678orig_e10.pth
│   ├── gpt_firefly_v3-e15.ckpt      # v3 多语言
│   ├── sovits_firefly_v3_e10.pth    # v3 多语言
│   └── genie/firefly_678orig/   # 已弃用 ONNX
├── reference_audio/       # 8 情绪原声参考（撒娇甜/生气已换新）
├── gsv_models/            # GSV 预训练模型（cnhubert/roberta/g2p/sv，~1.3GB，gitignore）
├── GenieData/             # genie 残留数据（gitignore）
├── output/                # 推理输出（bench/compare/stream，gitignore）
└── .gitattributes         # LFS 规则
```
