# AiriLife 流萤语音 v2 — GPT-SoVITS v2ProPlus + Genie-TTS

Live2D 桌宠「流萤」的语音方案 v2。目标是 **15 岁年龄感 + 多情绪表达**的本地 TTS，满足桌宠实时性要求。

> 本分支（`v2`）是 GPT-SoVITS 方案。旧 CosyVoice3 方案在 `master` 分支，已弃用。
>
> 开发/训练在 WSL2（`/home/witcheng/PROJECT/AiriLife_voice`），git 仓库在 Windows（`E:\PROJECT\Airilife_voice`）。

---

## 一、技术选型

| 环节 | 方案 | 说明 |
|---|---|---|
| 训练 | GPT-SoVITS **v2ProPlus** | 最终要转 genie，必须 v2ProPlus（v2Pro 与 genie 骨架不兼容） |
| 推理加速 | **Genie-TTS 2.0.2** | GPT-SoVITS 的 ONNX 推理引擎，CPU/GPU 均可，首包 ~1s |
| 声线方案 | **原声数据训练 + 参考音频带情绪/年龄感** | EQ/asetrate 变调全部无效，年轻感只能靠推理时换参考 |

**为什么不用 EQ/变调**：G1-G6 多档 EQ 实测无效（除杂声变大无听感变化）；asetrate +2/+3 失败（尖而不幼）。唯一可行的年轻化路径 = 原声训练 + 年轻参考。

---

## 二、数据

- **训练数据**：`data/firefly_678_orig` — 678 条**原声**（从 911 条 `firefly_clean` 里筛出），**无任何 asetrate/EQ 处理**。`train.list` 从 young_full 清单改路径前缀生成。
- **情绪参考**（推理时用，本仓库 `reference_audio/`）：8 情绪原声样本

| 情绪 | 参考样本 | 参考文本 |
|---|---|---|
| 平淡 | `chapter3_2_firefly_223` | 我叫流萤，是鸢尾花家系的艺者。 |
| 活泼 | `chapter3_2_firefly_164` | 好啦！看上去真不错，你好上相呀。 |
| 撒娇甜 | `chapter3_2_firefly_163` | 你想合影留念吗？那就把手机交给我吧，我来拍～ |
| 温柔 | `chapter3_3_firefly_177` | 这里是游客不会踏足的拓荒地，所以不像市中心那样热闹…但我很喜欢这种僻静的气氛。 |
| 悲伤 | `chapter3_30_firefly_125` | 嗯，是我不好。对不起。 |
| 惊讶 | `chapter3_30_firefly_123` | 真的吗？看来…卡芙卡教给我的没错。 |
| 生气 | `chapterfinality1_4_firefly_169_f` | 绝对不行，得把她和{NICKNAME}分开。 |
| 紧张 | `chapter3_20_firefly_162` | 小心，更多怪物进来了！准备开火！ |

> 参考必须与音频文本匹配（`prompt_text`）。选样本注意情绪纯粹性（曾踩坑：温柔样本选成道歉语气）。

---

## 三、训练管线

一键脚本：`tools/run_overnight_pipeline.sh`（已适配 v2ProPlus）

```
流程：数据准备 → 特征提取 → s2(SoVITS 10ep) → s1(GPT 15ep) → 多情绪推理
用法：
  bash tools/run_overnight_pipeline.sh                          # 全流程
  bash tools/run_overnight_pipeline.sh --skip-process --skip-features   # 复用已有数据/特征
```

配套脚本（放在 GPT-SoVITS 目录）：
- `auto_s2train.sh` / `monitor_train.sh`（s2 崩溃自动续训，第 3 参数 VERSION 默认 v2Pro）
- `run_young_features.sh`（特征提取，第 3 参数 VERSION）
- `tools/export_ckpt.py`（s2 权重导出，支持 `--version v2ProPlus` 字节 06）

### v2ProPlus vs v2Pro 关键差异
`upsample_initial_channel 512→768`、`upsample_kernel_sizes [16,16,8,2,2]→[20,16,8,2,2]`、`lora_rank=32`、预训练换 `s2Gv2ProPlus.pth / s2Dv2ProPlus.pth`、`version=v2ProPlus`。

### ⚠️ 重大教训：预训练路径必须绝对路径
`tmp_s2.json` 的 `pretrained_s2G/s2D` 若用相对路径 `GPT_SoVITS/pretrained_models/...`，`auto_s2train.sh` 在 `$BASE/GPT_SoVITS` 下运行时 `os.path.exists()==False` → **预训练被静默跳过，模型随机初始化**。v2ProPlus 量化器 `kmeans_init=True` 会用首 batch 重建 codebook（近零）→ genie 转换后用近零 codebook 建 t2s_encoder → 输出高频噪声/空音频。

**验证训练正确**：`log_s2_auto.txt` 应出现 `loaded pretrained ... <All keys matched successfully>` 且**无** `kmeans start`。

---

## 四、模型文件（Git LFS）

| 文件 | 大小 | 说明 |
|---|---|---|
| `models/gpt_firefly_678orig-e15.ckpt` | ~150MB | s1 GPT（文本→语义 token） |
| `models/sovits_firefly_678orig_e10.pth` | ~165MB | s2 SoVITS（语义→音频） |
| `models/genie/firefly_678orig/` | ~321MB | genie ONNX 产物（vits/t2s_encoder/t2s_shared/prompt_encoder） |

---

## 五、推理

### 1. PyTorch（GPT-SoVITS，精度最高）
```bash
python tools/infer_emotions.py \
  --s1 models/gpt_firefly_678orig-e15.ckpt \
  --s2 models/sovits_firefly_678orig_e10.pth \
  --ref-dir reference_audio/ \
  --out <输出目录>
```

### 2. Genie（ONNX，实时）
```python
import os
os.environ["GENIE_DATA_DIR"] = "/path/to/GenieData"
import genie_tts as genie   # 注意包名是 genie_tts，不是 genie

genie.convert_to_onnx(torch_ckpt_path=s1, torch_pth_path=s2, output_dir="genie/firefly_678orig")
genie.load_character("firefly_678orig", "genie/firefly_678orig", "Chinese")
genie.set_reference_audio("firefly_678orig", ref_wav, ref_text, "Chinese")
genie.tts("firefly_678orig", text, save_path="out.wav")   # 返回 None，落盘到 save_path
```

GenieData（G2P、chinese-hubert-base、speaker_encoder）从 genie 官方下载；中文 RoBERTa 可选。

---

## 六、部署（桌宠 TTS 服务）

**架构**：独立 FastAPI 服务（WSL，GPU 可选）→ AiriLife backend（Windows）通过 HTTP 调用。

```
POST http://localhost:9880/api/tts
Content-Type: application/json
{
  "text": "今天天气真好呢，我们去散步吧",
  "emotion": "温柔",      // 平淡|活泼|撒娇甜|温柔|悲伤|惊讶|生气|紧张，默认平淡
  "speaker": "firefly"
}
→ 200 OK, Content-Type: audio/wav
```

**部署资源**（实测）：
- 硬盘：~720MB（genie onnx 321MB + GenieData 392MB + 参考 2MB）
- 环境：Python 3.11 + `genie_tts` + `onnxruntime` + `fastapi`/`uvicorn`（**无需 PyTorch**）
- 内存：~500MB-1GB（模型常驻）
- GPU：可选（CPU 可跑，GPU ~1s/句）

**完整开发环境**（对比）：conda `tts` env 8.5GB + GPT-SoVITS 完整目录 47GB。

---

## 七、验证进度

- [x] genie-tts 2.0.2 装于 `tts` conda 环境，`GENIE_DATA_DIR` 指向 GenieData
- [x] v2Pro **不行**（转换成功但 ONNX 加载失败，骨架不兼容）
- [x] v2ProPlus **可行**（晓伊v2 权重转换+推理正常）
- [x] 定位 678orig genie 噪声根因 = 预训练未加载（codebook 近零），已修复重训
- [ ] 重训后重新转 genie 验证（进行中）
- [ ] TTS FastAPI 服务搭建
- [ ] AiriLife 端对接（streaming 流式 / QQ 语音）

---

## 八、目录结构（本分支）

```
├── README.md              # 本文档
├── tools/                 # 核心脚本（训练管线/推理/导出）
│   ├── run_overnight_pipeline.sh
│   ├── infer_emotions.py
│   └── export_ckpt.py
├── models/                # Git LFS
│   ├── gpt_firefly_678orig-e15.ckpt
│   ├── sovits_firefly_678orig_e10.pth
│   └── genie/firefly_678orig/
├── reference_audio/       # 8 情绪原声参考
└── .gitattributes         # LFS 规则
```
