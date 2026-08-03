# v3 资源盘点（2026-08-03 清理前存档）

## 模型（models/，v3 变体 + v2 基线）

| 保留 | 大小 | 说明 |
|---|---|---|
| `gpt_firefly_678orig-e15.ckpt` | 155MB | v2 s1（中文基线，已推 LFS） |
| `sovits_firefly_678orig_e10.pth` | 172MB | v2 s2（已推 LFS） |
| `gpt_firefly_v3-e15.ckpt` | 155MB | **v3 s1 e15**（发音较清晰） |
| `gpt_firefly_v3-e20.ckpt` | 155MB | **v3 s1 e20**（定版） |
| `sovits_firefly_v3_e30_cont.pth` | 172MB | v3 s2 e30 |
| `sovits_firefly_v3_e35_cont.pth` | 172MB | v3 s2 e35 |
| `sovits_firefly_v3_e40_cont.pth` | 172MB | **v3 s2 e40（定版）** |
| `sovits_firefly_v3_e45_cont.pth` | 172MB | v3 s2 e45 |
| `sovits_firefly_v3_e50_cont.pth` | 172MB | v3 s2 e50 |

**删除**：v3 s1 e5/e10/e25/e30/e35/e40/e45、v3 s2 e10/e15/e20/e25（网格排查用变体，已出结论）。
**定版组合**：s1e20_s2e40 + text_language=zh（s1e15 作为更清晰备选）。

## 多语言数据集（data/firefly_multilingual/，保留）

| 语种 | wav 数 | 大小 |
|---|---|---|
| zh | 678 | 213MB |
| en | 200 | 43MB |
| ja | 200 | 46MB |
| mix | 80 | 19MB |
| 平铺 | 1158 | ~320MB |

来源：678 中文原声 + 200 英/日（SJTU 翻译 + CosyVoice3 克隆）+ 80 混合。

## 源数据（删除）

- data/firefly_clean (261MB)、firefly_young_full (206MB)、firefly_zh (548MB)、firefly_678_orig (206MB) —— 原始源，已并入 firefly_multilingual
- WSL 原始包：archive/pck_full/wem (~33GB) + wav/pck (~1.5GB) —— 游戏原始音频包，无需保留

## 训练日志（docs/v3_training_logs/，保留）

特征提取 + s1/s2 训练日志 + 配置（排查用）。
