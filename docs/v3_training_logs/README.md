# v3 多语言训练日志（排查用）

## 现状：v3 发音模糊（未解决）

- **v3 多语言 demo 可跑**（`output/v3_demo/`），但中文发音不清晰。
- s1 从 e15 续训到 e45 后 **更模糊**（top_3_acc 0.605→0.970 反而更差）。

## 现象与观察

| 版本 | top_3_acc | 发音 | 备注 |
|---|---|---|---|
| e15（15ep） | ~0.605 | 可以但发音不够清晰 | demo 可听 |
| e45（续训 45ep） | ~0.970 | **更模糊** | 语义 token 预测准了，但合成更糊 |

可能方向（待排查）：
1. **s2（SoVITS）没续训** —— 只有 s1 训到 45，s2 还是 10ep。发音模糊可能主要在 s2 语义→音频解码。
2. s1 过拟合 / 过度收敛导致韵律变平。
3. 多语言数据（英/日克隆音色）混入后，中文表达被稀释。
4. CosyVoice3 生成的英/日音频本身有伪影，训进去固化了。

## 日志文件说明

- `log_1text.txt` —— 音素提取（G2P）
- `log_hubert.txt` —— hubert 特征（0 字节=第二次成功跑未捕获，可忽略）
- `log_sv.txt` —— 说话人特征
- `log_semantic.txt` —— 语义 token 提取
- `log_s1_train.txt` —— s1 首次训练（e15）
- `log_s1_cont.txt` —— s1 续训（e15→e45）
- `tmp_s1.yaml` / `tmp_s1_cont.yaml` —— s1 训练配置（对比 epochs）
- `v3_STATUS.txt` / `v3_pipeline.log` —— 管线状态与日志

## 相关脚本

- `tools/continue_s1_v3.sh 45` —— s1 续训
- `tools/run_v3_pipeline.sh` —— v3 全流程训练
- `tools/gen_v3_emotions.py` —— 全情绪中文验证（改 GPT 指向 e15/e45 对比）
