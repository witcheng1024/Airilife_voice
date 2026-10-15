# 新海天中文训练与音色切换

本轮实验标识为 `zihan_cn_20261001`。优先完成新海天中文，保留现有流萤模型与默认音色。

## 状态补记（2026-10-15）

本机 `status.json` 为 `phase=finished, state=completed`；选定 `s1e5_s2e10`，固定测试 10/10 通过、平均 CER 约 0.0095，服务音频 ASR 为 8/8。`subjective_audition=pending`，仍待最终听音确认。

本文描述本机训练流程，所列训练脚本、梓涵权重和参考音频仍未随本次文档提交发布。双音色运行代码已在 `feat/voice-runtime-acceptance`，尚未合入 `v2`；资源与代码的分发范围见 [README](../README.md)。2026-10-15 已按用户要求停止本机 9880 服务，下文保留的是原训练/恢复流程，不表示服务仍在运行。

## 数据与模型

- 原始日文来源：HF Space `10086mea/GPT-SoVITS-NIMI_SORA` 的 `9nine/nimi_sora`。全量下载不作为训练前提，实际可用文件以本地清单为准。
- 从已下载素材中筛选两条原声参考，以及最多 300 条日文 anchor；筛除异常时长、静音、严重削波等样本。
- 用现有 Windows CosyVoice3 环境生成中文教师数据，最终模型仍是我们自己训练的 GPT-SoVITS **v2ProPlus**，不是部署 CosyVoice。
- 中文源语料为 **800 条去重的日常域模板组合**，不是 800 条独立人工撰写台词；含平淡、活泼、温柔、打趣、害羞、惊讶、生气、悲伤。
- 先保存 A/B 原声参考、自然/轻快指令的 30 条试听样本，按中文 ASR 可懂度自动选择教师方案。自动选择不等于用户已经试听认可。
- 中文教师样本使用 Paraformer ASR：CER ≤ 0.12、时长 1.2–12 秒、RMS ≥ 0.005。至少 600 条中文通过才训练，日文数量约为中文数量的 37.5%，上限 300。
- 原新海天权重的 GPT vocab=512、SoVITS text embedding=322，和当前训练基座的 732 不一致；旧 G 的 upsample channel=512，v2ProPlus=768。不能直接将旧权重当作本轮续训基座。
- 从匹配的绝对路径预训练权重开始；SoVITS discriminator 已核对发布 SHA256：`635cd84bf6f7f9b8d41c88c7106f81d782c794c61f931845214ea037b0c5bef2`。
- 本轮补齐了 WSL 原先缺少的 `G2PWModel/g2pW.onnx`。归档来自训练源码 README 指定的 ModelScope 镜像，下载后核对 SHA256 `46292be0374a49308069233cd5c147ae4c41806558e4781a2467a31a4d8099da`，只提取模型权重；没有修改 GPT-SoVITS 源码或安装新依赖。

## 独立训练

- WSL：`Ubuntu24.04`，Python：`/home/witcheng/miniforge3/envs/tts/bin/python`。
- 实验目录：`/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS/exp/zihan_cn_20261001`。
- 先提取文本/BERT、HuBERT/32k WAV、speaker embedding、semantic 特征，文本与语义覆盖率不足 97% 则失败。
- S2：20 epochs、batch 2、gradient checkpointing，每 5 epochs 导出；OOM 时回退 batch 1。
- S1：15 epochs、batch 4、DPO 关闭，每 epoch 导出；OOM 时依次回退 batch 2/1。
- 导出目录：`models/zihan/gpt` 与 `models/zihan/sovits`，不覆盖流萤。
- 使用文件锁防重入，数据指纹防止把变更后的数据与旧特征/训练状态混用。完成条件包含最终 epoch 权重，不以“生成过一个 checkpoint”冒充训练完成。

## 自动验证与发布

1. 从通过筛选的中文教师样本中，为八种情绪各选一条文本准确、时长适合的参考。
2. 对比 S1/S2：`5/10`、`10/10`、`15/10`、`10/20`、`15/20`。
3. 每组生成同一组 10 句，覆盖多情绪、数字与长句，保存 WAV、首块时间和 RTF。
4. 用同一中文 ASR 检查固定测试集。至少 8/10 通过、平均 CER ≤ 0.15 才允许发布。
5. 在 CER 接近最优的组合中，用 ERes2Net speaker cosine similarity 辅助选择；跨语种声纹分数只作参考，不代替人耳音色验收。
6. 原子写入 `models/zihan/voice.json`，TTS 服务动态读取音色注册表。
7. 实际 HTTP 验证流萤→新海天→流萤、token/sentence 输出、双请求排队、无效音色、中断后恢复，并保存 PCM 转换的 WAV。
8. 优先使用 9880；若旧服务占用且不支持双音色，则另选 9881–9885，不结束已有服务。成功启动的新服务仅绑定 `127.0.0.1`，保持运行。

## 状态与产物

- 当前状态：`output/zihan_20261001/status.json`；真实阶段事件：`events.jsonl`。
- 主日志：`worker.log`；教师生成：`teacher_generate.log`；训练入口：`training.log`。
- 教师筛选：`data/zihan_cn_20261001/teacher_quality.json`；最终训练数量：`dataset_summary.json`。
- 原模型兼容性：`output/zihan_20261001/checkpoint_compatibility.json`。
- 候选试听：`output/zihan_20261001/preview`；训练后对比：`output/zihan_20261001/evaluation`。
- 最终选择：`output/zihan_20261001/final_selection.json`；服务地址与 HTTP 检查：`service_connection.json`。
- `phase=finished, state=completed` 表示本轮自动流程通过；`subjective_audition=pending` 表示尚待用户听音验收。
- 任一阶段失败会记录真实错误并停止，不绕过质量门槛、不发布未通过验证的模型。

## 运行与恢复

在 Windows 项目根目录执行：

```powershell
D:/miniforge3/python.exe -u -B tools/run_zihan_overnight.py
```

不要重复启动已有 worker；先看 PID 与状态。生成、ASR、特征与训练各自支持复用/续跑。
不要使用旧 `run_overnight_pipeline.sh` 来恢复本轮，它包含清理进程/实验目录的操作。

若模型验证已完成、只需重跑服务检查：

```powershell
D:/miniforge3/envs/gsv/python.exe -u -B tools/verify_zihan_service.py
```

若上一轮仅在发布/服务检查阶段失败，可执行 `D:/miniforge3/python.exe -u -B tools/run_zihan_overnight.py --finalize-only`，保留原模型与验证结果并更新管线状态，不重复训练。

## TTS 接口

`GET /api/voices` 或 `/api/speakers` 返回 ID、标签、就绪状态和每个音色支持的情绪；`/demo` 可通过下拉菜单切换。

```json
{"text":"早上好呀，今天也要好好吃饭。","speaker":"zihan","emotion":"温柔","stream_mode":"sentence"}
```

省略 `speaker` 仍用 `firefly`。输出格式保持单声道、32kHz、int16 原始 PCM；`X-TTS-Speaker` 响应头标明实际音色。
为了在 8GB GPU 上保留显存余量，一次只驻留一套音色模型，切换时重载并预缓存参考；冷切换延迟包含模型加载，不应冒充稳态 TTFT。
流式请求在结束/取消前独占切换锁，避免上一条声音尚未合成完就换模型。现有 `--bench`、`--compare`、`GSV_GPT/GSV_SOV` 仍保留。
HTTP 流采用可中止的后台推理桥：断连后在下一音频块停止并等待推理线程退出，随后才释放切换锁，避免仍在执行的 CUDA 推理与新模型 CUDA Graph capture 冲突。

这些模型与素材仅用于当前本地研究/试听流程；没有完成素材权利或商业使用授权核验。音色相似度、自然度、日语口音与情绪强度仍需用户最终试听确认。
