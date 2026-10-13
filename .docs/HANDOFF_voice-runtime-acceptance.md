# Handoff: voice-runtime-acceptance

- 分支：`feat/voice-runtime-acceptance`
- 基线：`origin/v2=0874794`。此语音仓库没有 `origin/main`；v2 是现用服务基线。
- 作者：Codex
- 日期：2026-10-13

## 1. 目标

已修复 AiriLife 真实语音验收发现的公共模型反复重载和health阻塞；保持GPT-SoVITS音色、参考与PCM协议，直连短句冷切21.028/16.171s降至3.151/2.781s。所有请求恒定1–2秒仍未证明。

## 2. 本机基线与验证

本分支先在6149412纳入现用服务的 `tools/tts_server.py`、`tools/voice_registry.py`，包括角色注册、真实health、串行化推理与取消释放。这两个文件来自现用v2工作区的未提交改动；不包含该目录的README、训练脚本、梓涵模型或参考音频改动。随后公共引擎复用、单套角色权重、当前情绪按需准备、换权重清参考缓存、执行器加载和连续取消保护、英文G2P启动预热已实现并部署。

| 范围 | 回归/检查 | 真实证据 |
|---|---|---|
| 原生worker取消完成后才释放模型锁 | Python unittest 3项通过 | 加载前与首PCM后取消，后续真实请求可继续 |
| 音色冷切与正确模型路径 | Python编译检查 | 10次真实HTTP，speaker响应头与实际health模型对应 |
| 加载时health响应 | 不以mock代替 | 102次采样，65次loading，零超时，最大181ms |
| 英文文本初始化 | 启动预热 | 首G2P3795ms/次次0.314ms；连续轮询启动23.423s |
| AiriLife真实模型链路 | 另仓库回归及包构建 | 实际包冷切5.509/4.770s、热2.366s到Gateway PCM |

## 3. 未验证

音色听感、macOS/Linux和休眠未验。HTTP PCM、响应头和正确权重不能证明扬声器首音或听感；也没有证明所有请求稳定1–2秒。取消无法中断已运行的原生模型加载，立即换另一个角色可能需要两次加载，本次5–6秒。其它语音模型版本和长期GPU趋势未验。

## 4. 保留的用户工作

`E:/PROJECT/Airilife_voice_v2` [win] 是原有运行/训练工作区。应用优化前逐字比较tts_server.py基线并保存本机备份，只应用该单文件；当前原目录和本worktree源码再次逐字比较一致。README、gen_multilingual、voice_registry和梓涵模型/参考/训练未跟踪文件等用户工作保留，不reset或合并该工作区。

## 5. 假设与风险

GSV-TTS-Lite 0.4.7 的参考 prompt 依赖 SoVITS 权重，不能在两个权重之间直接共享同路径缓存。实现前已读实际引擎源码；不得只根据 API 名称假设缓存独立。

## 6. 已知问题

旧 `_load_speaker` 销毁公共引擎、重载所有情绪并同步阻塞HTTP，已经修复。现在保留公共BERT/CNHubert/SV，最多一套角色GPT/SoVITS权重。参考签名包含文件mtime/文本/语言，换权重清原生参考缓存，每次推理显式权重。模型/参考文件缺失或加载失败仍暴露错误，不用health包装为ready。

## 7. 测试边界

本仓库实测HTTP PCM、health、取消、GPU；AiriLife的目录包真实模型与Gateway证据另行记录，不冒充renderer、声学或QQ验收。资源快照Torch allocated1501.9MiB/reserved1694.0MiB，一套角色权重；整机GPU5571/8188MiB包含其它应用和非Torch分配，不能全部归本服务。未对真实音频作声学评分。

## 8. 环境

本机 Windows 推理使用 `D:/miniforge3/envs/gsv/python.exe` [win]，GSV-TTS-Lite 0.4.7。Git 元数据属于 Ubuntu24.04 WSL，使用该仓库现有 WSL Git；Windows Python仍在Windows执行。

## 9. 运行副作用

独立 worktree 位于 `E:/PROJECT/Airilife_voice/.claude/worktrees/voice-runtime-acceptance` [win]，LFS smudge 已跳过，目录内权重是指针，不能直接作为推理权重。后续验收必须显式读取原语音工作区的真实资源，不能误用指针文件。

当前9880/PID56592、`D:/miniforge3/envs/gsv/python.exe`，cwd为原语音工作区，服务保留。旧及临时候选PID按具体身份停止，没有结束用户个人QQ。新环境变量 `AIRILIFE_VOICE_RESOURCE_ROOT` 可显式分离代码和资源路径；默认根目录行为不变。

## 10. 下一步

本轮完成后提交/推送工作分支，保留原目录dirty和本worktree。后续测音色听感及平台恢复；不重复等两小时或改主语音模型。AiriLife发音词典独立在应用语音层，Airi→爱莉不放进本仓库或角色prompt。

## 11. 交付

工作分支独立提交，原工作区不合并。语音模型、生成音频、服务日志和 AiriLife 本机隔离库不提交。

详细证据由 `E:/PROJECT/AiriLife/.claude/worktrees/real-user-acceptance/.docs/TTS_LIVE_PIPELINE_ACCEPTANCE_2026-10-13.md` [win] 最后章节维护。本机原报告 `.artifacts/live-voice-switch-baseline`、`live-voice-switch-candidate`、`live-voice-switch-application`、`live-voice-switch-final-cancel`、`live-packaged-voice-runtime-warm-g2p` 与 `voice-runtime-resource-final.json` 保留。服务完整启动23.423s是连续轮询值，旧25.8/32秒是上界，不替代。
