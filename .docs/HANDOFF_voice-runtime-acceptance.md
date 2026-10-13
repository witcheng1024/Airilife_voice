# Handoff: voice-runtime-acceptance

- 分支：`feat/voice-runtime-acceptance`
- 基线：`origin/v2=0874794`。此语音仓库没有 `origin/main`；v2 是现用服务基线。
- 作者：Codex
- 日期：2026-10-13

## 1. 目标

修复 AiriLife 真实语音验收发现的角色冷切换等待，保持 GPT-SoVITS 模型、音色情绪参考和 PCM 协议。

## 2. 本机基线与验证

本分支先纳入现用服务的 `tools/tts_server.py`、`tools/voice_registry.py`，包括角色注册、真实 health、串行化推理与取消释放。这两个文件来自现用 v2 工作区的未提交改动；不包含该目录的 README、训练脚本、梓涵模型或参考音频改动。两文件 Python 编译检查通过，现用服务实际请求证据由 AiriLife 真实验收文档维护。

## 3. 未验证

冷切换优化尚未完成，不能以 health 成功宣称首音性能通过；音色听感、macOS/Linux 和休眠未验。

## 4. 保留的用户工作

`E:/PROJECT/Airilife_voice_v2` [win] 是原有运行/训练工作区，保持全部未提交改动与模型文件。本分支独立，不 reset、覆盖或合并该工作区。

## 5. 假设与风险

GSV-TTS-Lite 0.4.7 的参考 prompt 依赖 SoVITS 权重，不能在两个权重之间直接共享同路径缓存。实现前已读实际引擎源码；不得只根据 API 名称假设缓存独立。

## 6. 已知问题

现用 `_load_speaker` 在切换时销毁公共引擎、重载所有情绪，并在 async HTTP handler 同步执行。此前 AiriLife 真实梓涵冷切样本18.82秒；需要优化并实测，旧报告保留。

## 7. 测试边界

真实 HTTP PCM、加载期间 health、取消后再次请求和 GPU 占用是本轮目标；不冒充 LLM、renderer、声学和 QQ 验收。

## 8. 环境

本机 Windows 推理使用 `D:/miniforge3/envs/gsv/python.exe` [win]，GSV-TTS-Lite 0.4.7。Git 元数据属于 Ubuntu24.04 WSL，使用该仓库现有 WSL Git；Windows Python仍在Windows执行。

## 9. 运行副作用

独立 worktree 位于 `E:/PROJECT/Airilife_voice/.claude/worktrees/voice-runtime-acceptance` [win]，LFS smudge 已跳过，目录内权重是指针，不能直接作为推理权重。后续验收必须显式读取原语音工作区的真实资源，不能误用指针文件。

## 10. 下一步

保留改动前真实 HTTP 基线，完成引擎复用/按需参考准备、非阻塞加载和取消保护，再实际比较。

## 11. 交付

工作分支独立提交，原工作区不合并。语音模型、生成音频、服务日志和 AiriLife 本机隔离库不提交。
