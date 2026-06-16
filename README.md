# Airilife Voice — CosyVoice3 TTS 测试

CosyVoice3-0.5B 的本地测试环境，用于为 [Airilife](../Airilife) 项目评估 TTS 的延迟、音色克隆与情感控制能力。

## 测试结论速览

- **模型**：Fun-CosyVoice3-0.5B-2512（PyTorch，非 GGUF）
- **音色机制**：纯 zero-shot 克隆，**无内置音色库**。音色完全来自你提供的参考音频。
- **情感能力**：
  - 预训练（可靠）：开心 / 伤心 / 生气（3 种）
  - 自由指令（实验性）：撒娇 / 温柔 / 可爱 / 惊讶等，效果不保证
- **性能**：RTF ≈ 1.2–1.9（RTX 4070 Laptop，fp16）。注：speech_tokenizer 的 onnxruntime 目前跑在 CPU，装 `onnxruntime-gpu` 可提速。

## 环境搭建

### 1. 克隆官方源码 + 子模块（关键：必须带 Matcha-TTS）

```bash
git clone --depth 1 https://github.com/FunAudioLLM/CosyVoice.git
git clone --depth 1 https://github.com/shivammehta25/Matcha-TTS.git CosyVoice/third_party/Matcha-TTS
```

### 2. 创建虚拟环境并安装依赖

```powershell
python -m venv .venv
& ".venv\Scripts\python.exe" -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
& ".venv\Scripts\python.exe" -m pip install -r requirements.txt
```

> **关键依赖坑（务必注意）**：
> - `transformers==4.51.3` —— **必须钉死此版本**。装成 5.x 会导致 Qwen2 LLM 生成错乱，输出 70-80% 静音的垃圾音频。
> - 文本正则化用 `wetext`（纯 Python，pip 直装），**不需要** WeTextProcessing/pynini（Windows 编译失败且无必要）。

### 3. 下载模型

```python
from modelscope import snapshot_download
snapshot_download('FunAudioLLM/Fun-CosyVoice3-0.5B-2512',
                  local_dir='pretrained_models/Fun-CosyVoice3-0.5B')
```

> 注意：用官方 `AutoModel` 加载，它会自动识别 CosyVoice3 并注入 `qwen_pretrain_path` 绝对路径。
> **不要**手动复制/重命名 `cosyvoice3.yaml` 为 `cosyvoice.yaml`——会让 AutoModel 误判为 v1。

## 运行测试

```powershell
& ".venv\Scripts\python.exe" run_test.py
```

输出在 `test_outputs/`，并生成 `result.json` 性能报告。

## 关键 API 用法

```python
import sys
sys.path.append('CosyVoice')
sys.path.append('CosyVoice/third_party/Matcha-TTS')
from cosyvoice.cli.cosyvoice import AutoModel

cosyvoice = AutoModel(model_dir='pretrained_models/Fun-CosyVoice3-0.5B', fp16=True)

# zero-shot 克隆：prompt_text 必须带 <|endofprompt|> 标记
cosyvoice.inference_zero_shot(
    '要合成的文本',
    'You are a helpful assistant.<|endofprompt|>参考音频对应的文字',
    'reference_audio/xiaoyi_neutral.wav',  # 传文件路径，不是 tensor
    stream=False)

# instruct2 情感控制
cosyvoice.inference_instruct2(
    '要合成的文本',
    'You are a helpful assistant. 请非常开心地说一句话。<|endofprompt|>',
    'reference_audio/xiaoyi_neutral.wav',
    stream=False)
```

## 参考音频

`reference_audio/xiaoyi_neutral.wav` —— 用 edge-tts (`zh-CN-XiaoyiNeural`) 生成，对应 Airilife 当前的 Edge TTS 音色。
如需更多音色，生成多段不同参考音频即可（每段 = 一个可克隆的音色）。
