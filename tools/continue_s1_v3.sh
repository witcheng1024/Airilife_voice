#!/bin/bash
# v3 s1 续训：从现有 ckpt 继续更多 epoch（改善发音清晰度）
# s1_train.py 会自动从 output_dir 最新 ckpt 续训；这里把 epochs 提到目标值再跑。
#
# 用法（WSL）:
#   bash tools/continue_s1_v3.sh 45        # 续训到 45 epoch（默认）
#   bash tools/continue_s1_v3.sh 60        # 续训到 60 epoch
set -euo pipefail

GS=/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS
PY=/home/witcheng/miniforge3/envs/tts/bin/python
EXP=firefly_v3
EXP_DIR=$GS/exp/$EXP
TARGET_EP=${1:-45}

# 读现有 tmp_s1.yaml，把 epochs 提到 TARGET_EP，写 tmp_s1_cont.yaml
$PY - <<PY
import yaml
p = "$EXP_DIR/tmp_s1.yaml"
c = yaml.safe_load(open(p, encoding="utf-8"))
c["train"]["epochs"] = $TARGET_EP
out = "$EXP_DIR/tmp_s1_cont.yaml"
yaml.safe_dump(c, open(out, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print(f"续训配置: epochs -> {$TARGET_EP}, output_dir 不变（自动续训）")
PY

cd "$GS"
$PY -s GPT_SoVITS/s1_train.py --config_file "$EXP_DIR/tmp_s1_cont.yaml" > "$EXP_DIR/log_s1_cont.txt" 2>&1 || {
  echo "s1 续训退出码非0，重试一次"
  $PY -s GPT_SoVITS/s1_train.py --config_file "$EXP_DIR/tmp_s1_cont.yaml" >> "$EXP_DIR/log_s1_cont.txt" 2>&1 || true
}

# 最终权重
S1_W=$(ls -t "$GS/GPT_weights_v2ProPlus/${EXP}"-e*.ckpt 2>/dev/null | head -1 || true)
echo "s1 续训完成，最新权重: ${S1_W:-MISSING}"
