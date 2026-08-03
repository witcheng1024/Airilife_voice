#!/bin/bash
# v3 s2 分阶段续训：从 e10 依次续到 e15/e20/e25，每阶段导出推理权重。
# auto_s2train.sh 会从最新 ckpt 自动续训到目标 epoch 并退出。
#
# 用法（WSL）:
#   bash tools/continue_s2_v3.sh            # e15/e20/e25 三阶段
#   bash tools/continue_s2_v3.sh 20 25      # 自定义阶段列表
set -euo pipefail

GS=/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS
PY=/home/witcheng/miniforge3/envs/tts/bin/python
EXP=firefly_v3
VERSION=v2ProPlus
EXPORT=/mnt/e/PROJECT/Airilife_voice_v2/tools/export_ckpt.py
CKPT=$GS/exp/$EXP/logs_s2_v2ProPlus/G_233333333333.pth

STAGES=${@:-15 20 25}
rm -f /tmp/auto_s2train.lock 2>/dev/null || true

for TARGET in $STAGES; do
  echo "===== s2 续训到 e$TARGET ====="
  # 关键：s2_train.py 读的是 tmp_s2.json 里的 epochs，必须先把配置的 epochs 提到目标值
  $PY - <<PY
import json
cfg = json.load(open("$GS/exp/$EXP/tmp_s2.json", encoding="utf-8"))
cfg["train"]["epochs"] = $TARGET
json.dump(cfg, open("$GS/exp/$EXP/tmp_s2.json", "w", encoding="utf-8"), ensure_ascii=False)
print(f"tmp_s2.json epochs -> $TARGET")
PY
  bash $GS/auto_s2train.sh $EXP $TARGET $VERSION
  # 核对 iteration
  ep=$($PY -c "import torch; print(int(torch.load('$CKPT', map_location='cpu', weights_only=False).get('iteration',0)))" 2>/dev/null || echo 0)
  echo "当前 iteration=$ep"
  # 导出推理权重
  OUT=$GS/GPT_SoVITS/SoVITS_weights_v2ProPlus/${EXP}_e${TARGET}_cont.pth
  $PY $EXPORT --src "$CKPT" --dst "$OUT" --config "$GS/exp/$EXP/tmp_s2.json" --version v2ProPlus
  echo "导出: $OUT"
done
echo "===== s2 全部阶段完成 ====="
