#!/bin/bash
# v3 多语言训练管线 —— v2ProPlus（678中文 + 200英 + 200日 + 80代码切换）
# 在 WSL 跑。数据在 /mnt/e（Windows），训练环境在 WSL。
#
# 用法:
#   bash tools/run_v3_pipeline.sh              # 全流程
#   bash tools/run_v3_pipeline.sh --skip-features  # 特征已提好，直接训练
set -euo pipefail

GS=/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS
PY=/home/witcheng/miniforge3/envs/tts/bin/python
EXP=firefly_v3
DATA=/mnt/e/PROJECT/Airilife_voice_v2/data/firefly_multilingual
VERSION=v2ProPlus
EXP_DIR=$GS/exp/$EXP
OUT=/mnt/e/PROJECT/Airilife_voice/runs/Airi_v3_multilingual
S2_EPOCHS=10
S1_EPOCHS=15

SKIP_FEATURES=0
for arg in "$@"; do
  case "$arg" in
    --skip-features) SKIP_FEATURES=1 ;;
  esac
done

LOG=$DATA/../v3_pipeline.log
STATUS=$DATA/../v3_STATUS.txt
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

status() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

status "==== v3 多语言管线 version=$VERSION data=$DATA skip_features=$SKIP_FEATURES ===="

# ---------- 0. 清理 ----------
pkill -f "s2_train[.]py" 2>/dev/null || true
pkill -f "s1_train[.]py" 2>/dev/null || true
rm -f /tmp/auto_s2train.lock /tmp/monitor_train.lock /tmp/monitor_s1.lock 2>/dev/null || true
sleep 2

# ---------- 1. train.list 校验 ----------
if [ ! -f "$DATA/train.list" ]; then
  status "FATAL: 缺 $DATA/train.list（先跑 assemble_multilingual.py）"
  exit 1
fi
NLIST=$(wc -l < "$DATA/train.list")
status "1) train.list=$NLIST 行"
langs=$(cut -d'|' -f3 "$DATA/train.list" | sort | uniq -c)
status "1) 语言分布: $langs"

# ---------- 2. 特征提取 ----------
if [ "$SKIP_FEATURES" = "0" ]; then
  status "2) 特征提取 exp=$EXP"
  rm -rf "$EXP_DIR"
  mkdir -p "$EXP_DIR"
  bash "$GS/run_young_features.sh" "$EXP" "$DATA" "$VERSION"
  SEM=$(wc -l < "$EXP_DIR/6-name2semantic.tsv" 2>/dev/null || echo 0)
  HUB=$(ls "$EXP_DIR/4-cnhubert" 2>/dev/null | wc -l)
  WAV32=$(ls "$EXP_DIR/5-wav32k" 2>/dev/null | wc -l)
  status "2) semantic=$SEM hubert=$HUB wav32k=$WAV32"
  if [ "$SEM" -lt 300 ] || [ "$HUB" -lt 300 ] || [ "$WAV32" -lt 300 ]; then
    status "FATAL: 特征不完整 semantic=$SEM"
    tail -40 "$EXP_DIR/log_1text.txt" 2>/dev/null || true
    tail -40 "$EXP_DIR/log_semantic.txt" 2>/dev/null || true
    exit 1
  fi
else
  status "2) 跳过特征"
fi

# ---------- 3. 写训练配置（复用 v2 模板，phoneme_vocab_size 732 = 多语言） ----------
status "3) 写训练配置"
mkdir -p "$EXP_DIR/logs_s2_v2ProPlus" "$EXP_DIR/logs_s1_v2ProPlus"

$PY - <<PY
import json, pathlib
gs = pathlib.Path("$GS")
exp = gs / "exp" / "$EXP"
src = json.loads((gs / "logs/晓伊v2/config.json").read_text(encoding="utf-8"))
src["data"]["exp_dir"] = str(exp)
src["s2_ckpt_dir"] = str(exp)
src["name"] = "$EXP"
src["save_weight_dir"] = str(gs / "GPT_SoVITS/SoVITS_weights_v2ProPlus")
src["train"]["epochs"] = $S2_EPOCHS
src["train"]["save_every_epoch"] = 1
src["train"]["pretrained_s2G"] = str(gs / "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth")
src["train"]["pretrained_s2D"] = str(gs / "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth")
(exp / "logs_s2_v2ProPlus").mkdir(parents=True, exist_ok=True)
(exp / "tmp_s2.json").write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")
print("tmp_s2.json ok", src["version"], "upsample=", src["model"]["upsample_initial_channel"])
PY

cat > "$EXP_DIR/tmp_s1.yaml" <<YAML
data:
  max_eval_sample: 8
  max_sec: 54
  num_workers: 4
  pad_val: 1024
inference:
  top_k: 15
model:
  EOS: 1024
  dropout: 0
  embedding_dim: 512
  head: 16
  hidden_dim: 512
  linear_units: 2048
  n_layer: 24
  phoneme_vocab_size: 732
  random_bert: 0
  vocab_size: 1025
optimizer:
  decay_steps: 40000
  lr: 0.01
  lr_end: 0.0001
  lr_init: 1.0e-05
  warmup_steps: 2000
output_dir: $EXP_DIR/logs_s1_v2ProPlus
pretrained_s1: GPT_SoVITS/pretrained_models/s1v3.ckpt
train:
  batch_size: 8
  epochs: $S1_EPOCHS
  exp_name: $EXP
  gradient_clip: 1.0
  half_weights_save_dir: GPT_weights_v2ProPlus
  if_dpo: false
  if_save_every_weights: true
  if_save_latest: true
  precision: 16-mixed
  save_every_n_epoch: 1
  seed: 1234
train_phoneme_path: $EXP_DIR/2-name2text.txt
train_semantic_path: $EXP_DIR/6-name2semantic.tsv
YAML

# ---------- 4. s2 训练 ----------
status "4) s2 训练 (${S2_EPOCHS} epoch)"
ep=0
if [ -f "$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth" ]; then
  ep=$($PY -c "import torch; print(int(torch.load('$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth', map_location='cpu', weights_only=False).get('iteration',0)))" 2>/dev/null || echo 0)
fi
if [ "$ep" -ge "$S2_EPOCHS" ]; then
  status "4) s2 已完成，跳过"
else
  setsid nohup bash "$GS/auto_s2train.sh" "$EXP" "$S2_EPOCHS" "$VERSION" > /dev/null 2>&1 < /dev/null &
  disown || true
  setsid nohup bash "$GS/monitor_train.sh" "$EXP" "$S2_EPOCHS" "$VERSION" > /dev/null 2>&1 < /dev/null &
  disown || true
  for i in $(seq 1 360); do
    if [ -f "$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth" ]; then
      ep=$($PY -c "import torch; print(int(torch.load('$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth', map_location='cpu', weights_only=False).get('iteration',0)))" 2>/dev/null || echo 0)
      status "4) s2 epoch=$ep ($i)"
      [ "$ep" -ge "$S2_EPOCHS" ] && { status "4) s2 完成"; break; }
    else
      status "4) s2 等待首个 ckpt ($i)"
    fi
    if ! pgrep -f "s2_train[.]py" >/dev/null && ! pgrep -f "auto_s2train[.]sh" >/dev/null; then
      status "4) s2 进程死，重拉 (ep=$ep)"
      setsid nohup bash "$GS/auto_s2train.sh" "$EXP" "$S2_EPOCHS" "$VERSION" > /dev/null 2>&1 < /dev/null &
      disown || true
    fi
    sleep 120
  done
fi
pkill -f "auto_s2train[.]sh" 2>/dev/null || true
pkill -f "monitor_train[.]sh" 2>/dev/null || true
sleep 3

S2_W=$(ls -t "$GS/GPT_SoVITS/SoVITS_weights_v2ProPlus/${EXP}"*.pth 2>/dev/null | head -1 || true)
if [ -z "${S2_W:-}" ]; then
  status "4) 导出 s2 权重"
  S2_OUT="$GS/GPT_SoVITS/SoVITS_weights_v2ProPlus/${EXP}_e${S2_EPOCHS}_export.pth"
  $PY "$GS/../GPT-SoVITS/tools/export_ckpt.py" --src "$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth" --dst "$S2_OUT" --config "$EXP_DIR/tmp_s2.json" --version v2ProPlus 2>/dev/null || \
  $PY /mnt/e/PROJECT/Airilife_voice_v2/tools/export_ckpt.py --src "$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth" --dst "$S2_OUT" --config "$EXP_DIR/tmp_s2.json" --version v2ProPlus
  S2_W=$S2_OUT
fi
status "4) s2 权重: $S2_W"

# ---------- 5. s1 训练 ----------
status "5) s1 训练 (${S1_EPOCHS} epoch)"
cd "$GS"
if [ -f "$GS/GPT_weights_v2ProPlus/${EXP}-e${S1_EPOCHS}.ckpt" ]; then
  status "5) s1 已完成，跳过"
else
  $PY -s GPT_SoVITS/s1_train.py --config_file "$EXP_DIR/tmp_s1.yaml" > "$EXP_DIR/log_s1_train.txt" 2>&1 || true
  if [ ! -f "$GS/GPT_weights_v2ProPlus/${EXP}-e${S1_EPOCHS}.ckpt" ]; then
    $PY -s GPT_SoVITS/s1_train.py --config_file "$EXP_DIR/tmp_s1.yaml" >> "$EXP_DIR/log_s1_train.txt" 2>&1 || true
  fi
fi
S1_W=$(ls -t "$GS/GPT_weights_v2ProPlus/${EXP}"-e*.ckpt 2>/dev/null | head -1 || true)
status "5) s1 权重: ${S1_W:-MISSING}"
if [ -z "${S1_W:-}" ] || [ -z "${S2_W:-}" ]; then
  status "FATAL: 缺权重 s1=$S1_W s2=$S2_W"
  exit 1
fi

# ---------- 6. 输出 ----------
status "6) 输出 → $OUT"
mkdir -p "$OUT"
cat > "$OUT/权重与配方.txt" <<EOF
exp=$EXP
version=$VERSION
s1=$S1_W
s2=$S2_W
train_data=$DATA (678中文 + 200英 + 200日 + 80代码切换)
EOF
status "==== v3 训练全部完成 ===="
echo "DONE $(date)" >> "$STATUS"
