#!/bin/bash
# 定稿全链路（可复用）—— v2ProPlus 版
# 数据: firefly_678_orig（678条筛选原声，无 asetrate/EQ 处理，年轻感靠推理参考带）
# 流程: 数据准备 → 特征 → s2(10) → s1(15) → 多情绪推理
# 输出唯一目录: /mnt/e/PROJECT/Airilife_voice/runs/Airi_678orig_v2ProPlus/
#
# 用法:
#   bash tools/run_overnight_pipeline.sh
#   bash tools/run_overnight_pipeline.sh --skip-process   # train.list 已生成
#   bash tools/run_overnight_pipeline.sh --skip-features  # 特征已提好
set -euo pipefail

PROJ=/home/witcheng/PROJECT/AiriLife_voice
GS=/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS
PY=/home/witcheng/miniforge3/envs/tts/bin/python
EXP=firefly_678orig
DATA=$PROJ/data/firefly_678_orig
VERSION=v2ProPlus
EXP_DIR=$GS/exp/$EXP
OUT=/mnt/e/PROJECT/Airilife_voice/runs/Airi_678orig_v2ProPlus_v2
REF_DIR=/mnt/e/PROJECT/Airilife_voice/runs/Airi_678orig_v2ProPlus/参考
STATUS=$PROJ/OVERNIGHT_STATUS.txt
LOG=$PROJ/overnight_pipeline.log
S2_EPOCHS=10
S1_EPOCHS=15

SKIP_PROCESS=0
SKIP_FEATURES=0
for arg in "$@"; do
  case "$arg" in
    --skip-process) SKIP_PROCESS=1 ;;
    --skip-features) SKIP_FEATURES=1 ;;
  esac
done

# 追加日志（不截断，方便回看）
exec > >(tee -a "$LOG") 2>&1

status() {
  echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"
}

status "==== 管线启动 version=$VERSION data=678原声 skip_process=$SKIP_PROCESS skip_features=$SKIP_FEATURES ===="

# ---------- 0. 清理残留 ----------
status "0) 清理残留训练进程/锁"
pkill -f "s2_train[.]py" 2>/dev/null || true
pkill -f "s1_train[.]py" 2>/dev/null || true
pkill -f "auto_s2train[.]sh" 2>/dev/null || true
pkill -f "monitor_train[.]sh" 2>/dev/null || true
# 清 flock 锁文件（进程已死后锁可能残留概念上无文件句柄，但删无妨）
rm -f /tmp/auto_s2train.lock /tmp/monitor_train.lock /tmp/monitor_s1.lock 2>/dev/null || true
sleep 2
# GPU 空闲检查
mem=$($PY -c "import subprocess; print(subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).split()[0])" 2>/dev/null || echo 0)
status "0) GPU mem used=${mem}MiB"

# ---------- 1. 数据准备（678 原声，无需 asetrate/EQ 处理） ----------
if [ "$SKIP_PROCESS" = "0" ]; then
  status "1) 数据=678原声，准备 train.list"
  if [ ! -f "$DATA/train.list" ]; then
    # 与 young_full 同文件清单、同文本，仅替换 wav 路径前缀指向原声目录
    sed "s|/data/firefly_young_full/|/data/firefly_678_orig/|g" \
      "$PROJ/data/firefly_young_full/train.list" > "$DATA/train.list"
    status "1) 从 young_full 清单生成 train.list（原声路径）"
  fi
  NLIST=$(wc -l < "$DATA/train.list" || echo 0)
  status "1) train.list=$NLIST"
  if [ "$NLIST" -lt 100 ]; then
    status "FATAL: train.list 太少 ($NLIST)"
    exit 1
  fi
else
  status "1) 跳过数据准备"
  NLIST=$(wc -l < "$DATA/train.list" || echo 0)
  status "1) 现有 train.list=$NLIST"
fi

# ---------- 2. 特征提取 ----------
if [ "$SKIP_FEATURES" = "0" ]; then
  status "2) 特征提取 exp=$EXP（清旧特征）"
  rm -rf "$EXP_DIR"
  mkdir -p "$EXP_DIR"
  bash "$GS/run_young_features.sh" "$EXP" "$DATA" "$VERSION"
  SEM=$(wc -l < "$EXP_DIR/6-name2semantic.tsv" 2>/dev/null || echo 0)
  HUB=$(ls "$EXP_DIR/4-cnhubert" 2>/dev/null | wc -l)
  WAV32=$(ls "$EXP_DIR/5-wav32k" 2>/dev/null | wc -l)
  status "2) semantic=$SEM hubert=$HUB wav32k=$WAV32"
  if [ "$SEM" -lt 100 ] || [ "$HUB" -lt 100 ] || [ "$WAV32" -lt 100 ]; then
    status "FATAL: 特征不完整 semantic=$SEM hubert=$HUB wav32k=$WAV32"
    tail -30 "$EXP_DIR/log_semantic.txt" 2>/dev/null || true
    exit 1
  fi
else
  status "2) 跳过特征"
  SEM=$(wc -l < "$EXP_DIR/6-name2semantic.tsv" 2>/dev/null || echo 0)
  status "2) 现有 semantic=$SEM"
fi

# ---------- 3. 写 s2/s1 配置 ----------
status "3) 写训练配置 (version=$VERSION)"
mkdir -p "$EXP_DIR/logs_s2_v2ProPlus" "$EXP_DIR/logs_s1_v2ProPlus"

$PY - <<PY
import json, pathlib
gs = pathlib.Path("$GS")
exp = gs / "exp" / "$EXP"
# v2ProPlus 完整模板 = 晓伊v2 已训成的 config（含 upsample_initial_channel=768 / kernel[20,...] / lora_rank / pretrained v2ProPlus）
src = json.loads((gs / "logs/晓伊v2/config.json").read_text(encoding="utf-8"))
src["data"]["exp_dir"] = str(exp)
src["s2_ckpt_dir"] = str(exp)
src["name"] = "$EXP"
src["save_weight_dir"] = str(gs / "GPT_SoVITS/SoVITS_weights_v2ProPlus")
src["train"]["epochs"] = $S2_EPOCHS
src["train"]["save_every_epoch"] = 1
# 关键：预训练必须用绝对路径！auto_s2train.sh 在 GPT_SoVITS 子目录下运行，
# 相对路径 "GPT_SoVITS/..." 会 os.path.exists()==False → 预训练被静默跳过（曾导致随机初始化）
src["train"]["pretrained_s2G"] = str(gs / "GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth")
src["train"]["pretrained_s2D"] = str(gs / "GPT_SoVITS/pretrained_models/v2Pro/s2Dv2ProPlus.pth")
# 确保目录字段存在（s2_train 不会自动建 logs）
(exp / "logs_s2_v2ProPlus").mkdir(parents=True, exist_ok=True)
(exp / "tmp_s2.json").write_text(json.dumps(src, ensure_ascii=False), encoding="utf-8")
print("tmp_s2.json ok", src["version"], src["save_weight_dir"], "upsample=", src["model"]["upsample_initial_channel"])
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
status "4) s2 训练启动 (${S2_EPOCHS} epoch)"
# 若已有完成的 ckpt 且 epoch 够，跳过
ep=0
if [ -f "$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth" ]; then
  ep=$($PY -c "
import torch
d=torch.load('$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth', map_location='cpu', weights_only=False)
print(int(d.get('iteration', 0)))
" 2>/dev/null || echo 0)
fi

if [ "$ep" -ge "$S2_EPOCHS" ]; then
  status "4) s2 已完成 epoch=$ep，跳过训练"
else
  setsid nohup bash "$GS/auto_s2train.sh" "$EXP" "$S2_EPOCHS" "$VERSION" > /dev/null 2>&1 < /dev/null &
  disown || true
  setsid nohup bash "$GS/monitor_train.sh" "$EXP" "$S2_EPOCHS" "$VERSION" > /dev/null 2>&1 < /dev/null &
  disown || true

  for i in $(seq 1 360); do
    if [ -f "$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth" ]; then
      ep=$($PY -c "
import torch
d=torch.load('$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth', map_location='cpu', weights_only=False)
print(int(d.get('iteration', 0)))
" 2>/dev/null || echo 0)
      status "4) s2 当前 epoch=$ep ($i)"
      if [ "$ep" -ge "$S2_EPOCHS" ]; then
        status "4) s2 完成 epoch=$ep"
        break
      fi
    else
      status "4) s2 等待首个 checkpoint... ($i)"
    fi
    if ! pgrep -f "s2_train[.]py" >/dev/null && ! pgrep -f "auto_s2train[.]sh" >/dev/null; then
      if [ "$ep" -lt "$S2_EPOCHS" ]; then
        status "4) s2/auto 都死了，重拉 auto_s2train (ep=$ep)"
        setsid nohup bash "$GS/auto_s2train.sh" "$EXP" "$S2_EPOCHS" "$VERSION" > /dev/null 2>&1 < /dev/null &
        disown || true
      fi
    fi
    sleep 120
  done
fi

ep=$($PY -c "
import torch,os
p='$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth'
print(0 if not os.path.exists(p) else int(torch.load(p, map_location='cpu', weights_only=False).get('iteration',0)))
" 2>/dev/null || echo 0)
if [ "$ep" -lt "$S2_EPOCHS" ]; then
  status "FATAL: s2 未完成 epoch=$ep"
  tail -50 "$EXP_DIR/log_s2_auto.txt" 2>/dev/null || true
  exit 1
fi

pkill -f "auto_s2train[.]sh" 2>/dev/null || true
pkill -f "monitor_train[.]sh" 2>/dev/null || true
pkill -f "s2_train[.]py" 2>/dev/null || true
sleep 3

# s2 推理权重
S2_W=$(ls -t "$GS/GPT_SoVITS/SoVITS_weights_v2ProPlus/${EXP}"*.pth 2>/dev/null | head -1 || true)
if [ -z "${S2_W:-}" ]; then
  status "4) 导出 s2 权重"
  S2_OUT="$GS/GPT_SoVITS/SoVITS_weights_v2ProPlus/${EXP}_e${S2_EPOCHS}_export.pth"
  $PY "$PROJ/tools/export_ckpt.py" \
    --src "$EXP_DIR/logs_s2_v2ProPlus/G_233333333333.pth" \
    --dst "$S2_OUT" \
    --config "$EXP_DIR/tmp_s2.json" \
    --version v2ProPlus
  S2_W=$S2_OUT
fi
status "4) s2 权重: $S2_W"

# ---------- 5. s1 训练 ----------
status "5) s1 GPT 训练 (${S1_EPOCHS} epoch)"
cd "$GS"
if [ -f "$GS/GPT_weights_v2ProPlus/${EXP}-e${S1_EPOCHS}.ckpt" ]; then
  status "5) s1 e${S1_EPOCHS} 已存在，跳过"
else
  $PY -s GPT_SoVITS/s1_train.py --config_file "$EXP_DIR/tmp_s1.yaml" \
    > "$EXP_DIR/log_s1_train.txt" 2>&1 || status "WARN: s1 退出码非0，检查是否已完成"
  if [ ! -f "$GS/GPT_weights_v2ProPlus/${EXP}-e${S1_EPOCHS}.ckpt" ]; then
    status "5) 未见到 e${S1_EPOCHS}，再启一次"
    $PY -s GPT_SoVITS/s1_train.py --config_file "$EXP_DIR/tmp_s1.yaml" \
      >> "$EXP_DIR/log_s1_train.txt" 2>&1 || true
  fi
fi
S1_W=$(ls -t "$GS/GPT_weights_v2ProPlus/${EXP}-e${S1_EPOCHS}.ckpt" 2>/dev/null | head -1 || \
       ls -t "$GS/GPT_weights_v2ProPlus/${EXP}"-e*.ckpt 2>/dev/null | head -1 || true)
status "5) s1 权重: ${S1_W:-MISSING}"
if [ -z "${S1_W:-}" ] || [ -z "${S2_W:-}" ]; then
  status "FATAL: 缺权重 s1=$S1_W s2=$S2_W"
  exit 1
fi

# ---------- 6. 多情绪推理 ----------
status "6) 多情绪推理 → $OUT (ref=$REF_DIR)"
rm -rf "$OUT"
mkdir -p "$OUT"
$PY "$PROJ/tools/infer_emotions.py" --s1 "$S1_W" --s2 "$S2_W" --out "$OUT" --ref-dir "$REF_DIR"
# 写权重指针
cat > "$OUT/权重与配方.txt" <<EOF
exp=$EXP
version=$VERSION
s1=$S1_W
s2=$S2_W
train_data=$DATA (678 原声, 无 asetrate/EQ)
ref_dir=$REF_DIR (年轻化参考, 带 asetrate+0.5)
EOF
status "6) 推理完成，文件数: $(find "$OUT" -type f | wc -l)"
status "==== 全部完成 ===="
ls -la "$OUT/推理" 2>/dev/null || ls -la "$OUT"
echo "DONE $(date)" >> "$STATUS"
