#!/usr/bin/env python
"""将 GPT-SoVITS s2 训练检查点导出为推理格式权重（v2Pro 字节05 / v2ProPlus 字节06）"""
import os, sys, json, argparse
from collections import OrderedDict

import torch

GIT_ROOT = "/home/witcheng/PROJECT/TTS-Server/train/GPT-SoVITS"
sys.path.insert(0, GIT_ROOT)
from GPT_SoVITS.process_ckpt import my_save2

MODEL_VERSION2BYTE = {"v3": b"03", "v4": b"04", "v2Pro": b"05", "v2ProPlus": b"06"}


def export(src, dst, config_path, version="v2Pro"):
    ckpt = torch.load(src, map_location="cpu", weights_only=False)
    config = json.load(open(config_path, encoding="utf-8"))
    model = ckpt["model"] if "model" in ckpt else ckpt["weight"]
    opt = OrderedDict()
    opt["weight"] = OrderedDict((k, v.half()) for k, v in model.items() if "enc_q" not in k)
    opt["config"] = config
    opt["info"] = "epoch_%s" % ckpt.get("iteration", "?")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    my_save2(opt, dst, version)
    print(f"导出完成: {dst} ({os.path.getsize(dst)//1024//1024}MB)  iteration={ckpt.get('iteration')}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--version", default="v2Pro")
    a = ap.parse_args()
    export(a.src, a.dst, a.config, a.version)
