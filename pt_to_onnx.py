#!/usr/bin/env python3
# 將 PyTorch .pt 匯出為 ONNX（YOLOv7/一般模型皆可嘗試）
# 用法：
#   python pt_to_onnx.py --weights /path/to/best.pt --imgsz 640 640 --opset 12 --out model.onnx --yolov7
# 若非 YOLOv7，拿掉 --yolov7，並確保你的模型 load 程式正確。

import argparse, os, sys
import torch

def export_generic(weights, imgsz, opset, out):
    # 載入一般 PyTorch 模型（需你自行修改為對應架構的 load 程式）
    model = torch.load(weights, map_location="cpu")
    if isinstance(model, dict) and "model" in model:
        model = model["model"]
    model = model.float().eval()

    dummy = torch.randn(1, 3, imgsz[1], imgsz[0])
    torch.onnx.export(
        model, dummy, out,
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
        verbose=False
    )
    print(f"[OK] Exported ONNX -> {out}")

def export_yolov7(weights, imgsz, opset, out):
    # 需要 yolov7 專案在同層或已安裝為套件
    repo = os.path.join(os.path.dirname(__file__), "yolov7")
    if not os.path.isdir(repo):
        print("[INFO] 未找到 yolov7 專案，請先 git clone 到同層資料夾：")
        print("git clone https://github.com/WongKinYiu/yolov7.git")
        sys.exit(1)
    sys.path.insert(0, repo)
    from models.experimental import attempt_load
    from utils.torch_utils import TracedModel

    device = torch.device("cpu")
    model = attempt_load(weights, map_location=device).float().eval()
    model = TracedModel(model, device, 640)
    dummy = torch.randn(1, 3, imgsz[1], imgsz[0], device=device)

    torch.onnx.export(
        model, dummy, out,
        opset_version=opset,
        input_names=["images"],
        output_names=["output"],
        dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
        verbose=False
    )
    print(f"[OK] Exported YOLOv7 ONNX -> {out}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help=".pt 路徑")
    ap.add_argument("--imgsz", nargs=2, type=int, default=[640, 640], help="W H")
    ap.add_argument("--opset", type=int, default=12)
    ap.add_argument("--out", default="model.onnx")
    ap.add_argument("--yolov7", action="store_true", help="pt 為 YOLOv7 權重")
    args = ap.parse_args()

    if args.yolov7:
        export_yolov7(args.weights, args.imgsz, args.opset, args.out)
    else:
        export_generic(args.weights, args.imgsz, args.opset, args.out)
