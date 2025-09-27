#!/usr/bin/env python3
# 將 ONNX 轉 TensorFlow SavedModel，再轉 TFLite
# 需要：pip install tf2onnx tensorflow==2.10.0

import argparse, os
import tensorflow as tf
import onnx
import tf2onnx

def onnx_to_savedmodel(onnx_path, saved_dir):
    model_onnx = onnx.load(onnx_path)
    tf_rep, _ = tf2onnx.convert.from_onnx(model_onnx, output_path=None, opset=12)
    tf.saved_model.save(tf_rep.tf_module, saved_dir)
    print(f"[OK] SavedModel -> {saved_dir}")

def savedmodel_to_tflite(saved_dir, out_tflite, quant=None):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_dir)
    if quant == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif quant == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    tflite = converter.convert()
    open(out_tflite, "wb").write(tflite)
    print(f"[OK] TFLite -> {out_tflite}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--saved", default="./saved_model")
    ap.add_argument("--tflite", default="model.tflite")
    ap.add_argument("--quant", choices=["none","dynamic","float16"], default="none")
    args = ap.parse_args()

    os.makedirs(args.saved, exist_ok=True)
    onnx_to_savedmodel(args.onnx, args.saved)
    quant = None if args.quant == "none" else args.quant
    savedmodel_to_tflite(args.saved, args.tflite, quant)
