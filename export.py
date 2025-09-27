# ## 6) 模型轉檔：PyTorch -> ONNX -> TensorFlow SavedModel -> TFLite
# 
# 以下提供步驟與常見 code snippet。成功轉換需注意 PyTorch 與 ONNX 版本相容性，與模型輸入大小一致。


import torch

# 範例：將 yolov7 PyTorch weights 轉成 ONNX（需要在 yolov7 repo 中執行）
# 請先把 weights 放到 yolov7/runs/train/exp/weights/best.pt

pytorch_weights = 'yolov7/runs/train/exp/weights/best.pt'
onnx_output = 'yolov7/best.onnx'

print('以下為範例程式片段，請在含有 yolov7 repo 與正確環境中執行')

torch_export_snippet = """model = torch.load(pytorch_weights, map_location='cpu')['model'].float().fuse().eval()
dummy_input = torch.randn(1, 3, 640, 640).to('cpu')
torch.onnx.export(model, dummy_input, onnx_output, opset_version=12, verbose=False, input_names=['images'], output_names=['output'])
"""

print(torch_export_snippet)
print('然後使用 tf2onnx 或 onnx-tf 轉成 TensorFlow SavedModel，再用 TFLiteConverter 轉成 .tflite')




# ONNX -> TensorFlow SavedModel（示例 shell 命令）
# python -m tf2onnx.convert --opset 12 --input yolov7/best.onnx --output yolov7/saved_model --inputs-as-nchw images:0
# 或者使用 onnx-tf
# onnx-tf convert -i yolov7/best.onnx -o yolov7/saved_model

# TFLite 轉換示例（TensorFlow 環境）
# import tensorflow as tf
# converter = tf.lite.TFLiteConverter.from_saved_model('yolov7/saved_model')
# converter.optimizations = [tf.lite.Optimize.DEFAULT]
# tflite_model = converter.convert()
# open('model.tflite','wb').write(tflite_model)

print('請參考上述命令在支援的環境中執行轉檔流程')




# ## 7) 推論示範（本地影像）
# 
# 使用 yolov7 的 detect.py 或撰寫自訂程式載入 weights 做推論，下面提供一個簡單的 PyTorch 推論範例（讀圖並顯示 bounding boxes）。


import cv2
import numpy as np
from matplotlib import pyplot as plt

print('示範程式（需在已安裝 torch 與 yolov7 repo 的環境執行') 

# 以下為偽程式，說明如何呼叫 detect.py 產生圖片結果
print('在 yolov7 目錄下可執行:')
print("!python detect.py --weights runs/train/exp/weights/best.pt --img 640 --conf 0.25 --source ../test_images --save-txt --save-conf")




# ## 8) 結果分析範例
# 
# 如何產生混淆矩陣、Precision/Recall 與 PR 曲線（示例程式）：


import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 假設有二進位或多分類預測 label 與 ground-truth lists
# gt = [0,1,2, ...], pred = [0,2,1, ...]
# cm = confusion_matrix(gt, pred)
# print(cm)

print('請使用 test.py 或自訂程式輸出預測結果（每張影像的 pred 與 gt），再套用 sklearn 的 metrics 做分析')




# ## 9) 部署建議（Raspberry Pi / Jetson / Coral）
# 
# - **Jetson Nano/Xavier**: 可直接跑 PyTorch 或使用 TensorRT 加速（建議使用 TensorRT engine）
# - **Raspberry Pi**: 使用 TFLite 模型較合適；若需較高效能可搭配 Coral USB 或 Edge TPU
# - **Google Coral**: 需要編譯成 Edge TPU 相容的 .tflite（量化並編譯）
# 
# 詳細步驟視硬體而定，且轉換時需注意 opset、運算子相容性與量化策略。


# ## Appendix — 專案報告資料
# 
# 此 notebook 依據上傳之專案報告 `EcoSort__Beverage_Container_Classification_System.pdf`（請參閱報告內容）所建。

