# EcoSort — Beverage Container Classification System

這個專案是依據 *EcoSort: Beverage Container Classification System* 報告內容製作，使用 **YOLOv7** 進行飲料容器（塑膠瓶、鋁罐、玻璃瓶）分類。  
整體流程已拆分為四份 Python 腳本，方便在不同步驟分開執行與測試。  

## 說明

### 1. `dataset_prep.py`
- 功能：資料集下載與前處理  
- 主要內容：
  - 使用 **Roboflow API** 下載 Beverage Containers dataset（或手動上傳解壓）。  
  - 建立 YOLOv7 所需的目錄結構 (`images/train`, `labels/train` 等)。  
  - 可選擇執行 Data Augmentation（如旋轉、翻轉、亮度調整）。  

### 2. 模型訓練  
- 主要內容：
  - 使用 YOLOv7 官方 `train.py` 腳本。  
  - 設定超參數：
    - 影像大小：640  
    - batch size：16  
    - 訓練 epoch：300  
  - 會輸出訓練過程的 log 與最佳模型 `best.pt`。  

### 3. 模型測試與評估  
- 主要內容：
  - 使用 YOLOv7 官方 `test.py` 腳本，載入訓練好的 `best.pt` 權重。  
  - 輸出：
    - mAP (mean Average Precision)  
    - Precision / Recall  
    - 測試圖片推論結果（含 bounding box 與分類標籤）。  

### 4. 模型轉檔  
- 主要內容：
  - PyTorch `.pt` → ONNX `.onnx` → TensorFlow SavedModel → TensorFlow Lite `.tflite`  
  - 方便部署到 **Raspberry Pi / NVIDIA Jetson / Google Coral**。  

## 執行順序

1. `dataset_prep.py` — 先下載並整理資料集。  
2. 開始訓練 YOLOv7 模型。  
3. 測試模型效果，觀察 mAP 與分類正確率。  
4. 將模型轉換為適合嵌入式部署的格式。  
'''
# 1) 匯出 ONNX
python pt_to_onnx.py --weights /path/to/best.pt --imgsz 640 640 --opset 12 --out model.onnx --yolov7

# 2) ONNX → SavedModel → TFLite（可選量化）
pip install tf2onnx tensorflow==2.10.0 onnx
python onnx_to_tflite.py --onnx model.onnx --saved saved_model --tflite model.tflite --quant dynamic
# quant 可選：none / dynamic / float16
'''

## 需求環境
- Python 3.8+  
- GPU (建議使用 Colab 或 CUDA 環境)  
- 主要套件：`torch`, `torchvision`, `onnx`, `tensorflow`, `roboflow`, `opencv-python`, `matplotlib`, `tqdm`  
