# ## 2) 從 Roboflow 下載資料集
# 
# 方法 A（Roboflow Python）: 用 Roboflow API 下載 dataset，請先到 Roboflow 取得 `ROBOFLOW_API_KEY`。
# 
# 方法 B：在 Roboflow 網頁上匯出為 YOLOv5 / YOLO 格式手動下載，然後上傳到執行環境。


# 範例：使用 roboflow 套件下載（請先在環境變數或 secret 中設定 ROBOFLOW_API_KEY）
from roboflow import Roboflow
import os

# 請在系統環境變數中設置 ROBOFLOW_API_KEY，或直接覆寫下列變數（不建議在公共場所洩露）
ROBOFLOW_API_KEY = os.environ.get('ROBOFLOW_API_KEY', '<YOUR_API_KEY_HERE>')
if ROBOFLOW_API_KEY == '<YOUR_API_KEY_HERE>':
    print('請將 ROBOFLOW_API_KEY 設為你的 API Key，或改用手動下載。')
else:
    rf = Roboflow(api_key=ROBOFLOW_API_KEY)
    project = rf.workspace().project('<your-workspace-name-or-project-slug>')
    # 範例：project.version(1).download('yolov5')
    print('請改成你的 workspace 與專案 slug。')

# --- 或者，若你已經手動下載並上傳 zip，解壓範例 ---
# !unzip /path/to/beverage-containers.zip -d dataset




# ## 3) 資料前處理 (轉成 YOLO 格式)
# 
# 此區將 XML/COCO/YOLOv5 檔案轉成 yolov7 可用的資料夾結構（`datasets/yourdataset/images/train`, `labels/train` ...）。


import os
import shutil

# 假設 dataset 已解壓為 dataset/ 目錄，並遵循 Roboflow 的 structure
DATA_ROOT = 'dataset'
print('請確認', DATA_ROOT, '資料夾存在且包含 images/ 與 labels/ 子資料夾。')

# 範例函式：建立 yolov7 期望的目錄
def prepare_yolov7_structure(data_root='dataset', out_root='yolov7_dataset'):
    splits = ['train','valid','test']
    for s in splits:
        os.makedirs(os.path.join(out_root,'images',s), exist_ok=True)
        os.makedirs(os.path.join(out_root,'labels',s), exist_ok=True)
    print('建立目錄:', out_root)

prepare_yolov7_structure(DATA_ROOT, 'yolov7_dataset')
print('請手動將 images 與 labels 按 split 放到 yolov7_dataset 下')




# ### 資料增強 (augmentation)
# 
# 可以使用 Albumentations 或 YOLOv7 內建的 augmentation。以下示例用 albumentations 產生 augmented 圖片與對應標籤。


# 範例（簡化版）：用 albumentations 做 augmentation
# pip install albumentations

import random
from PIL import Image
import numpy as np

print('此 cell 僅為示範，請根據你的資料調整。')




