# config.py
import os
import shutil
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 환경 설정
# ==========================================
INPUT_SIZE = (1024, 1024) 
TEMP_DIR = "./temp_frames"
BASE_OUTPUT_DIR = "output"

# 출력 경로 설정
MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
ID_MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "id_masks")
OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")
BBOX_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "bboxes")
REVERSE_MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reverse_masks")
REVERSE_BBOX_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reverse_bboxes")
REVERSE_OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reverse_overlays")

checkpoin_path = 'models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt'

def setup_directories():
    if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
    if os.path.exists(BASE_OUTPUT_DIR): shutil.rmtree(BASE_OUTPUT_DIR)
    
    output_paths = [
        TEMP_DIR, MASK_OUTPUT_DIR, OVERLAY_OUTPUT_DIR, BBOX_OUTPUT_DIR, 
        ID_MASK_OUTPUT_DIR, REVERSE_MASK_OUTPUT_DIR, REVERSE_BBOX_OUTPUT_DIR, REVERSE_OVERLAY_OUTPUT_DIR
    ]
    for path in output_paths:
        os.makedirs(path, exist_ok=True)

# [사용자 데이터 경로]
defect = 'particles'  
REF_DIR = f"/data/Sam3/{defect}/ref"
TARGET_FOLDER = f"/data/Sam3/{defect}/target"
CONFIDENCE_SCORE = -2


# 색상 팔레트
np.random.seed(42)
COLORS = np.random.randint(0, 255, (255, 3), dtype=np.uint8)
MPL_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))