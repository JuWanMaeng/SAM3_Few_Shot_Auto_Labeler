import os
import glob
import cv2
import numpy as np
from PIL import Image
import torch

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ==========================================
# 1. 설정 (Configuration)
# ==========================================
# 입력 이미지가 있는 폴더 경로
INPUT_FOLDER = "/data/Sam3/bump/ref" 
# 결과 저장 경로
OUTPUT_FOLDER = "outputs_masks"
# 모든 이미지에 적용할 텍스트 프롬프트
TEXT_PROMPT = "line" 
# 모델 체크포인트 경로 (사용자 환경에 맞게 수정)
CHECKPOINT_PATH = 'models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt'

# 폴더 생성
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
os.makedirs(INPUT_FOLDER, exist_ok=True)

# ==========================================
# 2. 모델 로드 (한 번만 수행)
# ==========================================
print(f"Loading SAM3 Model...")
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"

model = build_sam3_image_model(
    bpe_path=bpe_path, 
    checkpoint_path=CHECKPOINT_PATH
)

# 프로세서 초기화
processor = Sam3Processor(model, confidence_threshold=0.5) # 필요시 threshold 조절
print("Model Loaded Successfully.")

# ==========================================
# 3. 이미지 처리 함수
# ==========================================
def process_batch_images():
    # 처리할 이미지 확장자들
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))
    
    # 파일 이름 순 정렬
    image_files.sort()
    
    if not image_files:
        print(f"No images found in '{INPUT_FOLDER}'. Please check the path.")
        return

    print(f"Found {len(image_files)} images. Start processing with prompt: '{TEXT_PROMPT}'")

    for idx, img_path in enumerate(image_files):
        filename = os.path.basename(img_path)
        base_name = os.path.splitext(filename)[0]
        
        print(f"[{idx+1}/{len(image_files)}] Processing: {filename} ...")

        try:
            # 1. 이미지 로드 및 전처리
            pil_image = Image.open(img_path).convert("RGB")
            
            # 2. SAM3 추론 준비
            inference_state = processor.set_image(pil_image)
            processor.reset_all_prompts(inference_state)
            
            # 3. 텍스트 프롬프트 적용 및 결과 획득
            # results 딕셔너리에 masks, scores 등이 포함됨
            results = processor.set_text_prompt(state=inference_state, prompt=TEXT_PROMPT)
            
            # 4. 결과 마스크 생성 및 저장
            save_combined_mask(pil_image, results, base_name)
            
        except Exception as e:
            print(f"Error processing {filename}: {e}")

def save_combined_mask(pil_img, results, base_name):
    """
    여러 객체가 검출되었을 경우 하나로 합쳐서(Union) 저장합니다.
    """
    # PIL -> OpenCV (BGR)
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    
    # 빈 마스크 캔버스 (0: 배경, 255: 객체)
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    masks_tensor = results.get("masks", [])
    scores_tensor = results.get("scores", [])
    
    nb_objects = len(scores_tensor)
    
    if nb_objects == 0:
        print(f" -> No objects found for prompt '{TEXT_PROMPT}'")
        return

    # --- 마스크 합치기 ---
    for i in range(nb_objects):
        # Tensor -> Numpy 변환
        m = masks_tensor[i].detach().cpu().squeeze().numpy()
        mask_uint8 = (m > 0).astype(np.uint8)
        
        # 합집합 (Union): 기존 마스크와 현재 마스크 중 하나라도 1이면 1
        combined_mask = np.maximum(combined_mask, mask_uint8)

    # --- 저장 ---
    # 1. 마스크 파일 (흑백)
    mask_filename = os.path.join(OUTPUT_FOLDER, f"{base_name}_mask.png")
    cv2.imwrite(mask_filename, combined_mask * 255)
    
    # 2. 오버레이 파일 (확인용, 원본 + 붉은색 마스크)
    overlay = img_cv.copy()
    overlay[combined_mask > 0] = [0, 0, 255] # Red
    blended = cv2.addWeighted(overlay, 0.5, img_cv, 0.5, 0)
    
    overlay_filename = os.path.join(OUTPUT_FOLDER, f"{base_name}_overlay.jpg")
    cv2.imwrite(overlay_filename, blended)
    
    print(f" -> Saved: {mask_filename} (Objects: {nb_objects})")

# ==========================================
# 4. 실행
# ==========================================
if __name__ == "__main__":
    # 입력 폴더에 이미지가 없다면 더미 파일 생성 안내 (테스트용)
    if not os.path.exists(INPUT_FOLDER) or not os.listdir(INPUT_FOLDER):
        print(f"Warning: '{INPUT_FOLDER}' is empty. Please put images there.")
    
    process_batch_images()