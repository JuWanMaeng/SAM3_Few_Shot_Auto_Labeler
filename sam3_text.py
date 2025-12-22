import os

import matplotlib.pyplot as plt
import numpy as np

import sam3
from PIL import Image
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


import cv2
import numpy as np

def save_opencv_results(pil_img, results, output_name="result"):
    """
    pil_img: 원본 PIL 이미지
    results: SAM3 모델 추론 결과 (masks, scores 포함)
    output_name: 저장할 파일명 접두사 (경로 포함 가능)
    """
    # 1. PIL 이미지를 OpenCV 포맷(BGR)으로 변환
    img_cv = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    h, w = img_cv.shape[:2]
    
    # 전체 마스크를 합치기 위한 빈 캔버스
    combined_mask = np.zeros((h, w), dtype=np.uint8)
    
    nb_objects = len(results["scores"])
    masks_tensor = results["masks"]
    scores_tensor = results["scores"]

    print(f"Total objects found: {nb_objects}")

    # --- [Loop] 각 객체별 처리 및 저장 ---
    for i in range(nb_objects):
        # (1) 데이터 추출
        # 마스크: Tensor -> Numpy
        m = masks_tensor[i].detach().cpu().squeeze().numpy()
        mask_uint8 = (m > 0).astype(np.uint8) # 0 또는 1
        
        # 점수 가져오기 (파일명에 쓰기 위함)
        score = scores_tensor[i].item()

        # (2) 전체 마스크(Combined)에 현재 마스크 누적 (Union)
        combined_mask = np.maximum(combined_mask, mask_uint8)

        # (3) ★ 개별 객체 Overlay 생성 및 저장 ★
        # 매번 원본 이미지를 복사해서 새로 그립니다.
        individual_overlay = img_cv.copy()
        
        # 해당 마스크 영역만 빨간색 칠하기
        individual_overlay[mask_uint8 > 0] = [0, 0, 255]
        
        # 투명도 합성
        alpha = 0.5
        individual_blended = cv2.addWeighted(individual_overlay, alpha, img_cv, 1 - alpha, 0)
        
        # 개별 파일 저장 (파일명: result_obj001_score0.89.jpg)
        # 점수를 파일명에 넣으면 나중에 분석하기 편합니다.
        indiv_filename = f"text_outputs/{output_name}_obj{i:03d}_score{score:.2f}.jpg"
        cv2.imwrite(indiv_filename, individual_blended)
        
        # (옵션) 개별 흑백 마스크도 필요하면 아래 주석 해제
        # cv2.imwrite(f"{output_name}_obj{i:03d}_mask.png", mask_uint8 * 255)

    # --- [Final] 전체 합쳐진 결과 저장 ---
    # 3. 전체 빨간색 오버레이 생성
    total_overlay = img_cv.copy()
    total_overlay[combined_mask > 0] = [0, 0, 255]
    
    # 4. 전체 합성
    total_blended = cv2.addWeighted(total_overlay, 0.5, img_cv, 0.5, 0)
    
    # 5. 전체 결과 저장
    total_overlay_filename = f"text_outputs/{output_name}_combined_overlay.jpg"
    total_mask_filename = f"text_outputs/{output_name}_combined_mask.png"

    cv2.imwrite(total_overlay_filename, total_blended)
    cv2.imwrite(total_mask_filename, combined_mask * 255)
    
    print(f"Saved Combined: {total_overlay_filename}")
    print(f"Saved Individuals: {nb_objects} files (pattern: {output_name}_objXXX_score.jpg)")

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")

bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(bpe_path=bpe_path, checkpoint_path='models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt')

image_path = f"{sam3_root}/assets/images/test_image.jpg"
image_path = f"cell_particle.png"
image = Image.open(image_path)
width, height = image.size
processor = Sam3Processor(model, confidence_threshold=0.5)
inference_state = processor.set_image(image)

processor.reset_all_prompts(inference_state)
inference_state = processor.set_text_prompt(state=inference_state, prompt="particle")

img0 = Image.open(image_path)
save_opencv_results(img0, inference_state,output_name='sam3_test')