import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
import glob
from PIL import Image
from sam3.model_builder import build_sam3_video_model
from sam3.visualization_utils import show_mask, show_points

# --- 1. 설정 및 경로 ---
INPUT_SIZE = (1024, 1024)
TEMP_DIR = "./temp_frames"

# [변경] 결과물 폴더 구조 개선
BASE_OUTPUT_DIR = "./output"
MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)

# --- 1. 설정 ---
INPUT_SIZE = (1024, 1024) 
TEMP_DIR = "./temp_frames"     # SAM3 구동용 임시 폴더

# [사용자 설정 필요] 이미지 및 모델 경로
REF_IMAGES = [
    "C:/data/Sam3/stitch/2_crop26_rot88.jpg", "C:/data/Sam3/stitch/2_crop26_rot220.jpg", "C:/data/Sam3/stitch/2_crop26_rot261.jpg", "C:/data/Sam3/stitch/2_crop26_rot298.jpg", "C:/data/Sam3/stitch/2_crop26_rot340.jpg"
]
TARGET_FOLDER = "C:/data/Sam3/stitch/target"

# [사용자 설정] 5장 Reference의 박스 좌표
HARDCODED_BOXES = [
    [215, 522, 1022, 680],
    [137, 2, 792, 709],
    [1, 293, 814, 542],
    [0, 265, 748, 781],
    [207, 198, 585, 1019]
]

class BatchLabelerWithOverlay:
    def __init__(self):
        print("Loading SAM 3 Model...")
        # device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam3_model = build_sam3_video_model()
        self.predictor = self.sam3_model.tracker
        self.predictor.backbone = self.sam3_model.detector.backbone
        print("Model Loaded.")

        self.ref_frames = []
        self.ref_prompts = []
        
        print("Pre-loading reference images...")
        for i, path in enumerate(REF_IMAGES):
            img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            self.ref_frames.append(np.array(img))
            
            box = HARDCODED_BOXES[i]
            rel_box = np.array([[
                box[0] / INPUT_SIZE[0], box[1] / INPUT_SIZE[1],
                box[2] / INPUT_SIZE[0], box[3] / INPUT_SIZE[1]
            ]], dtype=np.float32)
            self.ref_prompts.append(rel_box)

    def run(self):
        target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.jpg")) + 
                              glob.glob(os.path.join(TARGET_FOLDER, "*.png")))
        
        print(f"Found {len(target_files)} images in {TARGET_FOLDER}")
        print(f"Results will be saved to: {BASE_OUTPUT_DIR}")

        for idx, target_path in enumerate(target_files):
            filename = os.path.basename(target_path)
            base_name = os.path.splitext(filename)[0]
            print(f"[{idx+1}/{len(target_files)}] Processing: {filename}...", end=" ")
            
            try:
                # 1. 원본 이미지 로드 (오버레이 생성을 위해 필요)
                target_img_pil = Image.open(target_path).convert("RGB")
                
                # 2. 추론 실행 (PIL 객체를 전달)
                final_mask = self.process_single_image(target_img_pil)
                
                if final_mask is not None:
                    # --- A. 마스크 저장 (PNG) ---
                    mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name + "_mask.png")
                    Image.fromarray((final_mask * 255).astype(np.uint8)).save(mask_save_path)
                    
                    # --- B. 오버레이 생성 및 저장 (JPG) ---
                    # 색상: (R, G, B), 투명도: alpha (0.0 ~ 1.0)
                    overlay_img = self.create_overlay(target_img_pil, final_mask, color=(255, 50, 50), alpha=0.6)
                    overlay_save_path = os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg")
                    overlay_img.save(overlay_save_path, quality=95)
                    
                    print(f"-> Success! Saved mask & overlay.")
                else:
                    print("-> Failed (No object detected)")
            except Exception as e:
                print(f"-> Error: {e}")
                import traceback
                traceback.print_exc()

    def process_single_image(self, target_img_pil):
        # 1. 이미지 전처리
        target_orig_size = target_img_pil.size
        target_img = np.array(target_img_pil.resize(INPUT_SIZE))
        
        # 2. 임시 파일 저장
        all_frames = self.ref_frames + [target_img]
        for i, frame in enumerate(all_frames):
            Image.fromarray(frame).save(os.path.join(TEMP_DIR, f"{i:05d}.jpg"))
            
        # 3. 추론 상태 초기화 및 Reference 주입
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        # self.predictor.clear_all_points_in_video(inference_state)
        
        obj_id = 1
        for i, box in enumerate(self.ref_prompts):
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=i,
                obj_id=obj_id,
                box=box
            )
            
        # 4. 추론 수행
        current_mask = None
        target_frame_idx = 5
        
        for out_idx, out_ids, _, masks, _ in self.predictor.propagate_in_video(
            inference_state, start_frame_idx=0, max_frame_num_to_track=6, reverse=False, propagate_preflight=True
        ):
            if out_idx == target_frame_idx:
                for i, oid in enumerate(out_ids):
                    if oid == obj_id:
                        current_mask = (masks[i] > 0.0).cpu().numpy().squeeze()

        # 5. 결과 리사이징 및 반환
        final_mask_resized = None
        if current_mask is not None:
            if current_mask.ndim == 3:
                current_mask = current_mask.squeeze()
                
            dsize = tuple(map(int, target_orig_size))
            final_mask_resized = cv2.resize(
                current_mask.astype(np.uint8), 
                dsize, 
                interpolation=cv2.INTER_NEAREST
            )
            
        return final_mask_resized

    # [추가됨] 오버레이 이미지 생성 함수
    def create_overlay(self, original_img_pil, mask_np, color=(255, 0, 0), alpha=0.5):
        """
        원본 이미지 위에 마스크를 지정된 색상과 투명도로 덧칠합니다.
        """
        img_np = np.array(original_img_pil)
        overlay_np = img_np.copy()
        
        # 마스크가 있는 영역(값이 1인 곳) 불리언 인덱싱
        mask_bool = mask_np > 0
        
        # 색상 배열 준비 (RGB)
        rgb_color = np.array(color, dtype=np.uint8)
        
        # 블렌딩 공식: 원본 * (1-alpha) + 색상 * alpha
        # 마스크된 영역에만 적용
        overlay_np[mask_bool] = (img_np[mask_bool] * (1 - alpha) + rgb_color * alpha).astype(np.uint8)
        
        return Image.fromarray(overlay_np)

if __name__ == "__main__":
    app = BatchLabelerWithOverlay()
    app.run()