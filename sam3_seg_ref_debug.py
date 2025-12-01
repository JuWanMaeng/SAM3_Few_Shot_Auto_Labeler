import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import torch
import cv2
import os
import glob,time
from PIL import Image
from sam3.model_builder import build_sam3_video_model

# ==========================================
# 1. 환경 설정 및 경로 지정
# ==========================================
INPUT_SIZE = (1024, 1024)  # SAM 3 모델의 표준 입력 해상도
TEMP_DIR = "./temp_frames" # 비디오 처리를 위해 임시로 프레임을 저장할 폴더
BASE_OUTPUT_DIR = "./output"
MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")

# 필요한 폴더가 없으면 생성
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)

# [사용자 데이터 경로 설정]
# 레퍼런스(참조) 이미지: 모델에게 "이걸 찾아줘"라고 알려줄 이미지들
REF_IMAGES = [
    "C:/data/Sam3/stitch/ref/2_crop26_rot88.jpg", 
    "C:/data/Sam3/stitch/ref/2_crop26_rot220.jpg", 
    "C:/data/Sam3/stitch/ref/2_crop26_rot261.jpg", 
    # "C:/data/Sam3/stitch/2_crop26_rot298.jpg", # 필요 시 주석 해제
    # "C:/data/Sam3/stitch/2_crop26_rot340.jpg"
]
TARGET_FOLDER = "C:/data/Sam3/stitch/target" # 찾을 대상(Target) 이미지가 있는 폴더

# [하드코딩된 박스 데이터]
# 디버깅 및 자동화를 위해 미리 좌표를 입력해 둡니다.
# 형식: { REF_IMAGES_INDEX: [[x1, y1, x2, y2], ...], ... }
HARDCODED_BOXES = {
    0: [ # 첫 번째 레퍼런스 이미지의 박스들
        [492, 84, 667, 241],
        [482, 503, 635, 681],
        [794, 514, 945, 687],
        [457, 902, 618, 1020],
        [791, 889, 933, 1019]
    ],
    1: [ # 두 번째 레퍼런스 이미지의 박스들
        [748, 92, 932, 260],
        [247, 109, 413, 258],
        [450, 345, 617, 502],
        [111, 615, 283, 786]
    ],
    2: [ # 세 번째 레퍼런스 이미지의 박스들
        [93, 304, 246, 454],
        [408, 341, 550, 506],
        [460, 6, 614, 121],
        [155, 2, 308, 81],
        [320, 785, 476, 935]
    ]
}

# ==========================================
# 2. 메인 처리 클래스
# ==========================================
class InteractiveBatchLabeler:
    def __init__(self):
        print("Loading SAM 3 Model... (모델 로딩 중)")
        # SAM 3 비디오 모델 빌드
        self.sam3_model = build_sam3_video_model()
        self.predictor = self.sam3_model.tracker
        # 트래커와 디텍터가 백본(Backbone)을 공유하도록 설정
        self.predictor.backbone = self.sam3_model.detector.backbone
        print("Model Loaded. (로딩 완료)")

        self.ref_frames = []   # 레퍼런스 이미지 데이터(numpy) 저장용
        self.ref_prompts = []  # 정규화된 박스 좌표 저장용

    def prepare_references(self):
        """
        하드코딩된 박스 데이터를 읽어와서 모델 입력용으로 준비합니다.
        좌표를 0~1 사이의 값으로 정규화(Normalization)하는 과정이 포함됩니다.
        """
        print("\n--- Step 1: Loading Hardcoded Boxes (박스 데이터 로드) ---")
        
        for i, path in enumerate(REF_IMAGES):
            # 하드코딩 데이터가 없으면 스킵
            if i not in HARDCODED_BOXES:
                print(f"  -> Skipping Ref {i+1} (설정된 박스 데이터 없음)")
                continue

            if not os.path.exists(path): 
                print(f"Warning: File not found {path}")
                continue
            
            # 이미지 로드 및 리사이징
            pil_img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            # 해당 이미지에 대한 박스 리스트 가져오기
            boxes = HARDCODED_BOXES[i] 

            self.ref_frames.append(img_arr)
            
            # [좌표 정규화]
            # SAM 모델은 이미지 크기에 상대적인 0.0 ~ 1.0 좌표를 선호합니다.
            np_boxes = np.array(boxes, dtype=np.float32)
            norm_boxes = np_boxes.copy()
            norm_boxes[:, 0] /= INPUT_SIZE[0] # xmin / width
            norm_boxes[:, 1] /= INPUT_SIZE[1] # ymin / height
            norm_boxes[:, 2] /= INPUT_SIZE[0] # xmax / width
            norm_boxes[:, 3] /= INPUT_SIZE[1] # ymax / height
            
            self.ref_prompts.append(norm_boxes)
            print(f"  -> Loaded Ref {i+1}: {len(boxes)} boxes (Hardcoded)")
            
        if not self.ref_frames:
            print("Error: 준비된 레퍼런스가 없습니다.")
            return False
        return True

    def run(self):
        """
        전체 프로세스를 실행합니다. (레퍼런스 준비 -> 타겟 이미지 순회 -> 추론 -> 저장)
        """
        if not self.prepare_references(): return

        # 타겟 폴더 내의 이미지 파일 검색
        target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.jpg")) + glob.glob(os.path.join(TARGET_FOLDER, "*.png")))
        print(f"\n--- Step 2: Processing {len(target_files)} images (타겟 이미지 처리) ---")

        for idx, target_path in enumerate(target_files):
            filename = os.path.basename(target_path)
            base_name = os.path.splitext(filename)[0]
            print(f"[{idx+1}/{len(target_files)}] Processing: {filename}...", end=" ")
            
            try:
                target_img_pil = Image.open(target_path).convert("RGB")
                
                # [핵심] 한 장의 이미지에 대해 SAM 3 추론 실행
                final_mask = self.process_single_image(target_img_pil)
                
                if final_mask is not None:
                    # 1. 마스크 저장 (PNG, 0 or 255)
                    mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name + "_mask.png")
                    Image.fromarray((final_mask * 255).astype(np.uint8)).save(mask_save_path)
                    
                    # 2. 오버레이 이미지 저장 (JPG, 시각적 확인용)
                    overlay_img = self.create_overlay(target_img_pil, final_mask, color=(255, 50, 50), alpha=0.6)
                    overlay_save_path = os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg")
                    overlay_img.save(overlay_save_path, quality=95)
                    
                    print(f"-> Success!")
                else:
                    print("-> Failed (탐지된 객체 없음)")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

    def process_single_image(self, target_img_pil):
        """
        Reference 이미지들과 현재 Target 이미지를 묶어서 비디오처럼 처리합니다.
        """
        target_orig_size = target_img_pil.size
        # 입력 크기로 리사이징
        target_img = np.array(target_img_pil.resize(INPUT_SIZE))
        
        # [프레임 구성] Ref 1, Ref 2, Ref 3 ... + Target Image
        all_frames = self.ref_frames + [target_img]
        
        # SAM 3는 폴더에 저장된 프레임을 읽어오는 방식을 사용하므로 임시 저장
        for i, frame in enumerate(all_frames):
            save_path = os.path.join(TEMP_DIR, f"{i:05d}.jpg")
            Image.fromarray(frame).save(save_path)
            
        # 추론 상태 초기화
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        
        # [중요 전략] 각 박스마다 고유한 Object ID 부여
        # 하나의 ID로 묶으면 특징이 뭉개지므로, 개별적으로 추적 후 나중에 합칩니다.
        global_obj_id = 1
        tracked_obj_ids = []

        for i, boxes_np in enumerate(self.ref_prompts):
            for box in boxes_np:
                # 프롬프트 추가 (Reference 프레임에 박스 지정)
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,            # 몇 번째 레퍼런스 이미지인지
                    obj_id=global_obj_id,   # 고유 ID (1, 2, 3...)
                    box=box,                # 박스 좌표 (정규화됨)
                    clear_old_points=True 
                )
                tracked_obj_ids.append(global_obj_id)
                global_obj_id += 1 # 다음 박스를 위해 ID 증가
            
        # 결과 마스크를 하나로 합칠 변수 (None으로 초기화)
        combined_mask = None
        target_frame_idx = len(self.ref_frames) # 타겟 이미지는 리스트의 맨 마지막
        
        # [추론 실행] 비디오 전파 (Propagate)
        # 과거 프레임(Reference)의 정보를 바탕으로 미래 프레임(Target)을 추론
        for out_idx, out_ids, _, masks, _ in self.predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=0, 
            max_frame_num_to_track=len(all_frames), 
            reverse=False, 
            propagate_preflight=True
        ):
            # 현재 처리 중인 프레임이 타겟 이미지라면
            if out_idx == target_frame_idx:
                for i, oid in enumerate(out_ids):
                    # 우리가 추적 요청한 ID라면 결과 수집
                    if oid in tracked_obj_ids:
                        # 마스크 추출 (True/False Boolean 배열)
                        pred_mask = (masks[i] > 0.0).cpu().numpy().squeeze()
                        
                        if combined_mask is None:
                            combined_mask = np.zeros_like(pred_mask, dtype=bool)
                        
                        # [마스크 병합] Logical OR 연산 (합집합)
                        # 개별적으로 찾은 객체들을 하나의 마스크로 합칩니다.
                        combined_mask = np.maximum(combined_mask, pred_mask)

        # 결과 리사이징 (원본 크기로 복원)
        final_mask_resized = None
        if combined_mask is not None:
            if combined_mask.ndim == 3:
                combined_mask = combined_mask.squeeze()
            
            dsize = tuple(map(int, target_orig_size))
            # Boolean -> Uint8 (0 or 1) 변환 후 리사이징
            final_mask_resized = cv2.resize(
                combined_mask.astype(np.uint8), 
                dsize, 
                interpolation=cv2.INTER_NEAREST
            )
            
        return final_mask_resized

    def create_overlay(self, original_img_pil, mask_np, color=(255, 0, 0), alpha=0.5):
        """
        원본 이미지 위에 반투명한 마스크를 덧씌워 시각화 이미지를 만듭니다.
        """
        img_np = np.array(original_img_pil)
        overlay_np = img_np.copy()
        
        mask_bool = mask_np > 0
        rgb_color = np.array(color, dtype=np.uint8)
        
        # 마스크 영역에 색상 블렌딩
        overlay_np[mask_bool] = (img_np[mask_bool] * (1 - alpha) + rgb_color * alpha).astype(np.uint8)
        
        return Image.fromarray(overlay_np)

# ==========================================
# 3. 실행부 (Monkey Patch 포함)
# ==========================================
if __name__ == "__main__":
    app = InteractiveBatchLabeler()
    
    # =================================================================
    # [시스템 패치] SAM 3 라이브러리 버그 수정 (Monkey Patch)
    # 설명: SAM 3 내부 코드의 버전 불일치로 인해 발생하는 
    #       'unexpected keyword argument' 에러를 런타임에 수정합니다.
    # =================================================================
    
    # 1. 원래 함수(track_step)를 따로 저장해 둡니다.
    original_track_step = app.predictor.track_step
    
    # 2. 에러를 일으키는 인자들을 걸러내는 '포장지(Wrapper)' 함수를 만듭니다.
    def patched_track_step(*args, **kwargs):
        # 'gt_masks' 인자가 있으면 삭제
        if 'gt_masks' in kwargs: 
            del kwargs['gt_masks']
        # 'frames_to_add_correction_pt' 인자가 있으면 삭제
        if 'frames_to_add_correction_pt' in kwargs: 
            del kwargs['frames_to_add_correction_pt']
            
        # 깨끗해진 인자들로 원래 함수를 호출합니다.
        return original_track_step(*args, **kwargs)
    
    # 3. 모델의 함수를 우리가 만든 패치 함수로 교체합니다.
    app.predictor.track_step = patched_track_step
    
    print(">> [System] Hardcoded Mode & Monkey Patch Applied. (시스템 패치 완료)")
    # =================================================================

    # 앱 실행
    start_time = time.time()
    app.run()
    end_time = time.time()
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")