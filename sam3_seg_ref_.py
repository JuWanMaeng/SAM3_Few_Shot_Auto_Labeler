import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import numpy as np
import torch
import cv2
import os
import glob
from PIL import Image
from sam3.model_builder import build_sam3_video_model

# --- 1. 설정 및 경로 ---
INPUT_SIZE = (1024, 1024)
TEMP_DIR = "./temp_frames"
BASE_OUTPUT_DIR = "./output"
MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)

# [사용자 설정]
defect = 'stitch'
REF_DIR = f"C:/data/Sam3/{defect}/ref"        # 참조 이미지가 있는 폴더
TARGET_FOLDER = f"C:/data/Sam3/{defect}/target" # 타겟 이미지가 있는 폴더

# 지정된 폴더에서 png, jpg, jpeg 파일을 모두 가져와서 정렬
REF_IMAGES = sorted(
    glob.glob(os.path.join(REF_DIR, "*.png")) + 
    glob.glob(os.path.join(REF_DIR, "*.jpg")) + 
    glob.glob(os.path.join(REF_DIR, "*.jpeg"))
)

print(f">> Found {len(REF_IMAGES)} reference images in '{REF_DIR}'")

# --- 2. Matplotlib 박스 선택기 (다중 박스 지원) ---
class MultiBoxSelector:
    def __init__(self, img_arr, title):
        self.img = img_arr
        self.boxes = [] # 여러 박스를 저장할 리스트
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.rs = None

    def select_boxes(self):
        self.ax.imshow(self.img)
        self.ax.set_title(f"{self.title}\n(Drag multiple boxes -> Press 'Q' or 'Enter' to finish)")
        self.ax.axis('off')

        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            
            # 너무 작은 박스는 무시
            if (xmax - xmin) < 5 or (ymax - ymin) < 5: return

            # 박스 저장 (xmin, ymin, xmax, ymax)
            current_box = [xmin, ymin, xmax, ymax]
            self.boxes.append(current_box)
            print(f"  [Added] Box {len(self.boxes)}: {current_box}")
            
            # 시각적 피드백 (빨간 박스 그리기)
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
            
            # 박스 번호 표시
            self.ax.text(xmin, ymin-5, str(len(self.boxes)), color='red', fontsize=12, fontweight='bold')
            self.fig.canvas.draw()

        def on_key(event):
            # Q, Enter, Escape 키를 누르면 종료
            if event.key in ['q', 'Q', 'enter', 'escape']:
                plt.close(self.fig)

        # 키보드 이벤트 연결
        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # RectangleSelector 설정
        self.rs = RectangleSelector(self.ax, on_select, useblit=False, button=[1], 
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        
        plt.show(block=True)
        return self.boxes

# --- 3. 메인 클래스 ---
class InteractiveBatchLabeler:
    def __init__(self):
        print("Loading SAM 3 Model...")
        self.sam3_model = build_sam3_video_model()
        self.predictor = self.sam3_model.tracker
        self.predictor.backbone = self.sam3_model.detector.backbone
        print("Model Loaded.")

        self.ref_frames = []
        self.ref_prompts = [] # 이제 (N, 4) numpy array들이 저장됨

    def prepare_references(self):
        print("\n--- Step 1: Draw Multiple Boxes (Press 'Q' to finish each image) ---")
        for i, path in enumerate(REF_IMAGES):
            if not os.path.exists(path): 
                print(f"Warning: File not found {path}")
                continue
            
            pil_img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            # [변경] 다중 박스 선택기 호출
            selector = MultiBoxSelector(img_arr, title=f"Ref {i+1} / {len(REF_IMAGES)}")
            boxes = selector.select_boxes() # List of [x1, y1, x2, y2]
            
            if not boxes:
                print(f"  -> No boxes selected for Ref {i+1}. Skipping.")
                continue

            self.ref_frames.append(img_arr)
            
            # [변경] 여러 박스를 하나의 Numpy Array (N, 4)로 변환 및 정규화
            np_boxes = np.array(boxes, dtype=np.float32)
            
            # 좌표 정규화 (0~1 사이 값)
            norm_boxes = np_boxes.copy()
            norm_boxes[:, 0] /= INPUT_SIZE[0] # xmin
            norm_boxes[:, 1] /= INPUT_SIZE[1] # ymin
            norm_boxes[:, 2] /= INPUT_SIZE[0] # xmax
            norm_boxes[:, 3] /= INPUT_SIZE[1] # ymax
            
            self.ref_prompts.append(norm_boxes)
            print(f"  -> Saved Ref {i+1}: {len(boxes)} boxes, shape {norm_boxes.shape}")
            
        if not self.ref_frames:
            print("Error: No references prepared.")
            return False
        return True

    def run(self):
        if not self.prepare_references(): return

        target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.jpg")) + glob.glob(os.path.join(TARGET_FOLDER, "*.png")))
        print(f"\n--- Step 2: Processing {len(target_files)} images ---")

        for idx, target_path in enumerate(target_files):
            filename = os.path.basename(target_path)
            base_name = os.path.splitext(filename)[0]
            print(f"[{idx+1}/{len(target_files)}] Processing: {filename}...", end=" ")
            try:
                target_img_pil = Image.open(target_path).convert("RGB")
                final_mask = self.process_single_image(target_img_pil)
                
                if final_mask is not None:
                    # --- A. 마스크 저장 (PNG) ---
                    mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name + "_mask.png")
                    Image.fromarray((final_mask * 255).astype(np.uint8)).save(mask_save_path)
                    
                    # --- B. 오버레이 생성 및 저장 (JPG) ---
                    overlay_img = self.create_overlay(target_img_pil, final_mask, color=(255, 50, 50), alpha=0.6)
                    overlay_save_path = os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg")
                    overlay_img.save(overlay_save_path, quality=95)
                    
                    print(f"-> Success!")
                else:
                    print("-> Failed (No object detected)")

            except Exception as e:
                print(f"Error: {e}")
                import traceback
                traceback.print_exc()

    def process_single_image(self, target_img_pil):
        target_orig_size = target_img_pil.size
        target_img = np.array(target_img_pil.resize(INPUT_SIZE))
        
        all_frames = self.ref_frames + [target_img]
        
        # 1. 임시 프레임 저장
        for i, frame in enumerate(all_frames):
            save_path = os.path.join(TEMP_DIR, f"{i:05d}.jpg")
            Image.fromarray(frame).save(save_path)
            
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        
        # [변경 핵심] 박스마다 고유한 obj_id 부여
        global_obj_id = 1  # ID 카운터 시작
        tracked_obj_ids = [] # 추적 중인 ID 목록 저장

        for i, boxes_np in enumerate(self.ref_prompts):
            # boxes_np: (N, 4) -> 해당 프레임에 그려진 N개의 박스들
            for box in boxes_np:
                # box: [x1, y1, x2, y2] (shape: (4,))
                
                # 이번에는 'box' 인자를 그대로 사용합니다.
                # 왜냐하면 루프를 돌면서 한 번에 '하나의 박스'만 넣기 때문입니다.
                # SAM API는 box 인자에 (4,) 배열이 들어오면 (1, 2, 2)로 내부 변환하여 잘 처리합니다.
                
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=global_obj_id, # 각 박스마다 다른 ID 부여 (1, 2, 3...)
                    box=box,              # 개별 박스 좌표
                    clear_old_points=True 
                )
                
                tracked_obj_ids.append(global_obj_id)
                global_obj_id += 1 # 다음 박스를 위해 ID 증가
            
        print(f"  -> Tracking {len(tracked_obj_ids)} separate objects...")

        combined_mask = None
        target_frame_idx = len(self.ref_frames) 
        
        # 2. 추론 실행
        for out_idx, out_ids, _, masks, _ in self.predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=0, 
            max_frame_num_to_track=len(all_frames), 
            reverse=False, 
            propagate_preflight=True
        ):
            # 타겟 프레임에 도달했을 때
            if out_idx == target_frame_idx:
                # out_ids에는 감지된 모든 객체의 ID가 들어있음
                for i, oid in enumerate(out_ids):
                    if oid in tracked_obj_ids:
                        # 개별 마스크 추출 (True/False)
                        pred_mask = (masks[i] > 0.0).cpu().numpy().squeeze()
                        
                        # [병합 로직] 개별 마스크들을 하나의 큰 마스크로 합침 (Logical OR)
                        if combined_mask is None:
                            combined_mask = np.zeros_like(pred_mask, dtype=bool)
                        
                        # 기존 마스크와 현재 마스크를 합침 (둘 중 하나라도 True면 True)
                        combined_mask = np.maximum(combined_mask, pred_mask)

        # 3. 결과 리사이징 및 반환
        final_mask_resized = None
        if combined_mask is not None:
            if combined_mask.ndim == 3:
                combined_mask = combined_mask.squeeze()
                
            dsize = tuple(map(int, target_orig_size))
            
            # Boolean -> Uint8 (0 or 255) 변환 후 리사이징
            final_mask_resized = cv2.resize(
                combined_mask.astype(np.uint8), 
                dsize, 
                interpolation=cv2.INTER_NEAREST
            )
            
        return final_mask_resized

    def create_overlay(self, original_img_pil, mask_np, color=(255, 0, 0), alpha=0.5):
        img_np = np.array(original_img_pil)
        overlay_np = img_np.copy()
        mask_bool = mask_np > 0
        rgb_color = np.array(color, dtype=np.uint8)
        overlay_np[mask_bool] = (img_np[mask_bool] * (1 - alpha) + rgb_color * alpha).astype(np.uint8)
        return Image.fromarray(overlay_np)

if __name__ == "__main__":
    app = InteractiveBatchLabeler()
    
    # =================================================================
    # [BUG FIX] SAM 3 Library Monkey Patch (Updated)
    # 설명: Sam3TrackerBase.track_step()이 불필요한 인자들을 받아
    #       에러가 나는 현상을 런타임에 수정합니다.
    # =================================================================
    
    original_track_step = app.predictor.track_step
    
    def patched_track_step(*args, **kwargs):
        # 1. gt_masks 제거
        if 'gt_masks' in kwargs:
            del kwargs['gt_masks']
            
        # 2. frames_to_add_correction_pt 제거 (새로 추가된 문제 해결)
        if 'frames_to_add_correction_pt' in kwargs:
            del kwargs['frames_to_add_correction_pt']
            
        # 원본 함수 호출
        return original_track_step(*args, **kwargs)
    
    app.predictor.track_step = patched_track_step
    
    print(">> [System] Applied fix for 'gt_masks' and 'frames_to_add_correction_pt'.")
    # =================================================================

    app.run()