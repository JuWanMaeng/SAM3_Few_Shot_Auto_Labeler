import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import numpy as np
import torch
import cv2
import os
import glob
import shutil
import time
from PIL import Image
from sam3.model_builder import build_sam3_video_model

# ==========================================
# 1. 환경 설정
# ==========================================
INPUT_SIZE = (1024, 1024)
TEMP_DIR = "./temp_frames"
BASE_OUTPUT_DIR = "./output"

MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")
BBOX_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "bboxes")

if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
if os.path.exists(BASE_OUTPUT_DIR): shutil.rmtree(BASE_OUTPUT_DIR)

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)
os.makedirs(BBOX_OUTPUT_DIR, exist_ok=True)

# [사용자 데이터 경로]
defect = 'stitch'
REF_DIR = f"C:/data/Sam3/{defect}/ref"
TARGET_FOLDER = f"C:/data/Sam3/{defect}/target"

# 참조 이미지 로드
REF_IMAGES = sorted(
    glob.glob(os.path.join(REF_DIR, "*.png")) + 
    glob.glob(os.path.join(REF_DIR, "*.jpg")) + 
    glob.glob(os.path.join(REF_DIR, "*.jpeg"))
)
print(f">> Found {len(REF_IMAGES)} reference images.")

# ==========================================
# 2. 박스 선택 도구
# ==========================================
class MultiBoxSelector:
    def __init__(self, img_arr, title):
        self.img = img_arr
        self.boxes = []
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.rs = None

    def select_boxes(self):
        self.ax.imshow(self.img)
        self.ax.set_title(f"{self.title}\n(Drag mouse to draw box -> Press 'Q' or 'Enter' to Finish)")
        self.ax.axis('off')

        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            
            if (xmax - xmin) < 5 or (ymax - ymin) < 5: return

            current_box = [xmin, ymin, xmax, ymax]
            self.boxes.append(current_box)
            print(f"  [Added] Box {len(self.boxes)}: {current_box}")
            
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
            self.ax.text(xmin, ymin-5, str(len(self.boxes)), color='red', fontsize=12, fontweight='bold')
            self.fig.canvas.draw()

        def on_key(event):
            if event.key in ['q', 'Q', 'enter', 'escape']:
                plt.close(self.fig)

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        self.rs = RectangleSelector(self.ax, on_select, useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        plt.show(block=True)
        return self.boxes

# ==========================================
# 3. 메인 프로세서 (Fast Inference Logic)
# ==========================================
class InteractiveBatchLabeler:
    def __init__(self):
        print("Loading SAM 3 Model...")
        self.sam3_model = build_sam3_video_model()
        self.predictor = self.sam3_model.tracker
        self.predictor.backbone = self.sam3_model.detector.backbone
        print("Model Loaded.")

        self.ref_frames = []
        self.ref_prompts = [] 

    def prepare_references(self):
        print("\n--- Step 1: Draw Reference Boxes ---")
        for i, path in enumerate(REF_IMAGES):
            if not os.path.exists(path): continue
            
            pil_img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            selector = MultiBoxSelector(img_arr, title=f"Ref {i+1} / {len(REF_IMAGES)}")
            boxes = selector.select_boxes()
            
            if not boxes: continue

            self.ref_frames.append(img_arr)
            
            np_boxes = np.array(boxes, dtype=np.float32)
            norm_boxes = np_boxes.copy()
            norm_boxes[:, 0] /= INPUT_SIZE[0]
            norm_boxes[:, 1] /= INPUT_SIZE[1]
            norm_boxes[:, 2] /= INPUT_SIZE[0]
            norm_boxes[:, 3] /= INPUT_SIZE[1]
            
            self.ref_prompts.append(norm_boxes)
            print(f"  -> Saved Ref {i+1}: {len(boxes)} boxes")
            
        if not self.ref_frames:
            print("Error: No references prepared.")
            return False
        return True

    def run(self):
        # 1. Reference 준비
        if not self.prepare_references(): return

        # 2. Target 파일 리스트 확보
        target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.jpg")) + glob.glob(os.path.join(TARGET_FOLDER, "*.png")))
        print(f"\n--- Step 2: Processing {len(target_files)} images (Fast Swapping Mode) ---")

        # 3. 고속 추론 실행 (리스트 전체 전달)
        self.run_fast_inference(target_files)

    def run_fast_inference(self, target_files):
        # -------------------------------------------------------------
        # [초기화] Ref Images + 1 Dummy Image
        # -------------------------------------------------------------
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        for i, frame in enumerate(self.ref_frames):
            Image.fromarray(frame).save(os.path.join(TEMP_DIR, f"{i:05d}.jpg"))
            
        dummy_idx = len(self.ref_frames)
        Image.fromarray(np.zeros_like(self.ref_frames[0])).save(os.path.join(TEMP_DIR, f"{dummy_idx:05d}.jpg"))
        
        print(f"\n[Initializing Model State...]")
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        
        # [중요] 실제 텐서 크기 확인 (1024 vs 1008 문제 해결)
        ref_tensor = inference_state["images"][0] 
        tensor_h, tensor_w = ref_tensor.shape[-2:] 
        print(f"  -> Actual Tensor Size in Memory: {tensor_w}x{tensor_h} (Fixed)")
        
        # -------------------------------------------------------------
        # [Step 1] Reference 학습 (Memory Encoding) - 1회 수행
        # -------------------------------------------------------------
        print("  -> Encoding References...")
        global_obj_id = 1
        tracked_ids = []
        
        for i, boxes_np in enumerate(self.ref_prompts):
            for box in boxes_np:
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=i,
                    obj_id=global_obj_id,
                    box=box,
                    clear_old_points=True
                )
                tracked_ids.append(global_obj_id)
                global_obj_id += 1
                
        # Ref만 기억시킴 (Dummy 직전까지만 Propagate)
        for _ in self.predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=0, 
            max_frame_num_to_track=dummy_idx, 
            reverse=False, propagate_preflight=True
        ): pass
        
        print("  -> References Locked. Starting High-Speed Inference Loop.")

        # -------------------------------------------------------------
        # [Step 2] 고속 추론 루프 (Image Swapping)
        # -------------------------------------------------------------
        device = inference_state["device"]
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        # [중요] Inference Mode 안에서 실행 (In-place update 에러 방지)
        with torch.inference_mode():
            for t_idx, t_path in enumerate(target_files):
                filename = os.path.basename(t_path)
                base_name = os.path.splitext(filename)[0]
                print(f"  [{t_idx+1}/{len(target_files)}] Processing: {filename}", end="\r")
                
                try:
                    # A. 로드
                    img_pil = Image.open(t_path).convert("RGB")
                    target_orig_size = img_pil.size
                    
                    # B. 리사이징 (모델 내부 실제 크기에 맞춤)
                    img_resized = img_pil.resize((tensor_w, tensor_h))
                    img_np = np.array(img_resized)
                    
                    # C. 텐서 변환
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().to(device) / 255.0
                    img_tensor = (img_tensor - pixel_mean) / pixel_std
                    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0) 
                    
                    # D. Swapping: 더미에 target 이미지 삽입
                    inference_state["images"][dummy_idx] = img_tensor
                    
                    # 캐시 삭제 (필수)
                    if dummy_idx in inference_state["cached_features"]:
                        del inference_state["cached_features"][dummy_idx]

                    # E. 추론 (Direct Inference)
                    combined_mask = None
                    
                    for oid in tracked_ids:
                        obj_idx = self.predictor._obj_id_to_idx(inference_state, oid)
                        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                        
                        # run_mem_encoder=False -> 결과만 받고 기억 안 함 (속도 향상)
                        current_out, _ = self.predictor._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=dummy_idx,
                            batch_size=1,
                            is_init_cond_frame=False,
                            point_inputs=None,
                            mask_inputs=None,
                            reverse=False,
                            run_mem_encoder=False, 
                        )
                        
                        # 결과 병합
                        pred_mask = current_out["pred_masks"]
                        if pred_mask is not None:
                            mask_bool = (pred_mask > 0.0).cpu().numpy().squeeze()
                            if combined_mask is None: combined_mask = np.zeros_like(mask_bool, dtype=bool)
                            combined_mask = np.maximum(combined_mask, mask_bool)
                        
                        # [메모리 누수 방지] 결과 기록 삭제
                        if dummy_idx in obj_output_dict["non_cond_frame_outputs"]:
                            del obj_output_dict["non_cond_frame_outputs"][dummy_idx]

                    # F. 결과 저장
                    if combined_mask is not None:
                        if combined_mask.ndim == 3: combined_mask = combined_mask.squeeze()
                        
                        # 원본 크기로 복원
                        final_mask_resized = cv2.resize(
                            combined_mask.astype(np.uint8), 
                            target_orig_size, 
                            interpolation=cv2.INTER_NEAREST
                        )
                        
                        # 1. 마스크 저장
                        mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name + "_mask.png")
                        Image.fromarray((final_mask_resized * 255).astype(np.uint8)).save(mask_save_path)
                        
                        # 2. Overlay 저장
                        overlay_img = self.create_mask_overlay(img_pil, final_mask_resized, color=(255, 50, 50), alpha=0.6)
                        overlay_img.save(os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg"), quality=95)

                        # 3. BBox 저장
                        bbox_img = self.create_bbox_overlay(img_pil, final_mask_resized, color=(0, 255, 0), thickness=3)
                        bbox_img.save(os.path.join(BBOX_OUTPUT_DIR, base_name + "_bbox.jpg"), quality=95)

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                
                # # 주기적 메모리 청소 (옵션)
                # if (t_idx + 1) % 50 == 0:
                #     import gc
                #     gc.collect()
                #     torch.cuda.empty_cache()
                
        print("\n\nAll Done.")

    # --- 유틸리티 함수들 ---
    def create_mask_overlay(self, original_img_pil, mask_np, color=(255, 0, 0), alpha=0.5):
        img_np = np.array(original_img_pil)
        overlay_np = img_np.copy()
        mask_bool = mask_np > 0
        rgb_color = np.array(color, dtype=np.uint8)
        overlay_np[mask_bool] = (img_np[mask_bool] * (1 - alpha) + rgb_color * alpha).astype(np.uint8)
        return Image.fromarray(overlay_np)

    def create_bbox_overlay(self, original_img_pil, mask_np, color=(0, 255, 0), thickness=3):
        img_np = np.array(original_img_pil)
        vis_img = img_np.copy()
        mask_u8 = (mask_np > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 10: continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, thickness)
        return Image.fromarray(vis_img)

if __name__ == "__main__":
    app = InteractiveBatchLabeler()
    
    # [Monkey Patch]
    original_track_step = app.predictor.track_step
    def patched_track_step(*args, **kwargs):
        if 'gt_masks' in kwargs: del kwargs['gt_masks']
        if 'frames_to_add_correction_pt' in kwargs: del kwargs['frames_to_add_correction_pt']
        return original_track_step(*args, **kwargs)
    app.predictor.track_step = patched_track_step
    
    print(">> [System] Optimized Mode & Patches Applied.")
    
    # 실행
    start_time = time.time()
    app.run()
    print(f"Total Time: {time.time() - start_time:.2f}s")