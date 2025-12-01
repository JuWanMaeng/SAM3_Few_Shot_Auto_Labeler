import matplotlib
# Matplotlib 백엔드 설정 (창이 팝업되도록 강제)
try:
    matplotlib.use('TkAgg') 
except:
    pass

import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import numpy as np
import torch
import cv2
import os
import glob
import shutil, time
from PIL import Image
from sam3.model_builder import build_sam3_video_model

# ==========================================
# 1. 환경 설정
# ==========================================
INPUT_SIZE = (1008, 1008)  # SAM 표준 입력 크기
TEMP_DIR = "./temp_frames"
BASE_OUTPUT_DIR = "./output"
MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")

# 폴더 초기화 (깨끗한 시작)
if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
if os.path.exists(BASE_OUTPUT_DIR): shutil.rmtree(BASE_OUTPUT_DIR)

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)

# [사용자 데이터 경로]
REF_IMAGES = [
    "C:/data/Sam3/stitch/ref/2_crop26_rot88.jpg", 
    "C:/data/Sam3/stitch/ref/2_crop26_rot220.jpg", 
    "C:/data/Sam3/stitch/ref/2_crop26_rot261.jpg", 
]
TARGET_FOLDER = "C:/data/Sam3/stitch/target"

# ==========================================
# 2. 박스 선택 도구 (Fix Applied)
# ==========================================
class MultiBoxSelector:
    def __init__(self, img_arr, title):
        self.img = img_arr
        self.boxes = []
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        self.rs = None # [중요] 가비지 컬렉션 방지용

    def select_boxes(self):
        self.ax.imshow(self.img)
        self.ax.set_title(f"{self.title}\n(Drag mouse to draw box -> Press 'Q' or 'Enter' to Finish)")
        self.ax.axis('off')

        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            
            if (xmax - xmin) < 5 or (ymax - ymin) < 5: return # 너무 작은 박스 무시

            current_box = [xmin, ymin, xmax, ymax]
            self.boxes.append(current_box)
            print(f"  [Added] Box {len(self.boxes)}: {current_box}")
            
            # 시각적 피드백
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
            self.ax.text(xmin, ymin-5, str(len(self.boxes)), color='red', fontsize=12, fontweight='bold')
            self.fig.canvas.draw()

        def on_key(event):
            if event.key in ['q', 'Q', 'enter', 'escape']:
                plt.close(self.fig)

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        
        # [중요] 객체 유지
        self.rs = RectangleSelector(
            self.ax, on_select, useblit=False, button=[1], 
            minspanx=5, minspany=5, spancoords='pixels', interactive=True
        )
        
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
            if not os.path.exists(path): 
                print(f"Warning: File not found {path}")
                continue
            
            pil_img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            selector = MultiBoxSelector(img_arr, title=f"Ref {i+1} / {len(REF_IMAGES)}")
            boxes = selector.select_boxes()
            
            if not boxes:
                print(f"  -> No boxes selected for Ref {i+1}. Skipping.")
                continue

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
        if not self.prepare_references(): return
        target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.jpg")) + glob.glob(os.path.join(TARGET_FOLDER, "*.png")))
        print(f"\n--- Step 2: Processing {len(target_files)} images (Fast Swapping Mode) ---")
        self.run_fast_inference(target_files)

    def run_fast_inference(self, target_files):
        # 1. 초기화
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        for i, frame in enumerate(self.ref_frames):
            Image.fromarray(frame).save(os.path.join(TEMP_DIR, f"{i:05d}.jpg"))
            
        dummy_idx = len(self.ref_frames)
        Image.fromarray(np.zeros_like(self.ref_frames[0])).save(os.path.join(TEMP_DIR, f"{dummy_idx:05d}.jpg"))
        
        print(f"\n[Initializing Model State...]")
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        
        # 실제 텐서 크기 확인 (1024 vs 1008 문제 해결용)
        ref_tensor = inference_state["images"][0] 
        tensor_h, tensor_w = ref_tensor.shape[-2:] 
        print(f"  -> Actual Tensor Size in Memory: {tensor_w}x{tensor_h} (Fixed)")
        
        # 2. Reference 학습
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
                
        for _ in self.predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=0, 
            max_frame_num_to_track=dummy_idx, 
            reverse=False, propagate_preflight=True
        ): pass
        
        print("  -> References Locked. Starting High-Speed Inference.")

        # 3. 고속 추론 루프
        device = inference_state["device"]
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        # =================================================================
        # [🔥 핵심 수정] 추론 모드 컨텍스트를 열어서 텐서 수정을 허용합니다.
        # =================================================================
        with torch.inference_mode():
            for t_idx, t_path in enumerate(target_files):
                filename = os.path.basename(t_path)
                base_name = os.path.splitext(filename)[0]
                print(f"  [{t_idx+1}/{len(target_files)}] Processing: {filename}", end="\r")
                
                try:
                    img_pil = Image.open(t_path).convert("RGB")
                    target_orig_size = img_pil.size
                    
                    # 텐서 크기에 맞춰 리사이징
                    img_resized = img_pil.resize((tensor_w, tensor_h))
                    img_np = np.array(img_resized)
                    
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().to(device) / 255.0
                    img_tensor = (img_tensor - pixel_mean) / pixel_std
                    
                    if img_tensor.dim() == 3:
                        img_tensor = img_tensor.unsqueeze(0) 
                    
                    # Swapping (이제 에러 안 남)
                    inference_state["images"][dummy_idx] = img_tensor
                    
                    if dummy_idx in inference_state["cached_features"]:
                        del inference_state["cached_features"][dummy_idx]

                    # Inference
                    combined_mask = None
                    
                    for oid in tracked_ids:
                        obj_idx = self.predictor._obj_id_to_idx(inference_state, oid)
                        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                        
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
                        
                        pred_mask = current_out["pred_masks"]
                        if pred_mask is not None:
                            mask_bool = (pred_mask > 0.0).cpu().numpy().squeeze()
                            if combined_mask is None: combined_mask = np.zeros_like(mask_bool, dtype=bool)
                            combined_mask = np.maximum(combined_mask, mask_bool)

                    if combined_mask is not None:
                        if combined_mask.ndim == 3: combined_mask = combined_mask.squeeze()
                        
                        final_mask_resized = cv2.resize(
                            combined_mask.astype(np.uint8), 
                            target_orig_size, 
                            interpolation=cv2.INTER_NEAREST
                        )
                        
                        mask_save_path = os.path.join(MASK_OUTPUT_DIR, base_name + "_mask.png")
                        Image.fromarray((final_mask_resized * 255).astype(np.uint8)).save(mask_save_path)
                        
                        overlay_img = self.create_overlay(img_pil, final_mask_resized, color=(255, 50, 50), alpha=0.6)
                        overlay_save_path = os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg")
                        overlay_img.save(overlay_save_path, quality=95)

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                
        print("\n\nAll Done.")

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
    # [System Patch] Fix SAM 3 Library Bugs
    # =================================================================
    original_track_step = app.predictor.track_step
    
    def patched_track_step(*args, **kwargs):
        if 'gt_masks' in kwargs: del kwargs['gt_masks']
        if 'frames_to_add_correction_pt' in kwargs: del kwargs['frames_to_add_correction_pt']
        return original_track_step(*args, **kwargs)
    
    app.predictor.track_step = patched_track_step
    print(">> [System] Monkey Patch Applied.")
    
    # 앱 실행
    start_time = time.time()
    app.run()
    end_time = time.time()
    print(f"Total Processing Time: {end_time - start_time:.2f} seconds")