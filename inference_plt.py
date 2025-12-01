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
REF_IMAGES = [
    "C:/data/Sam3/stitch/2_crop26_rot88.jpg", 
    "C:/data/Sam3/stitch/2_crop26_rot220.jpg", 
    "C:/data/Sam3/stitch/2_crop26_rot261.jpg", 
    "C:/data/Sam3/stitch/2_crop26_rot298.jpg", 
    "C:/data/Sam3/stitch/2_crop26_rot340.jpg"
]
TARGET_FOLDER = "C:/data/Sam3/stitch/target"

# --- 2. Matplotlib 박스 선택기 ---
class BoxSelector:
    def __init__(self, img_arr, title):
        self.img = img_arr
        self.box = None
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
    def select_box(self):
        self.ax.imshow(self.img)
        self.ax.set_title(f"{self.title}\n(Drag mouse -> Close window)")
        self.ax.axis('off')

        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            
            if (xmax - xmin) < 5 or (ymax - ymin) < 5: return

            self.box = [xmin, ymin, xmax, ymax]
            print(f"  Selected: {self.box}")
            
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor='r', facecolor='none')
            self.ax.add_patch(rect)
            self.fig.canvas.draw()

        rs = RectangleSelector(self.ax, on_select, useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        plt.show(block=True)
        return self.box

# --- 3. 메인 클래스 ---
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
        print("\n--- Step 1: Draw Boxes ---")
        for i, path in enumerate(REF_IMAGES):
            if not os.path.exists(path): continue
            
            pil_img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            selector = BoxSelector(img_arr, title=f"Ref {i+1}")
            box = selector.select_box() # [x1, y1, x2, y2]
            
            if box is None: return False

            self.ref_frames.append(img_arr)
            
            # [수정] 하드코딩 방식과 100% 동일한 구조로 생성
            # box: [xmin, ymin, xmax, ymax]
            # rel_box: numpy array (1, 4) float32
            
            rel_box = np.array([[
                box[0] / INPUT_SIZE[0], 
                box[1] / INPUT_SIZE[1],
                box[2] / INPUT_SIZE[0], 
                box[3] / INPUT_SIZE[1]
            ]], dtype=np.float32)
            
            self.ref_prompts.append(rel_box)
            print(f"  -> Saved Ref {i+1} as shape {rel_box.shape}")
            
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
                    # 색상: (R, G, B), 투명도: alpha (0.0 ~ 1.0)
                    overlay_img = self.create_overlay(target_img_pil, final_mask, color=(255, 50, 50), alpha=0.6)
                    overlay_save_path = os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg")
                    overlay_img.save(overlay_save_path, quality=95)
                    
                    print(f"-> Success! Saved mask & overlay.")
                else:
                    print("-> Failed (No object detected)")

            except Exception as e:
                print(f"Error: {e}")

    def process_single_image(self, target_img_pil):
        target_orig_size = target_img_pil.size
        target_img = np.array(target_img_pil.resize(INPUT_SIZE))
        
        all_frames = self.ref_frames + [target_img]
        for i, frame in enumerate(all_frames):
            Image.fromarray(frame).save(os.path.join(TEMP_DIR, f"{i:05d}.jpg"))
            
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        # self.predictor.clear_all_points_in_video(inference_state)
        
        obj_id = 1
        
        # [수정] 아무런 변환 없이 numpy array 그대로 주입 (하드코딩 예제와 동일)
        for i, box_np in enumerate(self.ref_prompts):
            self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=i,
                obj_id=obj_id,
                box=box_np # (1, 4) numpy array
            )
            
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
    app = InteractiveBatchLabeler()
    app.run()