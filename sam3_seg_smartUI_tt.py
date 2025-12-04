import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import cv2
import os
import glob
import shutil
import time
import gc
from PIL import Image
from sam3.model_builder import build_sam3_video_model

# ==========================================
# 1. 환경 설정
# ==========================================
INPUT_SIZE = (1024, 1024)
TEMP_DIR = "./temp_frames"
BASE_OUTPUT_DIR = "./output"

MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
ID_MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "id_masks")
OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")
BBOX_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "bboxes")

if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
if os.path.exists(BASE_OUTPUT_DIR): shutil.rmtree(BASE_OUTPUT_DIR)

os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(MASK_OUTPUT_DIR, exist_ok=True)
os.makedirs(OVERLAY_OUTPUT_DIR, exist_ok=True)
os.makedirs(BBOX_OUTPUT_DIR, exist_ok=True)
os.makedirs(ID_MASK_OUTPUT_DIR, exist_ok=True)

# [사용자 데이터 경로]
defect = 'particles'
REF_DIR = f"C:/data/Sam3/{defect}/ref"
TARGET_FOLDER = f"C:/data/Sam3/{defect}/target"

# 참조 이미지 로드
REF_IMAGES = sorted(
    glob.glob(os.path.join(REF_DIR, "*.png")) + 
    glob.glob(os.path.join(REF_DIR, "*.jpg")) + 
    glob.glob(os.path.join(REF_DIR, "*.bmp"))
)
print(f">> Found {len(REF_IMAGES)} reference images.")

# 시각화용 색상 팔레트
np.random.seed(42)
COLORS = np.random.randint(0, 255, (255, 3), dtype=np.uint8)
MPL_COLORS = plt.cm.tab10(np.linspace(0, 1, 10))

# ==========================================
# 2. 스마트 대시보드 (썸네일 4x2 배열 + 8개 ID)
# ==========================================
class SmartBoxSelector:
    def __init__(self, img_arr, title, global_thumbnails, global_counts):
        self.img = img_arr
        self.box_data = [] 
        self.title = title
        
        # 부모로부터 공유받은 데이터
        self.global_thumbnails = global_thumbnails 
        self.global_counts = global_counts 
        
        self.fig = plt.figure(figsize=(12, 8)) 
        
        # 4행 3열 그리드 (0열: 메인, 1~2열: 썸네일)
        gs = gridspec.GridSpec(4, 3, width_ratios=[4, 1, 1], figure=self.fig)
        
        # 메인 화면
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_main.imshow(self.img)
        self.ax_main.axis('off')
        
        # 사이드바 (ID 1~8)
        self.ax_previews = []
        for i in range(8):
            row = i // 2      
            col = (i % 2) + 1 
            
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            self.ax_previews.append(ax)
            self.refresh_sidebar_slot(i + 1)
            
        self.current_obj_id = 1
        self.rs = None
        self.update_title()

    def update_title(self):
        self.ax_main.set_title(
            f"{self.title}\n"
            f"Current ID: {self.current_obj_id} (Press '1'-'8' or 'N')\n"
            f"[Left Drag]: Draw / [Right Click on Box]: Delete / [Q]: Finish"
        )
        self.fig.canvas.draw()

    def refresh_sidebar_slot(self, obj_id):
        if 1 <= obj_id <= 8:
            ax = self.ax_previews[obj_id - 1]
            ax.clear()
            ax.set_xticks([])
            ax.set_yticks([])
            
            color = MPL_COLORS[(obj_id - 1) % 10]
            count = self.global_counts.get(obj_id, 0)
            
            if obj_id in self.global_thumbnails:
                ax.imshow(self.global_thumbnails[obj_id])
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                title_text = f"ID: {obj_id}"
            else:
                title_text = f"ID: {obj_id}"
                ax.text(0.5, 0.5, "Empty", ha='center', va='center', fontsize=10, color='gray')
            
            ax.set_title(title_text, fontsize=9, color=color, fontweight='bold', pad=2)
            ax.text(0.5, -0.1, f"Count: {count}", ha='center', va='top', transform=ax.transAxes, fontsize=9, fontweight='bold')

    def select_boxes(self):
        def on_select(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            xmin, xmax = sorted([x1, x2])
            ymin, ymax = sorted([y1, y2])
            
            if (xmax - xmin) < 5 or (ymax - ymin) < 5: return

            self.global_counts[self.current_obj_id] = self.global_counts.get(self.current_obj_id, 0) + 1
            print(f"  [ID: {self.current_obj_id}] Box Added (Total: {self.global_counts[self.current_obj_id]})")
            
            color = MPL_COLORS[(self.current_obj_id - 1) % 10]
            rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor=color, facecolor='none')
            self.ax_main.add_patch(rect)
            
            text_obj = self.ax_main.text(xmin, ymin-5, f"ID:{self.current_obj_id}", color=color, fontsize=12, fontweight='bold', backgroundcolor='white')
            
            current_box = [xmin, ymin, xmax, ymax]
            self.box_data.append({
                'box': current_box, 
                'id': self.current_obj_id,
                'rect': rect,
                'text': text_obj
            })

            h, w = self.img.shape[:2]
            bw, bh = xmax - xmin, ymax - ymin
            cx, cy = xmin + bw / 2, ymin + bh / 2
            
            scale = 1.5
            new_bw, new_bh = bw * scale, bh * scale
            exp_xmin, exp_ymin = int(cx - new_bw/2), int(cy - new_bh/2)
            exp_xmax, exp_ymax = int(cx + new_bw/2), int(cy + new_bh/2)
            
            safe_xmin = max(0, exp_xmin)
            safe_ymin = max(0, exp_ymin)
            safe_xmax = min(w, exp_xmax)
            safe_ymax = min(h, exp_ymax)
            
            crop_img = self.img[safe_ymin:safe_ymax, safe_xmin:safe_xmax]
            self.global_thumbnails[self.current_obj_id] = crop_img
            
            self.refresh_sidebar_slot(self.current_obj_id)
            self.fig.canvas.draw()

        def on_mouse_click(event):
            if event.button == 3 and event.inaxes == self.ax_main:
                click_x, click_y = event.xdata, event.ydata
                for i in range(len(self.box_data) - 1, -1, -1):
                    item = self.box_data[i]
                    bx1, by1, bx2, by2 = item['box']
                    if bx1 <= click_x <= bx2 and by1 <= click_y <= by2:
                        deleted_id = item['id']
                        item['rect'].remove()
                        item['text'].remove()
                        self.box_data.pop(i)
                        if self.global_counts[deleted_id] > 0:
                            self.global_counts[deleted_id] -= 1
                        print(f"  [ID: {deleted_id}] Box Deleted (Total: {self.global_counts[deleted_id]})")
                        self.refresh_sidebar_slot(deleted_id)
                        self.fig.canvas.draw()
                        return

        def on_key(event):
            if event.key in ['q', 'Q', 'enter', 'escape']:
                plt.close(self.fig)
            elif event.key in ['n', 'N']:
                self.current_obj_id = (self.current_obj_id % 8) + 1
                self.update_title()
            elif event.key in [str(i) for i in range(1, 9)]:
                self.current_obj_id = int(event.key)
                self.update_title()

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        self.fig.canvas.mpl_connect('button_press_event', on_mouse_click)
        self.rs = RectangleSelector(self.ax_main, on_select, useblit=False, button=[1], minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        
        plt.tight_layout()
        plt.show(block=True)
        
        clean_data = []
        for item in self.box_data:
            clean_data.append({'box': item['box'], 'id': item['id']})
            
        return clean_data

# ==========================================
# 3. 메인 프로세서
# ==========================================
class InteractiveBatchLabeler:
    def __init__(self):
        print("Loading SAM 3 Model...")
        self.sam3_model = build_sam3_video_model()
        self.predictor = self.sam3_model.tracker
        self.predictor.backbone = self.sam3_model.detector.backbone
        print("Model Loaded.")

        print(f"[System] Default Max Cond Frames: {self.predictor.max_cond_frames_in_attn}")
        self.predictor.max_cond_frames_in_attn = -1 
        print(f"[System] Updated Max Cond Frames: {self.predictor.max_cond_frames_in_attn} (Unlimited)")

        self.ref_frames = []
        self.ref_prompts = []
        self.id_thumbnails = {}
        self.id_counts = {}

    def prepare_references(self):
        print("\n--- Step 1: Draw Reference Boxes (Count Tracking) ---")
        
        for i, path in enumerate(REF_IMAGES):
            if not os.path.exists(path): continue
            
            pil_img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            selector = SmartBoxSelector(
                img_arr, 
                title=f"Ref {i+1} / {len(REF_IMAGES)}", 
                global_thumbnails=self.id_thumbnails,
                global_counts=self.id_counts
            )
            box_data_list = selector.select_boxes()
            
            if not box_data_list: continue

            self.ref_frames.append(img_arr)
            current_frame_prompts = []
            
            for item in box_data_list:
                box = item['box']
                obj_id = item['id']
                norm_box = np.array(box, dtype=np.float32)
                norm_box[0] /= INPUT_SIZE[0]
                norm_box[1] /= INPUT_SIZE[1]
                norm_box[2] /= INPUT_SIZE[0]
                norm_box[3] /= INPUT_SIZE[1]
                
                current_frame_prompts.append({'id': obj_id, 'box': norm_box})
            
            self.ref_prompts.append(current_frame_prompts)
            print(f"  -> Saved Ref {i+1}: {len(box_data_list)} boxes")
            
        if not self.ref_frames: return False
        return True

    def run(self):
        if not self.prepare_references(): return
        target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.jpg")) + glob.glob(os.path.join(TARGET_FOLDER, "*.png")) + glob.glob(os.path.join(TARGET_FOLDER, "*.bmp")))
        print(f"\n--- Step 2: Processing {len(target_files)} images ---")
        self.run_fast_inference(target_files)

    def run_fast_inference(self, target_files):
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        for i, frame in enumerate(self.ref_frames):
            Image.fromarray(frame).save(os.path.join(TEMP_DIR, f"{i:05d}.jpg"))
            
        dummy_idx = len(self.ref_frames)
        Image.fromarray(np.zeros_like(self.ref_frames[0])).save(os.path.join(TEMP_DIR, f"{dummy_idx:05d}.jpg"))
        
        print(f"\n[Initializing Model State...]")
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        
        ref_tensor = inference_state["images"][0] 
        tensor_h, tensor_w = ref_tensor.shape[-2:] 
        print(f"  -> Actual Tensor Size in Memory: {tensor_w}x{tensor_h} (Fixed)")
        
        print("  -> Encoding References...")
        tracked_ids = []
        for frame_idx, prompts in enumerate(self.ref_prompts):
            for item in prompts:
                obj_id = item['id']
                box = item['box']
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=box,
                    clear_old_points=True
                )
                if obj_id not in tracked_ids:
                    tracked_ids.append(obj_id)

        for _ in self.predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=0, 
            max_frame_num_to_track=dummy_idx, 
            reverse=False, propagate_preflight=True
        ): pass
        
        print(f"  -> References Locked. Tracking IDs: {tracked_ids}")

        device = inference_state["device"]
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

        with torch.inference_mode():
            total_time = 0
            cnt = 0
            for t_idx, t_path in enumerate(target_files):
                start_time = time.time()
                filename = os.path.basename(t_path)
                base_name = os.path.splitext(filename)[0]
                print(f"  [{t_idx+1}/{len(target_files)}] Processing: {filename}", end="\r")
                
                try:
                    img_pil = Image.open(t_path).convert("RGB")
                    target_orig_size = img_pil.size
                    
                    img_resized = img_pil.resize((tensor_w, tensor_h))
                    img_np = np.array(img_resized)
                    
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().to(device) / 255.0
                    img_tensor = (img_tensor - pixel_mean) / pixel_std
                    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0) 
                    
                    inference_state["images"][dummy_idx] = img_tensor
                    if dummy_idx in inference_state["cached_features"]:
                        del inference_state["cached_features"][dummy_idx]

                    detected_objects = [] 
                    combined_mask = None
                    
                    for oid in tracked_ids:
                        if oid >= 9: continue # ID 범위 체크 (1~8)
                        
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
            
                        pred_masks = current_out["pred_masks"]
                        obj_score = current_out["object_score_logits"]

                        # [변경] Threshold 조정 (0.0으로 설정하면 확률 50% 이상)
                        # 디버깅을 위해 조금 더 낮게(-2.0) 보고 싶다면 여기서 수정
                        if obj_score < -2: continue 

                        # [NEW] Logit -> Probability 변환
                        prob_score = 1.0 / (1.0 + np.exp(-obj_score.cpu().item()))

                        if pred_masks is not None:
                            if pred_masks.shape[1] > 1:
                                best_idx = torch.argmax(current_out["iou_score"], dim=1)
                                pred_mask = pred_masks[torch.arange(pred_masks.size(0)), best_idx].unsqueeze(1)
                            else:
                                pred_mask = pred_masks
                            
                            mask_bool = (pred_mask > 0.0).cpu().numpy().squeeze()
                            
                            if mask_bool.any():
                                # [NEW] Score 정보도 함께 저장
                                detected_objects.append({
                                    'id': oid, 
                                    'mask': mask_bool, 
                                    'score': prob_score
                                })
                                if combined_mask is None: combined_mask = np.zeros_like(mask_bool, dtype=bool)
                                combined_mask = np.maximum(combined_mask, mask_bool)
                        
                        if dummy_idx in obj_output_dict["non_cond_frame_outputs"]:
                            del obj_output_dict["non_cond_frame_outputs"][dummy_idx]

                    total_time += (time.time() - start_time)
                    cnt += 1

                    # 1. Mask 저장 (검출 없으면 검은색)
                    if combined_mask is not None:
                        if combined_mask.ndim == 3: combined_mask = combined_mask.squeeze()
                        final_mask_resized = cv2.resize(
                            combined_mask.astype(np.uint8), 
                            target_orig_size, 
                            interpolation=cv2.INTER_NEAREST
                        )
                    else:
                        final_mask_resized = np.zeros((target_orig_size[1], target_orig_size[0]), dtype=np.uint8)
                    
                    Image.fromarray((final_mask_resized * 255).astype(np.uint8)).save(os.path.join(MASK_OUTPUT_DIR, base_name + "_mask.png"))

                    # 2. Overlay / ID Mask / BBox 저장
                    if detected_objects:
                        vis_id = self.create_overlay_with_id(img_pil, detected_objects, target_orig_size)
                        vis_overlay = self.create_mask_overlay(img_pil, final_mask_resized)
                        vis_bbox = self.create_bbox_overlay(img_pil, final_mask_resized)
                    else:
                        vis_id = img_pil
                        vis_overlay = img_pil
                        vis_bbox = img_pil

                    vis_id.save(os.path.join(ID_MASK_OUTPUT_DIR, base_name + "_id_mask.jpg"), quality=95)
                    vis_overlay.save(os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg"), quality=95)
                    vis_bbox.save(os.path.join(BBOX_OUTPUT_DIR, base_name + "_bbox.jpg"), quality=95)

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                
        print("\n\nAll Done.")
        print(total_time / max(1, cnt))

    def create_overlay_with_id(self, original_img_pil, detected_objects, dsize):
        img_np = np.array(original_img_pil)
        overlay_np = img_np.copy()
        
        for obj in detected_objects:
            resized_mask = cv2.resize(obj['mask'].astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST)
            color = COLORS[obj['id'] % 255] 
            mask_bool = resized_mask > 0
            overlay_np[mask_bool] = (img_np[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)
            
        vis_img = overlay_np.copy()
        for obj in detected_objects:
            resized_mask = cv2.resize(obj['mask'].astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    
                    # [NEW] ID와 점수(%)를 함께 표시
                    score_txt = f"{obj['score']*100:.1f}%"
                    label = f"ID:{obj['id']} ({score_txt})"
                    
                    # 텍스트 외곽선 (가독성 UP)
                    cv2.putText(vis_img, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
                    # 텍스트 본문
                    cv2.putText(vis_img, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return Image.fromarray(vis_img)

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
    original_track_step = app.predictor.track_step
    def patched_track_step(*args, **kwargs):
        if 'gt_masks' in kwargs: del kwargs['gt_masks']
        if 'frames_to_add_correction_pt' in kwargs: del kwargs['frames_to_add_correction_pt']
        return original_track_step(*args, **kwargs)
    app.predictor.track_step = patched_track_step
    
    app.run()