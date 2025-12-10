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

# 일반 출력 경로
MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
ID_MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "id_masks")
OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")
BBOX_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "bboxes")


# 반전(Reverse) 출력 경로
REVERSE_MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reverse_masks")
REVERSE_BBOX_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reverse_bboxes")      # 복구완료
REVERSE_OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "reverse_overlays") # 복구완료

# 디렉토리 초기화 및 생성
if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
if os.path.exists(BASE_OUTPUT_DIR): shutil.rmtree(BASE_OUTPUT_DIR)

# 모든 경로 생성
output_paths = [
    TEMP_DIR, MASK_OUTPUT_DIR, OVERLAY_OUTPUT_DIR, BBOX_OUTPUT_DIR, 
    ID_MASK_OUTPUT_DIR, REVERSE_MASK_OUTPUT_DIR, REVERSE_BBOX_OUTPUT_DIR, REVERSE_OVERLAY_OUTPUT_DIR
]

for path in output_paths:
    os.makedirs(path, exist_ok=True)

# [사용자 데이터 경로]
defect = 'hynix'  
REF_DIR = f"C:/data/Sam3/{defect}/ref"
TARGET_FOLDER = f"C:/data/Sam3/{defect}/target"
CONFIDENCE_SCORE = 0

REF_IMAGES = sorted(
    glob.glob(os.path.join(REF_DIR, "*.png")) + 
    glob.glob(os.path.join(REF_DIR, "*.jpg")) + 
    glob.glob(os.path.join(REF_DIR, "*.bmp"))
)
print(f">> Found {len(REF_IMAGES)} reference images.")

# 색상 팔레트
np.random.seed(42)
COLORS = np.random.randint(0, 255, (255, 3), dtype=np.uint8)
MPL_COLORS = plt.cm.tab20(np.linspace(0, 1, 20))

# ==========================================
# 2. 스마트 박스 & 포인트 셀렉터
# ==========================================
class SmartSelector:
    def __init__(self, img_arr, title, global_thumbnails, global_counts):
        self.img = img_arr
        self.title = title
        self.global_thumbnails = global_thumbnails
        self.global_counts = global_counts
        
        self.annotations = {} 
        
        self.fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(4, 5, width_ratios=[6, 1, 1, 1, 1], figure=self.fig)
        
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_main.imshow(self.img)
        self.ax_main.axis('off')
        
        self.ax_previews = []
        for i in range(16):
            row, col = i // 4, (i % 4) + 1
            ax = self.fig.add_subplot(gs[row, col])
            ax.set_xticks([]); ax.set_yticks([])
            self.ax_previews.append(ax)
            self.refresh_sidebar_slot(i + 1)
            
        self.current_obj_id = 1
        self.press_event = None 
        self.update_title()

    def update_title(self):
        self.ax_main.set_title(
            f"{self.title}\n"
            f"Current ID: {self.current_obj_id} (Keys: 1-9, N=Next)\n"
            f"[Left Drag]: Box / [Left Click]: Positive(*) / [Right Click]: Negative(x)\n"
            f"[Middle Click] or [Key 'D']: Delete Nearest Item"
        , fontsize=10)
        self.fig.canvas.draw()

    def refresh_sidebar_slot(self, obj_id):
        if 1 <= obj_id <= 16:
            ax = self.ax_previews[obj_id - 1]
            ax.clear()
            ax.set_xticks([]); ax.set_yticks([])
            color = MPL_COLORS[(obj_id - 1) % 20]
            count = self.global_counts.get(obj_id, 0)
            
            if obj_id in self.global_thumbnails:
                ax.imshow(self.global_thumbnails[obj_id])
                for spine in ax.spines.values():
                    spine.set_edgecolor(color)
                    spine.set_linewidth(3)
                title_text = f"ID: {obj_id}"
            else:
                title_text = f"ID: {obj_id}"
                ax.text(0.5, 0.5, "Empty", ha='center', va='center', fontsize=8, color='gray')
            
            ax.set_title(title_text, fontsize=8, color=color, fontweight='bold', pad=2)
            ax.text(0.5, -0.1, f"cnt: {count}", ha='center', va='top', transform=ax.transAxes, fontsize=8, fontweight='bold')

    def get_color(self, obj_id):
        return MPL_COLORS[(obj_id - 1) % 20]

    def add_box(self, x1, y1, x2, y2):
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        if (xmax - xmin) < 5 or (ymax - ymin) < 5: return

        self._ensure_id_structure()
        
        if self.annotations[self.current_obj_id]['box'] is not None:
             self._remove_box_artist()

        color = self.get_color(self.current_obj_id)
        rect = patches.Rectangle((xmin, ymin), xmax-xmin, ymax-ymin, linewidth=2, edgecolor=color, facecolor='none')
        self.ax_main.add_patch(rect)
        text = self.ax_main.text(xmin, ymin-5, f"ID:{self.current_obj_id}", color=color, fontsize=10, fontweight='bold', backgroundcolor='white')
        
        self.annotations[self.current_obj_id]['box'] = [xmin, ymin, xmax, ymax]
        self.annotations[self.current_obj_id]['artists'].append(rect)
        self.annotations[self.current_obj_id]['artists'].append(text)
        
        self._update_thumbnail(xmin, ymin, xmax, ymax)
        print(f"  [ID: {self.current_obj_id}] Box Set.")

    def add_point(self, x, y, label):
        self._ensure_id_structure()
        
        color = 'lime' if label == 1 else 'red'
        marker = '*' if label == 1 else 'x'
        
        pt_artist = self.ax_main.scatter([x], [y], c=color, marker=marker, s=100, zorder=5, edgecolors='black')
        
        self.annotations[self.current_obj_id]['points'].append([x, y])
        self.annotations[self.current_obj_id]['labels'].append(label)
        self.annotations[self.current_obj_id]['artists'].append(pt_artist)
        
        type_str = "Positive" if label == 1 else "Negative"
        print(f"  [ID: {self.current_obj_id}] {type_str} Point Added.")
        self.fig.canvas.draw()

    def _ensure_id_structure(self):
        if self.current_obj_id not in self.annotations:
            self.annotations[self.current_obj_id] = {'box': None, 'points': [], 'labels': [], 'artists': []}
            self.global_counts[self.current_obj_id] = self.global_counts.get(self.current_obj_id, 0) + 1
            self.refresh_sidebar_slot(self.current_obj_id)

    def _remove_box_artist(self):
        to_remove = []
        for art in self.annotations[self.current_obj_id]['artists']:
            if isinstance(art, patches.Rectangle) or isinstance(art, plt.Text):
                art.remove()
                to_remove.append(art)
        for art in to_remove:
            self.annotations[self.current_obj_id]['artists'].remove(art)
        self.annotations[self.current_obj_id]['box'] = None

    def _update_thumbnail(self, xmin, ymin, xmax, ymax):
        h, w = self.img.shape[:2]
        bw, bh = xmax - xmin, ymax - ymin
        cx, cy = xmin + bw / 2, ymin + bh / 2
        scale = 1.5
        new_bw, new_bh = bw * scale, bh * scale
        exp_xmin, exp_ymin = int(cx - new_bw/2), int(cy - new_bh/2)
        exp_xmax, exp_ymax = int(cx + new_bw/2), int(cy + new_bh/2)
        safe_xmin = max(0, exp_xmin); safe_ymin = max(0, exp_ymin)
        safe_xmax = min(w, exp_xmax); safe_ymax = min(h, exp_ymax)
        
        crop_img = self.img[safe_ymin:safe_ymax, safe_xmin:safe_xmax]
        self.global_thumbnails[self.current_obj_id] = crop_img
        self.refresh_sidebar_slot(self.current_obj_id)

    def process(self):
        def on_select(eclick, erelease):
            self.add_box(int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata))

        def on_mouse_press(event):
            self.press_event = event

        def on_mouse_release(event):
            if self.press_event is None or event.inaxes != self.ax_main: return
            
            dx = abs(event.xdata - self.press_event.xdata)
            dy = abs(event.ydata - self.press_event.ydata)
            dist = (dx**2 + dy**2)**0.5
            
            if dist < 5: 
                if event.button == 1:   # 좌클릭 -> Positive Point
                    self.add_point(event.xdata, event.ydata, 1)
                elif event.button == 3: # 우클릭 -> Negative Point
                    self.add_point(event.xdata, event.ydata, 0)
                elif event.button == 2: # 휠클릭 -> 삭제
                    self.delete_nearest(event.xdata, event.ydata)
            
            self.press_event = None

        def on_key(event):
            if event.key == 'd' and event.inaxes == self.ax_main:
                self.delete_nearest(event.xdata, event.ydata)

            if event.key in ['q', 'Q', 'enter', 'escape'] and self.current_obj_id <= 10: 
                if event.key.lower() == 'q' and event.key != 'Q': pass 
                else: plt.close(self.fig)
            
            if event.key in ['n', 'N']:
                self.current_obj_id = (self.current_obj_id % 16) + 1
            elif event.key in [str(i) for i in range(1, 10)]:
                self.current_obj_id = int(event.key)
            elif event.key == '0': self.current_obj_id = 10
            else:
                key_map = {'w': 11, 'e': 12, 'r': 13, 't': 14, 'y': 15, 'a': 16}
                if event.key.lower() in key_map: self.current_obj_id = key_map[event.key.lower()]
            
            self.update_title()

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        self.fig.canvas.mpl_connect('button_press_event', on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', on_mouse_release)
        
        self.rs = RectangleSelector(self.ax_main, on_select, useblit=False, button=[1], 
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        
        plt.tight_layout()
        plt.show(block=True)
        return self.annotations

    def delete_nearest(self, x, y):
        if self.current_obj_id not in self.annotations: return
        
        data = self.annotations[self.current_obj_id]
        min_dist = float('inf')
        target_type = None
        target_idx = -1
        
        if data['box'] is not None:
            bx1, by1, bx2, by2 = data['box']
            cx, cy = (bx1+bx2)/2, (by1+by2)/2
            dist = ((x-cx)**2 + (y-cy)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                target_type = 'box'
        
        for i, (px, py) in enumerate(data['points']):
            dist = ((x-px)**2 + (y-py)**2)**0.5
            if dist < min_dist:
                min_dist = dist
                target_type = 'point'
                target_idx = i
        
        if min_dist > 100: return 
        
        if target_type == 'box':
            self._remove_box_artist()
            print(f"  [ID: {self.current_obj_id}] Box Deleted.")
            
        elif target_type == 'point':
            data['points'].pop(target_idx)
            data['labels'].pop(target_idx)
            
            to_remove = [art for art in data['artists'] if isinstance(art, type(plt.scatter([0],[0])))]
            for art in to_remove: 
                try: art.remove()
                except: pass
                if art in data['artists']: data['artists'].remove(art)
                
            for i, (px, py) in enumerate(data['points']):
                lbl = data['labels'][i]
                c = 'lime' if lbl == 1 else 'red'
                m = '*' if lbl == 1 else 'x'
                art = self.ax_main.scatter([px], [py], c=c, marker=m, s=100, zorder=5, edgecolors='black')
                data['artists'].append(art)
                
            print(f"  [ID: {self.current_obj_id}] Point Deleted.")
            
        self.fig.canvas.draw()

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

        self.predictor.max_cond_frames_in_attn = -1 
        self.ref_frames = []
        self.ref_names = []
        self.ref_prompts = [] 
        self.id_thumbnails = {}
        self.id_counts = {}

    def prepare_references(self):
        print("\n--- Step 1: Draw Reference Boxes & Points ---")
        print("Controls:")
        print("  Left Drag  : Draw Box (Area)")
        print("  Left Click : Add Positive Point (*)")
        print("  Right Click: Add Negative Point (x)")
        print("  Mid Click  : Delete nearest item")
        print("  Keys       : 1-9, N(Next ID), Q(Finish)")
        
        for i, path in enumerate(REF_IMAGES):
            if not os.path.exists(path): continue
            
            pil_img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            selector = SmartSelector(
                img_arr, 
                title=f"Ref {i+1} / {len(REF_IMAGES)}", 
                global_thumbnails=self.id_thumbnails,
                global_counts=self.id_counts
            )
            annotations = selector.process()
            
            if not annotations: continue

            self.ref_frames.append(img_arr)
            self.ref_names.append(os.path.basename(path)) 
            
            frame_prompts = []
            
            for obj_id, data in annotations.items():
                prompt_item = {'id': obj_id, 'box': None, 'points': None, 'labels': None}
                
                # Box Normalize
                if data['box'] is not None:
                    box = np.array(data['box'], dtype=np.float32)
                    box[0] /= INPUT_SIZE[0]
                    box[1] /= INPUT_SIZE[1]
                    box[2] /= INPUT_SIZE[0]
                    box[3] /= INPUT_SIZE[1]
                    prompt_item['box'] = box
                
                # Points Normalize
                if data['points']:
                    pts = np.array(data['points'], dtype=np.float32)
                    pts[:, 0] /= INPUT_SIZE[0]
                    pts[:, 1] /= INPUT_SIZE[1]
                    prompt_item['points'] = pts
                    prompt_item['labels'] = np.array(data['labels'], dtype=np.int32)
                
                if prompt_item['box'] is not None or prompt_item['points'] is not None:
                    frame_prompts.append(prompt_item)
            
            self.ref_prompts.append(frame_prompts)
            print(f"  -> Saved Ref {i+1}: {len(frame_prompts)} objects")
            
        if not self.ref_frames: return False
        return True

    def run(self):
        if not self.prepare_references(): return
        target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.jpg")) + glob.glob(os.path.join(TARGET_FOLDER, "*.png")) + glob.glob(os.path.join(TARGET_FOLDER, "*.bmp")))
        print(f"\n--- Step 2: Processing {len(target_files)} images ---")
        self.run_fast_inference(target_files)

    def run_fast_inference(self, target_files):
        # 1. Prepare Dummy Video Structure
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        for i, frame in enumerate(self.ref_frames):
            Image.fromarray(frame).save(os.path.join(TEMP_DIR, f"{i:05d}.jpg"))
            
        dummy_idx = len(self.ref_frames)
        Image.fromarray(np.zeros_like(self.ref_frames[0])).save(os.path.join(TEMP_DIR, f"{dummy_idx:05d}.jpg"))
        
        print(f"\n[Initializing Model State...]")
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        
        # 2. Add References (Box + Points)
        print("  -> Encoding References...")
        tracked_ids = []
        for frame_idx, prompts in enumerate(self.ref_prompts):
            for item in prompts:
                obj_id = item['id']
                
                self.predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=frame_idx,
                    obj_id=obj_id,
                    box=item['box'],
                    points=item['points'],
                    labels=item['labels'],
                    clear_old_points=True
                )
                if obj_id not in tracked_ids:
                    tracked_ids.append(obj_id)

        # 3. Propagate to lock references
        for _ in self.predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=dummy_idx, propagate_preflight=True, reverse=False): pass
        
        device = inference_state["device"]
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        ref_tensor = inference_state["images"][0]
        tensor_h, tensor_w = ref_tensor.shape[-2:]

        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                
                # [메모리 저장] Reference Mask 저장 로직
                print("\n[Saving Reference Masks (Memory ID Masks)...]")
                for r_idx, r_name in enumerate(self.ref_names):
                    base_r_name = os.path.splitext(r_name)[0]
                    orig_h, orig_w = self.ref_frames[r_idx].shape[:2]

                    
                    for oid in tracked_ids:
                        obj_idx = self.predictor._obj_id_to_idx(inference_state, oid)
                        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                        
                        current_out, _ = self.predictor._run_single_frame_inference(
                            inference_state=inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=r_idx,
                            batch_size=1,
                            is_init_cond_frame=True,
                            point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=False
                        )
                        
                        pred_masks = current_out["pred_masks"]
                        if pred_masks is not None:
                            if pred_masks.dim() == 3: pred_masks = pred_masks.unsqueeze(0)
                            if pred_masks.shape[1] > 1:
                                best_idx = torch.argmax(current_out["iou_score"], dim=1) if current_out["iou_score"] is not None else 0
                                pred_mask = pred_masks[torch.arange(pred_masks.size(0)), best_idx].unsqueeze(1)
                            else:
                                pred_mask = pred_masks
                        
                            
                    print(f"  -> Saved Reference Mask: {r_name}")
                print("Reference Masks Saved.\n")

                # [타겟 추론]
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
                            obj_idx = self.predictor._obj_id_to_idx(inference_state, oid)
                            obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                            
                            current_out, _ = self.predictor._run_single_frame_inference(
                                inference_state=inference_state,
                                output_dict=obj_output_dict,
                                frame_idx=dummy_idx,
                                batch_size=1,
                                is_init_cond_frame=False,
                                point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=False
                            )
                
                            pred_masks = current_out["pred_masks"]
                            obj_score = current_out["object_score_logits"] 
                            iou_score = current_out["iou_score"]       

                            # 안전장치
                            if iou_score is not None and iou_score.dim() == 1:
                                iou_score = iou_score.unsqueeze(0)
                                current_out["iou_score"] = iou_score
                            
                            if pred_masks is not None and pred_masks.dim() == 3:
                                pred_masks = pred_masks.unsqueeze(0)

                            if obj_score < CONFIDENCE_SCORE: continue

                            if isinstance(obj_score, torch.Tensor):
                                obj_score = obj_score.cpu().float().item()

                            prob_score = 1.0 / (1.0 + np.exp(-obj_score))
                            
                            if pred_masks is not None:
                                if pred_masks.shape[1] > 1:
                                    best_idx = torch.argmax(current_out["iou_score"], dim=1)
                                    pred_mask = pred_masks[torch.arange(pred_masks.size(0)), best_idx].unsqueeze(1)
                                else:
                                    pred_mask = pred_masks
                                
                                mask_bool = (pred_mask > 0.0).cpu().numpy().squeeze()
                                
                                if mask_bool.ndim > 0 and mask_bool.any():
                                    detected_objects.append({'id': oid, 'mask': mask_bool, 'score': prob_score})
                                    if combined_mask is None: combined_mask = np.zeros_like(mask_bool, dtype=bool)
                                    combined_mask = np.maximum(combined_mask, mask_bool)
                            
                            if dummy_idx in obj_output_dict["non_cond_frame_outputs"]:
                                del obj_output_dict["non_cond_frame_outputs"][dummy_idx]

                        total_time += (time.time() - start_time)
                        cnt += 1

                        # 결과 저장
                        if combined_mask is not None:
                            if combined_mask.ndim == 3: combined_mask = combined_mask.squeeze()
                            final_mask = cv2.resize(combined_mask.astype(np.uint8), target_orig_size, interpolation=cv2.INTER_NEAREST)
                            reverse_mask = 1 - final_mask
                        else:
                            final_mask = np.zeros((target_orig_size[1], target_orig_size[0]), dtype=np.uint8)
                            reverse_mask = np.ones((target_orig_size[1], target_orig_size[0]), dtype=np.uint8)
                        
                        Image.fromarray((final_mask * 255).astype(np.uint8)).save(os.path.join(MASK_OUTPUT_DIR, base_name + "_mask.png"))
                        Image.fromarray((reverse_mask * 255).astype(np.uint8)).save(os.path.join(REVERSE_MASK_OUTPUT_DIR, base_name + "_rev_mask.png"))
                        
                        # [복구된 부분] 일반 오버레이/BBox 저장
                        if detected_objects:
                            vis_id = self.create_overlay_with_id(img_pil, detected_objects, target_orig_size)
                            vis_overlay = self.create_mask_overlay(img_pil, final_mask, color=(255, 0, 0))
                            vis_bbox = self.create_bbox_overlay(img_pil, final_mask, color=(0, 255, 0))
                            vis_id.save(os.path.join(ID_MASK_OUTPUT_DIR, base_name + "_id_mask.jpg"))
                            vis_overlay.save(os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg"))
                            vis_bbox.save(os.path.join(BBOX_OUTPUT_DIR, base_name + "_bbox.jpg"))
                        else:
                            img_pil.save(os.path.join(ID_MASK_OUTPUT_DIR, base_name + "_id_mask.jpg"))
                            img_pil.save(os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg"))
                            img_pil.save(os.path.join(BBOX_OUTPUT_DIR, base_name + "_bbox.jpg"))

                        # [복구된 부분] Reverse 오버레이/BBox 저장
                        vis_rev_overlay = self.create_mask_overlay(img_pil, reverse_mask, color=(255, 0, 0))
                        vis_rev_bbox = self.create_bbox_overlay(img_pil, reverse_mask, color=(0, 255, 0))
                        
                        vis_rev_overlay.save(os.path.join(REVERSE_OVERLAY_OUTPUT_DIR, base_name + "_rev_overlay.jpg"))
                        vis_rev_bbox.save(os.path.join(REVERSE_BBOX_OUTPUT_DIR, base_name + "_rev_bbox.jpg"))

                    except Exception as e:
                        print(f"\nError processing {filename}: {e}")
                        import traceback
                        traceback.print_exc()

        print("\n\nAll Done.")
        print(f"Avg Time: {total_time / max(1, cnt):.4f} sec")

    # (Visualization helpers remain the same as previous code...)
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
                    label = f"ID:{obj['id']} ({obj['score']*100:.0f}%)"
                    cv2.putText(vis_img, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
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
    # Patch track_step if needed (same as before)
    original_track_step = app.predictor.track_step
    def patched_track_step(*args, **kwargs):
        if 'gt_masks' in kwargs: del kwargs['gt_masks']
        if 'frames_to_add_correction_pt' in kwargs: del kwargs['frames_to_add_correction_pt']
        return original_track_step(*args, **kwargs)
    app.predictor.track_step = patched_track_step
    
    app.run()