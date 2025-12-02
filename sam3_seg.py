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
import gc
from PIL import Image, ImageEnhance
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

# ==========================================
# 2. 박스 선택 도구
# ==========================================
class MultiBoxSelector:
    def __init__(self, img_arr, title):
        self.img = img_arr
        self.boxes = []
        self.title = title
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.rs = None

    def select_boxes(self):
        self.ax.imshow(self.img)
        self.ax.set_title(f"{self.title}\n(Drag mouse to draw box -> Press 'Q' or 'Enter')")
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
# 3. 메인 프로세서
# ==========================================
class InteractiveBatchLabeler:
    def __init__(self):
        print("Loading SAM 3 Model...")
        self.sam3_model = build_sam3_video_model()
        self.predictor = self.sam3_model.tracker
        self.predictor.backbone = self.sam3_model.detector.backbone
        print("Model Loaded.")

        self.ref_data = [] # {original_img, boxes} 저장

    def prepare_references(self):
        print("\n--- Step 1: Draw Reference Boxes (Once) ---")
        for i, path in enumerate(REF_IMAGES):
            if not os.path.exists(path): continue
            
            pil_img = Image.open(path).convert("RGB").resize(INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            # 사용자에게는 '원본'만 보여줌
            selector = MultiBoxSelector(img_arr, title=f"Ref {i+1} / {len(REF_IMAGES)}")
            boxes = selector.select_boxes()
            
            if not boxes: continue

            # 박스 정규화 (0~1)
            np_boxes = np.array(boxes, dtype=np.float32)
            norm_boxes = np_boxes.copy()
            norm_boxes[:, 0] /= INPUT_SIZE[0]
            norm_boxes[:, 1] /= INPUT_SIZE[1]
            norm_boxes[:, 2] /= INPUT_SIZE[0]
            norm_boxes[:, 3] /= INPUT_SIZE[1]
            
            # 원본 데이터 저장
            self.ref_data.append({
                'image': pil_img,
                'boxes': norm_boxes
            })
            print(f"  -> Saved Ref {i+1}: {len(boxes)} boxes")
            
        if not self.ref_data: return False
        return True

    def _augment_image(self, pil_img):
        """
        이미지 1장을 받아서 다양한 변환(Augmentation)이 적용된 리스트로 반환
        박스 좌표가 변하지 않는 픽셀 변환만 수행함.
        """
        augs = []
        
        # 1. 원본
        augs.append(np.array(pil_img))
        
        # 2. CLAHE (실금 검출 핵심 비기) - 대비 극대화
        # 원본 PIL 이미지를 numpy로 변환 후 적용
        # img_np = np.array(pil_img)
        # if len(img_np.shape) == 3:
        #     lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        #     l, a, b = cv2.split(lab)
        #     # ClipLimit 4.0: 대비를 강하게 줌
        #     clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8)) 
        #     cl = clahe.apply(l)
        #     limg = cv2.merge((cl, a, b))
        #     final_clahe = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        #     augs.append(final_clahe)
        
        # 3. 어둡게 (0.6배)
        # enhancer = ImageEnhance.Brightness(pil_img)
        # augs.append(np.array(enhancer.enhance(0.6)))
        
        # # 4. 밝게 (1.4배)
        # augs.append(np.array(enhancer.enhance(1.4)))
        
        # # 5. 대비 강조 (1.5배)
        # enhancer = ImageEnhance.Contrast(pil_img)
        # augs.append(np.array(enhancer.enhance(1.5)))
        
        # # 6. 블러 (초점 나간 경우 대비)
        # blurred = cv2.GaussianBlur(img_np, (5, 5), 0)
        # augs.append(blurred)
        
        return augs

    def run(self):
        if not self.prepare_references(): return
        target_files = sorted(glob.glob(os.path.join(TARGET_FOLDER, "*.jpg")) + glob.glob(os.path.join(TARGET_FOLDER, "*.png")) + glob.glob(os.path.join(TARGET_FOLDER, "*.bmp")))
        print(f"\n--- Step 2: Processing {len(target_files)} images (Augmented Ref Mode) ---")
        self.run_fast_inference(target_files)

    def run_fast_inference(self, target_files):
        # 1. 초기화 및 증강(Augmentation) 적용
        if os.path.exists(TEMP_DIR): shutil.rmtree(TEMP_DIR)
        os.makedirs(TEMP_DIR, exist_ok=True)
        
        print("  -> Generating Augmented References...")
        
        # 증강된 모든 레퍼런스를 저장할 리스트
        # 구조: {'path': 저장경로, 'boxes': 박스좌표, 'original_ref_idx': 몇번째Ref인지}
        expanded_refs = [] 
        
        frame_idx_counter = 0
        
        for ref_item in self.ref_data:
            original_pil = ref_item['image']
            boxes = ref_item['boxes']
            
            # [핵심] 1장의 이미지를 5장으로 뻥튀기 (좌표는 공유)
            aug_imgs_np = self._augment_image(original_pil)
            
            for img_np in aug_imgs_np:
                save_path = os.path.join(TEMP_DIR, f"{frame_idx_counter:05d}.jpg")
                Image.fromarray(img_np).save(save_path)
                
                expanded_refs.append({
                    'frame_idx': frame_idx_counter,
                    'boxes': boxes # 모든 증강 이미지에 동일한 박스 적용
                })
                frame_idx_counter += 1
                
        print(f"  -> Total References Expanded: {len(self.ref_data)} images => {len(expanded_refs)} images")

        # Dummy 이미지 추가
        dummy_idx = len(expanded_refs)
        # (주의: shape 맞추기 위해 첫번째 Ref 이미지 크기 사용)
        dummy_img_np = np.zeros((INPUT_SIZE[1], INPUT_SIZE[0], 3), dtype=np.uint8)
        Image.fromarray(dummy_img_np).save(os.path.join(TEMP_DIR, f"{dummy_idx:05d}.jpg"))

        print(f"\n[Initializing Model State...]")
        inference_state = self.predictor.init_state(video_path=TEMP_DIR)
        
        # 실제 텐서 크기 확인
        ref_tensor = inference_state["images"][0] 
        tensor_h, tensor_w = ref_tensor.shape[-2:] 
        print(f"  -> Actual Tensor Size in Memory: {tensor_w}x{tensor_h} (Fixed)")
        
        # 2. Reference 학습 (증강된 모든 이미지에 대해 수행)
        print("  -> Encoding All References...")
        global_obj_id = 1
        tracked_ids = []
        
        # Ref Image 1장에 박스가 3개였다면, 증강된 5장에 대해 각각 3개씩 박스를 입력해야 함?
        # 아니면 객체 ID를 공유해야 할까? 
        # -> SAM3 Video에서는 "같은 ID"를 여러 프레임에 걸쳐 보여주면 더 강력하게 학습합니다.
        # 즉, 원본의 1번 박스와 어두운 버전의 1번 박스는 '같은 객체(ID 1)'라고 알려주는 것이 좋습니다.
        
        # 객체 ID 할당 로직:
        # Ref 이미지 A의 1번 박스 -> 모든 증강 이미지에서 ID 1
        # Ref 이미지 A의 2번 박스 -> 모든 증강 이미지에서 ID 2
        # Ref 이미지 B의 1번 박스 -> 모든 증강 이미지에서 ID 3 (새로운 객체니까)
        
        obj_id_start = 1
        
        # 원래 사용자가 등록한 Ref 이미지 단위로 루프
        # (expanded_refs 리스트를 순회하면 ID 관리가 복잡해지므로 다시 구조화)
        
        # 현재 expanded_refs는 순서대로 저장되어 있음.
        # 예: [Ref1_Org, Ref1_Dark, Ref1_Bright, ..., Ref2_Org, Ref2_Dark, ...]
        # 1개의 Ref 당 증강 개수(N)를 알아야 함.
        num_augs = len(self._augment_image(self.ref_data[0]['image']))
        
        for i, ref_item in enumerate(self.ref_data):
            boxes = ref_item['boxes']
            
            # 이 Ref 이미지에 있는 박스들의 로컬 ID 리스트
            # 예: 박스가 2개면 current_ids = [1, 2] (Global Offset 적용 전)
            
            for box_idx, box in enumerate(boxes):
                # 이 박스에 대한 Global ID
                current_obj_id = obj_id_start + box_idx
                if current_obj_id not in tracked_ids:
                    tracked_ids.append(current_obj_id)
                
                # 증강된 모든 프레임에 대해 "이 박스가 그 놈(ID)이다"라고 알려줌
                # i번째 Ref의 증강 이미지들은 (i * num_augs) 부터 (i * num_augs + num_augs - 1) 까지임
                start_frame = i * num_augs
                end_frame = start_frame + num_augs
                
                for f_idx in range(start_frame, end_frame):
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=f_idx,
                        obj_id=current_obj_id,
                        box=box,
                        clear_old_points=True
                    )
            
            # 다음 Ref 이미지의 객체들은 새로운 ID를 써야 하므로 Offset 증가
            obj_id_start += len(boxes)

        # Ref 학습 (Dummy 전까지)
        for _ in self.predictor.propagate_in_video(
            inference_state, 
            start_frame_idx=0, 
            max_frame_num_to_track=dummy_idx, 
            reverse=False, propagate_preflight=True
        ): pass
        
        print("  -> References Locked. Starting High-Speed Inference Loop.")

        # 3. 고속 추론 루프
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
                    
                    # Swapping
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
                        
                        # [메모리 누수 방지] 결과 기록 삭제
                        if dummy_idx in obj_output_dict["non_cond_frame_outputs"]:
                            del obj_output_dict["non_cond_frame_outputs"][dummy_idx]

                    total_time += (time.time() - start_time)
                    cnt += 1
                    
                    if combined_mask is not None:
                        if combined_mask.ndim == 3: combined_mask = combined_mask.squeeze()
                        
                        final_mask_resized = cv2.resize(
                            combined_mask.astype(np.uint8), 
                            target_orig_size, 
                            interpolation=cv2.INTER_NEAREST
                        )
                        
                        # 결과 저장 (Mask, Overlay, BBox)
                        Image.fromarray((final_mask_resized * 255).astype(np.uint8)).save(os.path.join(MASK_OUTPUT_DIR, base_name + "_mask.png"))
                        self.create_mask_overlay(img_pil, final_mask_resized).save(os.path.join(OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg"), quality=95)
                        self.create_bbox_overlay(img_pil, final_mask_resized).save(os.path.join(BBOX_OUTPUT_DIR, base_name + "_bbox.jpg"), quality=95)

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")
                    import traceback
                    traceback.print_exc()
                
                if (t_idx + 1) % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()
                
        print("\n\nAll Done.")
        print(total_time / max(1, cnt))

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
    # Patch
    original_track_step = app.predictor.track_step
    def patched_track_step(*args, **kwargs):
        if 'gt_masks' in kwargs: del kwargs['gt_masks']
        if 'frames_to_add_correction_pt' in kwargs: del kwargs['frames_to_add_correction_pt']
        return original_track_step(*args, **kwargs)
    app.predictor.track_step = patched_track_step
    
    start_time = time.time()
    app.run()
    print(f"Total Time: {time.time() - start_time:.2f}s")