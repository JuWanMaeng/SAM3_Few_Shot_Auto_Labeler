# main.py
import os
import glob
import time
import numpy as np
import torch
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_video_model

# 분리된 모듈 임포트
import config
from gui import SmartSelector
from utils import Visualizer  # [추가] utils 불러오기

class InteractiveBatchLabeler:
    def __init__(self):
        config.setup_directories()
        
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

        # 데이터 로드
        self.ref_images = sorted(
            glob.glob(os.path.join(config.REF_DIR, "*.png")) + 
            glob.glob(os.path.join(config.REF_DIR, "*.jpg")) + 
            glob.glob(os.path.join(config.REF_DIR, "*.bmp"))
        )
        print(f">> Found {len(self.ref_images)} reference images.")

    def prepare_references(self):
        print("\n--- Step 1: Draw Reference Boxes, Points & Brush ---")
        
        for i, path in enumerate(self.ref_images):
            if not os.path.exists(path): continue
            
            pil_img = Image.open(path).convert("RGB").resize(config.INPUT_SIZE)
            img_arr = np.array(pil_img)
            
            selector = SmartSelector(
                img_arr, 
                title=f"Ref {i+1} / {len(self.ref_images)}", 
                global_thumbnails=self.id_thumbnails,
                global_counts=self.id_counts
            )
            annotations = selector.process()
            
            if not annotations: continue

            self.ref_frames.append(img_arr)
            self.ref_names.append(os.path.basename(path)) 
            
            frame_prompts = []
            
            for obj_id, data in annotations.items():
                prompt_item = {'id': obj_id, 'box': None, 'points': None, 'labels': None, 'mask': None}
                
                # Box Normalize
                if data['box'] is not None:
                    box = np.array(data['box'], dtype=np.float32)
                    box[[0, 2]] /= config.INPUT_SIZE[0]
                    box[[1, 3]] /= config.INPUT_SIZE[1]
                    prompt_item['box'] = box
                
                # Points Normalize
                if data['points']:
                    pts = np.array(data['points'], dtype=np.float32)
                    pts[:, 0] /= config.INPUT_SIZE[0]
                    pts[:, 1] /= config.INPUT_SIZE[1]
                    prompt_item['points'] = pts
                    prompt_item['labels'] = np.array(data['labels'], dtype=np.int32)
                
                # Mask (Brush)
                if data.get('mask') is not None:
                    # 마스크가 존재하면 저장 (0/1 binary)
                    prompt_item['mask'] = (data['mask'] > 0).astype(np.float32)

                if any(x is not None for x in [prompt_item['box'], prompt_item['points'], prompt_item['mask']]):
                    frame_prompts.append(prompt_item)
            
            self.ref_prompts.append(frame_prompts)
            print(f"  -> Saved Ref {i+1}: {len(frame_prompts)} objects")
            
        if not self.ref_frames: return False
        return True

    def run(self):
        if not self.prepare_references(): return
        
        target_files = sorted(glob.glob(os.path.join(config.TARGET_FOLDER, "*.jpg")) + 
                              glob.glob(os.path.join(config.TARGET_FOLDER, "*.png")) + 
                              glob.glob(os.path.join(config.TARGET_FOLDER, "*.bmp")))
        
        print(f"\n--- Step 2: Processing {len(target_files)} images ---")
        self.run_fast_inference(target_files)

    def run_fast_inference(self, target_files):
        # 1. Prepare Dummy Video
        for i, frame in enumerate(self.ref_frames):
            Image.fromarray(frame).save(os.path.join(config.TEMP_DIR, f"{i:05d}.jpg"))
            
        dummy_idx = len(self.ref_frames)
        Image.fromarray(np.zeros_like(self.ref_frames[0])).save(os.path.join(config.TEMP_DIR, f"{dummy_idx:05d}.jpg"))
        
        print(f"\n[Initializing Model State...]")
        inference_state = self.predictor.init_state(video_path=config.TEMP_DIR)
        device = inference_state["device"]

        # 2. Add References
        print("  -> Encoding References...")
        tracked_ids = []
        for frame_idx, prompts in enumerate(self.ref_prompts):
            for item in prompts:
                obj_id = item['id']
                
                # 1) Box와 Point가 있는 경우 처리
                if item['box'] is not None or item['points'] is not None:
                    self.predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        box=item['box'],
                        points=item['points'],
                        labels=item['labels'],
                        clear_old_points=True
                    )
                
                # 2) Mask (Brush)가 있는 경우 별도 함수(add_new_mask)로 처리
                if item['mask'] is not None:
                    # (H, W)의 2D Tensor (bool 타입)
                    mask_tensor = torch.from_numpy(item['mask']).bool().to(device)
                    self.predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        mask=mask_tensor
                    )

                if obj_id not in tracked_ids:
                    tracked_ids.append(obj_id)

        # 3. Propagate
        for _ in self.predictor.propagate_in_video(inference_state, start_frame_idx=0, max_frame_num_to_track=dummy_idx, propagate_preflight=True, reverse=False): pass
        
        # [메모리 저장 및 타겟 추론 준비]
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        
        ref_tensor = inference_state["images"][0]
        tensor_h, tensor_w = ref_tensor.shape[-2:]

        # Reference Mask 저장은 로그만 남기고 생략 (필요시 추가)
        print("\n[Saving Reference Masks... Done]")
        
        # [타겟 이미지 추론 루프]
        cnt = 0
        total_time = 0
        
        # [중요] InferenceMode 사용
        with torch.inference_mode():  
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
                    
                    # 추론 버퍼 업데이트
                    inference_state["images"][dummy_idx] = img_tensor
                    
                    if dummy_idx in inference_state["cached_features"]:
                        del inference_state["cached_features"][dummy_idx]

                    detected_objects = []
                    combined_mask = None
                    
                    for oid in tracked_ids:
                        obj_idx = self.predictor._obj_id_to_idx(inference_state, oid)
                        obj_output_dict = inference_state["output_dict_per_obj"][obj_idx]
                        
                        # [중요] 필수 인자 추가됨
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
                        
                        if isinstance(obj_score, torch.Tensor): obj_score = obj_score.item()
                        if obj_score < config.CONFIDENCE_SCORE: continue
                        
                        prob_score = 1.0 / (1.0 + np.exp(-obj_score))
                        
                        if pred_masks is not None:
                            if pred_masks.dim() == 3: pred_masks = pred_masks.unsqueeze(0)
                            best_idx = 0 
                            pred_mask = pred_masks[0, best_idx].unsqueeze(0)
                            mask_bool = (pred_mask > 0.0).cpu().numpy().squeeze()
                            
                            if mask_bool.ndim > 0 and mask_bool.any():
                                detected_objects.append({'id': oid, 'mask': mask_bool, 'score': prob_score})
                                if combined_mask is None: combined_mask = np.zeros_like(mask_bool, dtype=bool)
                                combined_mask = np.maximum(combined_mask, mask_bool)
                        
                        if dummy_idx in obj_output_dict["non_cond_frame_outputs"]:
                            del obj_output_dict["non_cond_frame_outputs"][dummy_idx]

                    total_time += (time.time() - start_time)
                    cnt += 1

                    # 결과 마스크 생성
                    if combined_mask is not None:
                        if combined_mask.ndim == 3: combined_mask = combined_mask.squeeze()
                        final_mask = cv2.resize(combined_mask.astype(np.uint8), target_orig_size, interpolation=cv2.INTER_NEAREST)
                        reverse_mask = 1 - final_mask
                    else:
                        final_mask = np.zeros((target_orig_size[1], target_orig_size[0]), dtype=np.uint8)
                        reverse_mask = np.ones((target_orig_size[1], target_orig_size[0]), dtype=np.uint8)

                    # 마스크 저장
                    Image.fromarray((final_mask * 255).astype(np.uint8)).save(os.path.join(config.MASK_OUTPUT_DIR, base_name + "_mask.png"))
                    
                    # Visualizer 클래스를 사용하여 시각화 저장
                    Visualizer.save_visualizations(img_pil, detected_objects, final_mask, reverse_mask, base_name, target_orig_size)

                except Exception as e:
                    print(f"\nError processing {filename}: {e}")
                    import traceback; traceback.print_exc()

        print("\n\nAll Done.")
        print(f"Avg Time: {total_time / max(1, cnt):.4f} sec")

if __name__ == "__main__":
    app = InteractiveBatchLabeler()
    
    # Patch track_step (Same as original)
    original_track_step = app.predictor.track_step
    def patched_track_step(*args, **kwargs):
        if 'gt_masks' in kwargs: del kwargs['gt_masks']
        if 'frames_to_add_correction_pt' in kwargs: del kwargs['frames_to_add_correction_pt']
        return original_track_step(*args, **kwargs)
    app.predictor.track_step = patched_track_step
    
    app.run()