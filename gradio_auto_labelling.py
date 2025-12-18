import gradio as gr
import os
import glob
import time
import numpy as np
import torch
import cv2
from PIL import Image
from sam3.model_builder import build_sam3_video_model
import shutil
import zipfile

# ==========================================
# [Config] 설정
# ==========================================
class Config:
    INPUT_SIZE = (1024, 1024)
    TEMP_DIR = "./temp_frames"
    BASE_OUTPUT_DIR = "./output_gradio"
    MASK_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "masks")
    OVERLAY_OUTPUT_DIR = os.path.join(BASE_OUTPUT_DIR, "overlays")
    
    # ★ 모델 경로를 본인 환경에 맞게 꼭 수정해주세요 ★
    MODEL_PATH = 'models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt'
    CONFIDENCE_SCORE = -2

def setup_directories():
    if os.path.exists(Config.TEMP_DIR): shutil.rmtree(Config.TEMP_DIR)
    if os.path.exists(Config.BASE_OUTPUT_DIR): shutil.rmtree(Config.BASE_OUTPUT_DIR)
    
    os.makedirs(Config.TEMP_DIR, exist_ok=True)
    os.makedirs(Config.MASK_OUTPUT_DIR, exist_ok=True)
    os.makedirs(Config.OVERLAY_OUTPUT_DIR, exist_ok=True)

# ==========================================
# [Logic] SAM3 Engine
# ==========================================
class SAM3Engine:
    def __init__(self):
        self.model = None
        self.predictor = None
        self.inference_state = None
        self.tracked_ids = []
        self.ref_tensor_shape = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model(self):
        if self.model is not None: return "✅ Model already loaded."
        print("Loading SAM 3 Model...")
        try:
            self.model = build_sam3_video_model(Config.MODEL_PATH)
            self.predictor = self.model.tracker
            self.predictor.backbone = self.model.detector.backbone
            self.predictor.max_cond_frames_in_attn = -1
            
            original_track_step = self.predictor.track_step
            def patched_track_step(*args, **kwargs):
                if 'gt_masks' in kwargs: del kwargs['gt_masks']
                if 'frames_to_add_correction_pt' in kwargs: del kwargs['frames_to_add_correction_pt']
                return original_track_step(*args, **kwargs)
            self.predictor.track_step = patched_track_step
            
            print("Model Loaded.")
            return "✅ Model Loaded Successfully."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def generate_memory_bank(self, ref_files, all_annotations):
        setup_directories()
        
        if not self.model: return "❌ Model not loaded."
        if not ref_files: return "❌ No reference images uploaded."
        if not all_annotations: return "❌ No annotations found."

        print("\n[Step 1] Generating Memory Bank...")
        
        ref_frames_data = []
        for i, path in enumerate(ref_files):
            img = Image.open(path).convert("RGB").resize(Config.INPUT_SIZE)
            save_path = os.path.join(Config.TEMP_DIR, f"{i:05d}.jpg")
            img.save(save_path)
            ref_frames_data.append(np.array(img))
            
        dummy_idx = len(ref_files)
        Image.fromarray(np.zeros_like(ref_frames_data[0])).save(os.path.join(Config.TEMP_DIR, f"{dummy_idx:05d}.jpg"))

        self.inference_state = self.predictor.init_state(video_path=Config.TEMP_DIR)
        self.tracked_ids = set()
        
        print(" -> Encoding Annotations...")
        for frame_idx, obj_dict in all_annotations.items():
            if frame_idx >= len(ref_files): continue
            
            for obj_id, data in obj_dict.items():
                box = None
                if data['box']:
                    box = np.array(data['box'], dtype=np.float32)
                    box[[0, 2]] /= Config.INPUT_SIZE[0]
                    box[[1, 3]] /= Config.INPUT_SIZE[1]
                
                points, labels = None, None
                if data['points']:
                    points = np.array(data['points'], dtype=np.float32)
                    points[:, 0] /= Config.INPUT_SIZE[0]
                    points[:, 1] /= Config.INPUT_SIZE[1]
                    labels = np.array(data['labels'], dtype=np.int32)
                
                if box is not None or points is not None:
                    self.predictor.add_new_points_or_box(
                        inference_state=self.inference_state,
                        frame_idx=frame_idx,
                        obj_id=obj_id,
                        box=box,
                        points=points,
                        labels=labels,
                        clear_old_points=True
                    )
                    self.tracked_ids.add(obj_id)

        if not self.tracked_ids: return "❌ No valid annotations found."

        print(" -> Propagating Memory...")
        for _ in self.predictor.propagate_in_video(self.inference_state, start_frame_idx=0, max_frame_num_to_track=dummy_idx, propagate_preflight=True,reverse=False):
            pass
        
        ref_tensor = self.inference_state["images"][0]
        self.ref_tensor_shape = ref_tensor.shape[-2:]

        return f"✅ Memory Bank Ready! (Tracked IDs: {list(self.tracked_ids)})"

    def run_target_inference(self, target_files):
        if self.inference_state is None: return "❌ Memory Bank not generated yet.", [], []
        if not target_files: return "❌ No target images.", [], []

        print(f"\n[Step 2] Inference on {len(target_files)} targets...")
        
        pixel_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        pixel_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        tensor_h, tensor_w = self.ref_tensor_shape
        dummy_idx = len(self.inference_state["images"]) - 1
        
        mask_results = []
        overlay_results = []

        with torch.inference_mode():
            for t_path in target_files:
                try:
                    base_name = os.path.basename(t_path)
                    
                    img_pil = Image.open(t_path).convert("RGB")
                    target_orig_size = img_pil.size
                    img_resized = img_pil.resize((tensor_w, tensor_h))
                    img_np = np.array(img_resized)
                    
                    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1).float().to(self.device) / 255.0
                    img_tensor = (img_tensor - pixel_mean) / pixel_std
                    if img_tensor.dim() == 3: img_tensor = img_tensor.unsqueeze(0)
                    
                    self.inference_state["images"][dummy_idx] = img_tensor
                    if dummy_idx in self.inference_state["cached_features"]:
                        del self.inference_state["cached_features"][dummy_idx]
                    
                    combined_mask = None
                    
                    for oid in self.tracked_ids:
                        obj_idx = self.predictor._obj_id_to_idx(self.inference_state, oid)
                        obj_output_dict = self.inference_state["output_dict_per_obj"][obj_idx]
                        
                        current_out, _ = self.predictor._run_single_frame_inference(
                            inference_state=self.inference_state,
                            output_dict=obj_output_dict,
                            frame_idx=dummy_idx,
                            batch_size=1,
                            is_init_cond_frame=False,
                            point_inputs=None, mask_inputs=None, reverse=False, run_mem_encoder=False
                        )
                        
                        pred_masks = current_out["pred_masks"]
                        obj_score = current_out["object_score_logits"]
                        if isinstance(obj_score, torch.Tensor): obj_score = obj_score.item()
                        
                        if obj_score > Config.CONFIDENCE_SCORE and pred_masks is not None:
                             if pred_masks.dim() == 3: pred_masks = pred_masks.unsqueeze(0)
                             mask_bool = (pred_masks[0, 0].unsqueeze(0) > 0.0).cpu().numpy().squeeze()
                             
                             if mask_bool.ndim > 0 and mask_bool.any():
                                 if combined_mask is None: combined_mask = np.zeros_like(mask_bool, dtype=bool)
                                 combined_mask = np.maximum(combined_mask, mask_bool)
                        
                        if dummy_idx in obj_output_dict["non_cond_frame_outputs"]:
                            del obj_output_dict["non_cond_frame_outputs"][dummy_idx]

                    if combined_mask is not None:
                        if combined_mask.ndim == 3: combined_mask = combined_mask.squeeze()
                        final_mask = cv2.resize(combined_mask.astype(np.uint8), target_orig_size, interpolation=cv2.INTER_NEAREST)
                    else:
                        final_mask = np.zeros((target_orig_size[1], target_orig_size[0]), dtype=np.uint8)
                    
                    mask_save_path = os.path.join(Config.MASK_OUTPUT_DIR, base_name + "_mask.png")
                    Image.fromarray((final_mask * 255).astype(np.uint8)).save(mask_save_path)
                    mask_results.append(mask_save_path)

                    orig_img_np = np.array(img_pil)
                    overlay_img = orig_img_np.copy()
                    
                    if combined_mask is not None:
                        color_mask = np.zeros_like(orig_img_np)
                        color_mask[final_mask > 0] = [0, 255, 0] 
                        overlay_img = cv2.addWeighted(orig_img_np, 0.7, color_mask, 0.3, 0)
                        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.drawContours(overlay_img, contours, -1, (0, 255, 0), 2)

                    overlay_save_path = os.path.join(Config.OVERLAY_OUTPUT_DIR, base_name + "_overlay.png")
                    Image.fromarray(overlay_img).save(overlay_save_path)
                    overlay_results.append(overlay_save_path)

                except Exception as e:
                    print(f"Error processing {t_path}: {e}")

        return f"✅ Done. {len(mask_results)} images processed.", overlay_results, mask_results

    def zip_results(self):
        zip_filename = "inference_results.zip"
        if os.path.exists(zip_filename): os.remove(zip_filename)
        shutil.make_archive("inference_results", 'zip', Config.BASE_OUTPUT_DIR)
        return "inference_results.zip"

sam_engine = SAM3Engine()

# ==========================================
# [Helper] Visualization (개선됨)
# ==========================================
def draw_annotations_on_image(img_arr, annotations_dict, current_view_id):
    """이미지 위에 현재 상태(박스, 점)를 그립니다."""
    # 원본 보호를 위해 복사
    vis = img_arr.copy()
    if annotations_dict is None: return vis

    # 비활성 객체 (회색으로 얇게)
    for oid, data in annotations_dict.items():
        if oid == current_view_id: continue
        _draw_obj(vis, data, (200, 200, 200), 1)

    # 활성 객체 (현재 선택된 ID, 초록색으로 굵게)
    if current_view_id in annotations_dict:
        _draw_obj(vis, annotations_dict[current_view_id], (0, 255, 0), 3)

    return vis

def _draw_obj(img, data, color, thickness):
    """단일 객체(Box/Points) 그리기 함수 - X표시 추가"""
    
    # 1. Box 그리기
    if data.get('box') is not None:
        x1, y1, x2, y2 = map(int, data['box'])
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # 2. Points 그리기
    points = data.get('points', [])
    labels = data.get('labels', [])
    
    for pt, lbl in zip(points, labels):
        px, py = map(int, pt)
        
        if lbl == 1:
            # [Positive Point] - 초록색 원 (Green Circle)
            # 시인성을 위해 검은 테두리를 먼저 그림
            cv2.circle(img, (px, py), 6, (0, 0, 0), 2) 
            cv2.circle(img, (px, py), 4, (0, 255, 0), -1) # 내부 채움
            
        else:
            # [Negative Point] - 빨간색 X (Red Cross)
            r = 6 # 십자가 크기
            red_color = (255, 0, 0) # RGB format (Pillow array is RGB)
            
            # 시인성을 위해 검은 두꺼운 X를 먼저 그림 (배경)
            cv2.line(img, (px-r, py-r), (px+r, py+r), (0, 0, 0), 3)
            cv2.line(img, (px+r, py-r), (px-r, py+r), (0, 0, 0), 3)
            
            # 빨간 얇은 X를 위에 그림
            cv2.line(img, (px-r, py-r), (px+r, py+r), red_color, 2)
            cv2.line(img, (px+r, py-r), (px-r, py+r), red_color, 2)

# ==========================================
# [Gradio Logic]
# ==========================================
def on_files_upload(files):
    if not files: return 0, [], {}, None, "Please upload images."
    return 0, files, {}, None, f"Loaded {len(files)} reference images."

def update_view(file_idx, file_list, all_anns, current_obj_id):
    if not file_list or file_idx >= len(file_list): return None, "No Image"
    path = file_list[file_idx]
    # Gradio Image는 numpy array를 기대함
    img = np.array(Image.open(path).convert("RGB").resize(Config.INPUT_SIZE))
    frame_anns = all_anns.get(file_idx, {})
    vis_img = draw_annotations_on_image(img, frame_anns, current_obj_id)
    return vis_img, f"Image {file_idx + 1} / {len(file_list)}"

def navigate(direction, current_idx, file_list, all_anns, current_obj_id):
    if not file_list: return None, 0, "No files"
    new_idx = max(0, min(current_idx + direction, len(file_list) - 1))
    vis_img, status = update_view(new_idx, file_list, all_anns, current_obj_id)
    return vis_img, new_idx, status

def on_image_click(evt: gr.SelectData, current_idx, file_list, all_anns, mode, obj_id, click_state):
    if not file_list: return None, all_anns, click_state
    x, y = evt.index[0], evt.index[1]
    
    if current_idx not in all_anns: all_anns[current_idx] = {}
    if obj_id not in all_anns[current_idx]:
        all_anns[current_idx][obj_id] = {'box': None, 'points': [], 'labels': []}
    
    data = all_anns[current_idx][obj_id]
    
    if mode == "Point (Positive)":
        data['points'].append([x, y])
        data['labels'].append(1) # Label 1
        click_state = None
    elif mode == "Point (Negative)":
        data['points'].append([x, y])
        data['labels'].append(0) # Label 0
        click_state = None
    elif mode == "Box":
        if click_state is None: click_state = (x, y)
        else:
            x1, y1 = click_state
            data['box'] = [sorted([x1, x])[0], sorted([y1, y])[0], sorted([x1, x])[1], sorted([y1, y])[1]]
            click_state = None
            
    vis_img, _ = update_view(current_idx, file_list, all_anns, obj_id)
    return vis_img, all_anns, click_state

def clear_current_annotation(current_idx, file_list, all_anns, obj_id):
    if current_idx in all_anns and obj_id in all_anns[current_idx]:
        del all_anns[current_idx][obj_id]
    vis_img, _ = update_view(current_idx, file_list, all_anns, obj_id)
    return vis_img, all_anns, None

# ==========================================
# [Gradio Interface Build]
# ==========================================
with gr.Blocks(title="SAM 3 Batch Labeler") as app:
    state_ref_files = gr.State([])
    state_current_idx = gr.State(0)
    state_all_annotations = gr.State({})
    state_click_temp = gr.State(None)

    gr.Markdown("## SAM 3 Interactive Labeler")
    
    with gr.Row():
        btn_load = gr.Button("Step 0: Load Model", variant="secondary")
        lbl_system = gr.Textbox(label="System Status", value="Model Not Loaded", interactive=False)

    gr.Markdown("---")
    with gr.Row():
        with gr.Column(scale=4):
            gr.Markdown("### Step 1: Reference Annotation")
            files_ref_input = gr.File(label="Upload References", file_count="multiple", type="filepath")
            
            with gr.Row():
                btn_prev = gr.Button("◀")
                lbl_idx = gr.Textbox(value="0/0", show_label=False, container=False, interactive=False, scale=0)
                btn_next = gr.Button("▶")
            
            img_view = gr.Image(label="Annotate Here", interactive=False, height=500)
            
            with gr.Row():
                num_id = gr.Number(label="Object ID", value=1, minimum=1, precision=0, step=1)
                radio_tool = gr.Radio(["Box", "Point (Positive)", "Point (Negative)"], label="Tool", value="Box")
                btn_clear = gr.Button("Clear ID")
            
            btn_gen_mem = gr.Button("2. GENERATE MEMORY BANK", variant="primary")
            lbl_mem_status = gr.Textbox(label="Memory Status", interactive=False)

        with gr.Column(scale=6):
            gr.Markdown("### Step 2: Target Inference & Results")
            files_target_input = gr.File(label="Upload Targets", file_count="multiple", type="filepath")
            btn_run_inf = gr.Button("3. RUN INFERENCE", variant="primary", size="lg")
            lbl_inf_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Tabs():
                with gr.TabItem("Overlay Results"):
                    gallery_overlay = gr.Gallery(label="Overlays", columns=3, object_fit="contain")
                
                with gr.TabItem("Mask Results (Binary)"):
                    gallery_mask = gr.Gallery(label="Masks", columns=3, object_fit="contain")
            
            gr.Markdown("### 4. Download")
            btn_zip = gr.Button("Download All Results (ZIP)")
            file_zip_output = gr.File(label="Download ZIP")

    btn_load.click(sam_engine.load_model, outputs=lbl_system)
    
    files_ref_input.change(on_files_upload, files_ref_input, [state_current_idx, state_ref_files, state_all_annotations, state_click_temp, lbl_system]) \
                   .then(update_view, [state_current_idx, state_ref_files, state_all_annotations, num_id], [img_view, lbl_idx])

    btn_prev.click(fn=lambda i, f, a, o: navigate(-1, i, f, a, o), inputs=[state_current_idx, state_ref_files, state_all_annotations, num_id], outputs=[img_view, state_current_idx, lbl_idx])
    btn_next.click(fn=lambda i, f, a, o: navigate(1, i, f, a, o), inputs=[state_current_idx, state_ref_files, state_all_annotations, num_id], outputs=[img_view, state_current_idx, lbl_idx])
    
    img_view.select(on_image_click, [state_current_idx, state_ref_files, state_all_annotations, radio_tool, num_id, state_click_temp], [img_view, state_all_annotations, state_click_temp])
    btn_clear.click(clear_current_annotation, [state_current_idx, state_ref_files, state_all_annotations, num_id], [img_view, state_all_annotations, state_click_temp])

    btn_gen_mem.click(sam_engine.generate_memory_bank, inputs=[state_ref_files, state_all_annotations], outputs=lbl_mem_status)
    
    btn_run_inf.click(
        fn=sam_engine.run_target_inference,
        inputs=[files_target_input],
        outputs=[lbl_inf_status, gallery_overlay, gallery_mask]
    )

    btn_zip.click(sam_engine.zip_results, inputs=[], outputs=[file_zip_output])

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=True)