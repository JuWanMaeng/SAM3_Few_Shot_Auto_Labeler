import os
import numpy as np
import cv2
import PIL.Image
import gradio as gr

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor
from collections import OrderedDict

# ==========================================
# 0. 설정 및 모델 로드
# ==========================================
WORK_SIZE = (1024, 1024)
OUTPUT_ROOT = "outputs"
os.makedirs(OUTPUT_ROOT, exist_ok=True)

sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
model = build_sam3_image_model(
    bpe_path=bpe_path,
    checkpoint_path="models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt",
    enable_inst_interactivity=True,
)
processor = Sam3Processor(model)

# ==========================================
# 1. 헬퍼 함수
# ==========================================
def resize_with_padding(pil_img, target_size):
    w, h = pil_img.size
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized_img = pil_img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
    new_img = PIL.Image.new("RGB", target_size, (0, 0, 0))
    pad_w, pad_h = (target_w - new_w) // 2, (target_h - new_h) // 2
    new_img.paste(resized_img, (pad_w, pad_h))
    return new_img, (scale, pad_w, pad_h)

def draw_points_on_image(pil_img, points, labels):
    if pil_img is None: return None
    img_arr = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
    for point, label in zip(points, labels):
        x, y = int(point[0]), int(point[1])
        color = (0, 255, 0) if label == 1 else (0, 0, 255)
        if label == 1:
            cv2.circle(img_bgr, (x, y), 8, color, -1)
            cv2.circle(img_bgr, (x, y), 8, (0, 0, 0), 1)
        else:
            cv2.line(img_bgr, (x-5, y-5), (x+5, y+5), color, 2)
            cv2.line(img_bgr, (x+5, y-5), (x-5, y+5), color, 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def format_status_list(files, seg_status, current_idx):
    """파일 목록과 상태 딕셔너리를 받아서 스크롤 가능한 텍스트로 포맷"""
    if not files:
        return "No files loaded."
        
    lines = []
    
    # Ordered dict를 사용하여 순서를 유지하거나, files 순서를 따름
    filenames = [os.path.basename(f.name) for f in files]
    
    for i, filename in enumerate(filenames):
        status_icon = seg_status.get(filename, "⚫") # '⚫ Ready' 대신 '⚫'만 사용해 간결하게
        
        # 현재 파일이면 강조 표시
        prefix = "-> " if i == current_idx else "   "
        
        lines.append(f"{prefix}[{status_icon}] {filename}")
        
    return "\n".join(lines)

# ==========================================
# 2. 로직 함수
# ==========================================
def on_select(evt: gr.SelectData, img, pts, lbls, mode):
    if img is None: return pts, lbls, None
    x, y = evt.index
    pts.append([x, y])
    lbls.append(1 if mode == "🟢Positive🟢" else 0)
    return pts, lbls, draw_points_on_image(img, pts, lbls)

def on_undo(img, pts, lbls):
    if not pts: return pts, lbls, img
    pts.pop()
    lbls.pop()
    # 복구된 이미지를 다시 그리는 함수 호출
    res = img if not pts else draw_points_on_image(img, pts, lbls)
    return pts, lbls, res

def load_image(files, idx, seg_status):
    if not files: 
        return [None]*10 + ["", [], "No files", "⚫ Ready", {}, "No files loaded."]
        
    idx = max(0, min(idx, len(files)-1))
    file_obj = files[idx]
    path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    filename = os.path.basename(path)
    
    img_orig = PIL.Image.open(path).convert("RGB")
    img_work, resize_info = resize_with_padding(img_orig, WORK_SIZE)
    status_msg = f"File: {filename} ({idx+1}/{len(files)})"
    
    # 📌 이미지별 상태 확인 및 설정
    current_status = seg_status.get(filename, "⚫ Ready")
    
    # 📌 전체 상태 목록 업데이트
    status_list = format_status_list(files, seg_status, idx)

    return (
        img_work, img_orig, img_work, [], [], idx, 
        status_msg, None, None, None, resize_info, filename, files, 
        current_status, seg_status, status_list # 상태 텍스트, 딕셔너리, 목록 반환
    )

def run_inference(orig_image, points_1024, labels, resize_info, current_fname, seg_status, files, current_idx):
    if orig_image is None or not points_1024: 
        return None, None, "No points.", None, None, None, "", "⚫ Ready", seg_status, format_status_list(files, seg_status, current_idx)
    
    scale, pad_w, pad_h = resize_info
    points_orig = [[(p[0] - pad_w) / scale, (p[1] - pad_h) / scale] for p in points_1024]
    
    inference_state = processor.set_image(orig_image)
    masks, scores, _ = model.predict_inst(
        inference_state, 
        point_coords=np.array(points_orig), 
        point_labels=np.array(labels), 
        multimask_output=True,
    )
    
    best_idx = np.argmax(scores)
    mask = masks[best_idx] > 0
    coverage = (np.sum(mask) / (orig_image.size[0] * orig_image.size[1])) * 100.0
    
    mask_uint8 = (mask * 255).astype(np.uint8)
    overlay = np.array(orig_image)
    overlay[mask] = (overlay[mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    
    info_text = f"Score: {scores[best_idx]:.3f}\nArea: {coverage:.2f}%"
    
    # 📌 상태 딕셔너리 업데이트
    seg_status[current_fname] = "🟢 Done"
    
    # 📌 전체 상태 목록 업데이트
    status_list = format_status_list(files, seg_status, current_idx)
    
    return mask_uint8, overlay, info_text, mask_uint8, overlay, coverage, current_fname, "🟢 Done", seg_status, status_list

def reset_points_and_status(img, current_fname, seg_status, files, current_idx):
    # 📌 상태 딕셔너리 업데이트
    if current_fname in seg_status:
        seg_status[current_fname] = "⚫ Ready"
    
    # 📌 전체 상태 목록 업데이트
    status_list = format_status_list(files, seg_status, current_idx)
    
    # 포인트와 라벨, 이미지를 리셋
    return [], [], img, "⚫ Ready", seg_status, status_list

def save_all_local(mask, over, cov, fname, slot, seg_status, files, current_idx):
    if mask is None: 
        gr.Warning("No result to save.")
        return "Save Failed", seg_status, format_status_list(files, seg_status, current_idx)

    base = os.path.splitext(fname)[0] if fname else "result"
    
    # 저장 로직
    mask_path = os.path.join(OUTPUT_ROOT, f"{base}_mask_{slot}.png")
    overlay_path = os.path.join(OUTPUT_ROOT, f"{base}_overlay_{slot}.png")
    log_path = os.path.join(OUTPUT_ROOT, f"{base}_log_{slot}.txt")

    cv2.imwrite(mask_path, mask)
    cv2.imwrite(overlay_path, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    with open(log_path, "w") as f:
        f.write(f"File: {fname}\nSlot: {slot}\nCoverage: {cov:.2f}%")
    
    save_msg = f"✅ Slot {slot} Saved: {os.path.basename(mask_path)}"
    
    # 📌 저장 알림(Toast) 추가
    gr.Info(f"Saved to Slot {slot}!")
        
    return save_msg, seg_status, format_status_list(files, seg_status, current_idx)

# ==========================================
# 3. UI 구성
# ==========================================
with gr.Blocks(title="SAM3 Multi-Slot Tool") as demo:
    s_files, s_idx = gr.State([]), gr.State(0)
    s_pts, s_lbls = gr.State([]), gr.State([])
    s_orig, s_work, s_resize = gr.State(None), gr.State(None), gr.State(None)
    s_fname_load, s_fname_save = gr.State(""), gr.State("")
    s_res_mask, s_res_over, s_res_cov = gr.State(None), gr.State(None), gr.State(None)
    s_seg_status = gr.State({})

    gr.Markdown("### 🖼️ SAM3 Point Labeling Tool (Per-Image Status)")
    
    with gr.Row():
        with gr.Column(scale=3):
            # 📌 파일 업로드 및 전체 상태 표시 리스트 (좌측 상단)
            with gr.Row():
                files = gr.File(file_count="multiple", label="Upload Folder", scale=4, height=200)
                image_indicator = gr.Textbox(label="Seg Status", value="⚫ Ready", interactive=False, scale=1)
            
            # 📌 전체 파일 상태를 한눈에 볼 수 있는 스크롤 박스
            status_list_box = gr.Textbox(
                label="📁 File Status List",
                interactive=False, 
                lines=8, 
                value="Upload files to begin.",
                max_lines=15
            )
            
            file_status = gr.Textbox(label="Current Action Status", interactive=False, lines=5)

            with gr.Row():
                prev_btn = gr.Button("◀ Prev")
                next_btn = gr.Button("Next ▶")

            mode_radio = gr.Radio(["🟢Positive🟢", "❌Negative❌"], value="🟢Positive🟢", label="Mode")
            run_btn = gr.Button("❤️Run Segment❤️", variant="primary")

            with gr.Row():
                undo_btn = gr.Button("⏱️Undo⏱️")
                reset_btn = gr.Button("🚨Reset🚨", variant='stop')
            
        with gr.Column(scale=5): 
            img_in = gr.Image(type="pil", interactive=True, label="Work Area", width=1024, height=1024)

        with gr.Column(scale=3): 
            with gr.Row():
                mask_out = gr.Image(label="Mask", height=300)
                over_out = gr.Image(label="Overlay", height=300)
            info_out = gr.Textbox(label="Inference Info", lines=3)
            
            gr.Markdown("### 💾 Save to Slots")
            for i in range(1, 5):
                btn = gr.Button(f"Save All to Slot {i}")
                btn.click(
                    save_all_local, 
                    [s_res_mask, s_res_over, s_res_cov, s_fname_save, gr.State(i), s_seg_status, s_files, s_idx], 
                    [file_status, s_seg_status, status_list_box]
                )

    # --- 이벤트 핸들링 ---
    # load_image의 새로운 출력 리스트 (image_indicator, s_seg_status, status_list_box 추가됨)
    load_outputs = [img_in, s_orig, s_work, s_pts, s_lbls, s_idx, file_status, s_res_mask, s_res_over, s_res_cov, s_resize, s_fname_load, s_files, image_indicator, s_seg_status, status_list_box]

    # 1. 파일 업로드 (초기화)
    files.upload(lambda f: load_image(f, 0, {}), [files], load_outputs)
    
    # 2. 내비게이션
    prev_btn.click(lambda f, i, ss: load_image(f, i - 1, ss), [s_files, s_idx, s_seg_status], load_outputs)
    next_btn.click(lambda f, i, ss: load_image(f, i + 1, ss), [s_files, s_idx, s_seg_status], load_outputs)

    # 3. 점 찍기 및 취소
    img_in.select(on_select, [s_work, s_pts, s_lbls, mode_radio], [s_pts, s_lbls, img_in])
    undo_btn.click(on_undo, [s_work, s_pts, s_lbls], [s_pts, s_lbls, img_in])
    
    # 4. 리셋 (파일별 상태 및 리스트 업데이트)
    reset_btn.click(
        reset_points_and_status, 
        [s_work, s_fname_load, s_seg_status, s_files, s_idx], 
        [s_pts, s_lbls, img_in, image_indicator, s_seg_status, status_list_box]
    )

    # 5. 실행 (파일별 상태 및 리스트 업데이트)
    run_btn.click(
        run_inference,
        [s_orig, s_pts, s_lbls, s_resize, s_fname_load, s_seg_status, s_files, s_idx],
        [mask_out, over_out, info_out, s_res_mask, s_res_over, s_res_cov, s_fname_save, image_indicator, s_seg_status, status_list_box]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)