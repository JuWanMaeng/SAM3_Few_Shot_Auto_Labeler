import os
import numpy as np
import cv2
import PIL.Image
import gradio as gr

import sam3
from sam3 import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ==========================================
# 0. 설정
# ==========================================
WORK_SIZE = (1024, 1024)
OUTPUT_ROOT = "outputs"  # 결과물이 저장될 로컬 폴더명

# 폴더가 없으면 생성
os.makedirs(OUTPUT_ROOT, exist_ok=True)
print(f"📂 Save directory: {os.path.abspath(OUTPUT_ROOT)}")

# ==========================================
# 1. SAM3 모델 로드
# ==========================================
sam3_root = os.path.join(os.path.dirname(sam3.__file__), "..")
bpe_path = f"{sam3_root}/assets/bpe_simple_vocab_16e6.txt.gz"
print("⏳ Loading SAM3 Model...")
model = build_sam3_image_model(
    bpe_path=bpe_path,
    checkpoint_path="models--facebook--sam3/snapshots/3c879f39826c281e95690f02c7821c4de09afae7/sam3.pt",
    enable_inst_interactivity=True,
)
processor = Sam3Processor(model)
print("✅ Model Loaded.")

# ==========================================
# 2. 헬퍼 함수들
# ==========================================
def resize_with_padding(pil_img, target_size):
    w, h = pil_img.size
    target_w, target_h = target_size
    scale = min(target_w / w, target_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    
    resized_img = pil_img.resize((new_w, new_h), PIL.Image.Resampling.LANCZOS)
    new_img = PIL.Image.new("RGB", target_size, (0, 0, 0))
    
    pad_w = (target_w - new_w) // 2
    pad_h = (target_h - new_h) // 2
    new_img.paste(resized_img, (pad_w, pad_h))
    
    return new_img, (scale, pad_w, pad_h)

def draw_points_on_image(pil_img, points, labels):
    if pil_img is None: return None
    img_arr = np.array(pil_img.convert("RGB"))
    img_bgr = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)

    for point, label in zip(points, labels):
        x, y = int(point[0]), int(point[1])
        if label == 1:
            color = (0, 255, 0)
            pts = np.array([[x, y], [x-6, y-12], [x+6, y-12]], np.int32).reshape((-1, 1, 2))
            cv2.drawContours(img_bgr, [pts], 0, color, -1)
            cv2.drawContours(img_bgr, [pts], 0, (0,0,0), 1)
        else:
            color = (0, 0, 255)
            cv2.line(img_bgr, (x-5, y-5), (x+5, y+5), color, 2)
            cv2.line(img_bgr, (x+5, y-5), (x-5, y+5), color, 2)
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# ==========================================
# 3. 로직 함수들
# ==========================================

def load_image(files, idx):
    empty_res = (None, None, None, [], [], 0, "No files", None, None, None, None, "", [])
    if not files: return empty_res
    
    idx = max(0, min(idx, len(files)-1))
    file_obj = files[idx]
    path = file_obj.name if hasattr(file_obj, 'name') else file_obj
    filename_str = os.path.basename(path)
    
    img_orig = PIL.Image.open(path).convert("RGB")
    img_work, resize_info = resize_with_padding(img_orig, WORK_SIZE)
    status_msg = f"Current Image: [ {idx + 1} / {len(files)} ]\nFile: {filename_str}\nOriginal: {img_orig.size}"
    
    return (
        img_work, img_orig, img_work, [], [], idx, status_msg, 
        None, None, None, resize_info, filename_str, files
    )

def run_inference(orig_image, points_1024, labels, resize_info, current_fname):
    if orig_image is None or not points_1024: 
        return None, None, "No info.", None, None, None, ""
    
    scale, pad_w, pad_h = resize_info
    points_orig = []
    for p in points_1024:
        px = (p[0] - pad_w) / scale
        py = (p[1] - pad_h) / scale
        points_orig.append([px, py])
    
    pil_orig = orig_image.convert("RGB")
    w_orig, h_orig = pil_orig.size
    inference_state = processor.set_image(pil_orig)
    
    masks, scores, _ = model.predict_inst(
        inference_state, 
        point_coords=np.array(points_orig), 
        point_labels=np.array(labels), 
        multimask_output=True,
    )
    
    best_idx = np.argmax(scores)
    current_mask = masks[best_idx] > 0
    mask_pixels = int(np.sum(current_mask))
    coverage = (mask_pixels / (w_orig * h_orig)) * 100.0
    
    mask_uint8 = (current_mask * 255).astype(np.uint8)
    overlay_arr = np.array(pil_orig)
    overlay_arr[current_mask] = (overlay_arr[current_mask] * 0.5 + np.array([255, 0, 0]) * 0.5).astype(np.uint8)
    
    info_text = f"File: {current_fname}\nScore: {float(scores[best_idx]):.3f}\nArea: {coverage:.2f}%"
    
    return mask_uint8, overlay_arr, info_text, mask_uint8, overlay_arr, coverage, current_fname

# ==========================================
# [중요] 저장 함수 (슬롯 번호 받음)
# ==========================================
def get_safe_name(fname):
    if not fname: return "result", "result.png"
    return os.path.splitext(fname)[0], fname

def save_m_local(mask, fname, slot_num):
    if mask is None: 
        gr.Warning("No result to save.")
        return
    base, _ = get_safe_name(fname)
    # 파일명 뒤에 _1, _2 등을 붙임
    path = os.path.join(OUTPUT_ROOT, f"{base}_mask_{slot_num}.png")
    cv2.imwrite(path, mask)
    gr.Info(f"✅ Slot {slot_num}: Mask Saved ({os.path.basename(path)})")

def save_o_local(over, fname, slot_num):
    if over is None: 
        gr.Warning("No result to save.")
        return
    base, _ = get_safe_name(fname)
    path = os.path.join(OUTPUT_ROOT, f"{base}_overlay_{slot_num}.png")
    cv2.imwrite(path, cv2.cvtColor(over, cv2.COLOR_RGB2BGR))
    gr.Info(f"✅ Slot {slot_num}: Overlay Saved ({os.path.basename(path)})")

def save_l_local(cov, fname, slot_num):
    if cov is None: return
    base, realname = get_safe_name(fname)
    path = os.path.join(OUTPUT_ROOT, f"{base}_log_{slot_num}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"File: {realname}\nSlot: {slot_num}\nCoverage: {cov:.2f}%")
    gr.Info(f"✅ Slot {slot_num}: Log Saved")

# UI 인터랙션 함수들
def on_select(evt: gr.SelectData, img, pts, lbls, mode):
    if img is None: return pts, lbls, None
    x, y = evt.index
    x = max(0, min(x, 1023)); y = max(0, min(y, 1023))
    lbl = 0 if mode == "Negative" else 1
    pts.append([x, y]); lbls.append(lbl)
    return pts, lbls, draw_points_on_image(img, pts, lbls)

def undo(img, pts, lbls):
    if pts: pts.pop(); lbls.pop()
    res = img if not pts else draw_points_on_image(img, pts, lbls)
    return pts, lbls, res

def reset(img): return [], [], img

# ==========================================
# 4. UI 구성
# ==========================================
with gr.Blocks(title="SAM3 Multi-Slot Tool") as demo:
    # ---------------------------------------------------------
    # [FIX] State 정의를 가장 위로 올림 (UI에서 참조하기 위해)
    # ---------------------------------------------------------
    s_files = gr.State([])
    s_idx = gr.State(0)
    s_pts = gr.State([])
    s_lbls = gr.State([])
    s_orig = gr.State(None)
    s_work = gr.State(None)
    s_resize = gr.State(None)
    
    s_fname_load = gr.State("") 
    s_fname_save = gr.State("") # Run 클릭 시 파일명 저장
    
    s_res_mask = gr.State(None)
    s_res_over = gr.State(None)
    s_res_cov = gr.State(None)
    # ---------------------------------------------------------

    gr.Markdown(f"### 🖼️ SAM3 Multi-Slot Labeling")
    gr.Markdown(f"**Save Location:** `{os.path.abspath(OUTPUT_ROOT)}`")
    
    with gr.Row():
        with gr.Column(scale=1):
            files = gr.File(file_count="multiple", label="Upload Folder")
            file_status = gr.Textbox(label="Status", interactive=False)
            with gr.Row():
                prev_btn = gr.Button("◀ Prev")
                next_btn = gr.Button("Next ▶")
            
            mode_radio = gr.Radio(["Positive", "Negative"], value="Positive", label="Mode")
            with gr.Row():
                undo_btn = gr.Button("Undo")
                reset_btn = gr.Button("Reset")
            
            # 추론 버튼만 존재 (자동 저장 안 함)
            run_btn = gr.Button("1. Run Segment", variant="primary")
            
        with gr.Column(scale=4): 
            img_in = gr.Image(type="pil", interactive=True, label="Work Area", width=1024, height=1024)
            log_disp = gr.TextArea(label="Session Log (Read Only)", interactive=False, lines=5)

        with gr.Column(scale=4): 
            mask_out = gr.Image(label="Mask")
            over_out = gr.Image(label="Overlay")
            info_out = gr.Textbox(label="Info")
            
            gr.Markdown("### 💾 Save to Slots (Result required)")
            # 4개의 슬롯(행) 생성
            for i in range(1, 5):
                with gr.Row():
                    gr.Markdown(f"**Slot {i}**")
                    btn_m = gr.Button(f"Mask {i}")
                    btn_o = gr.Button(f"Overlay {i}")
                    btn_l = gr.Button(f"Log {i}")
                    
                    # 이제 s_fname_save 등이 이미 정의되어 있으므로 에러가 나지 않습니다.
                    btn_m.click(lambda m, f, s=i: save_m_local(m, f, s), [mask_out, s_fname_save], None)
                    btn_o.click(lambda o, f, s=i: save_o_local(o, f, s), [over_out, s_fname_save], None)
                    btn_l.click(lambda c, f, s=i: save_l_local(c, f, s), [s_res_cov, s_fname_save], None)

    # --- EVENTS ---
    outs_load = [img_in, s_orig, s_work, s_pts, s_lbls, s_idx, file_status, s_res_mask, s_res_over, s_res_cov, s_resize, s_fname_load, s_files]

    files.upload(lambda f: load_image(f, 0), [files], outs_load)
    prev_btn.click(lambda f, i: load_image(f, i-1), [s_files, s_idx], outs_load)
    next_btn.click(lambda f, i: load_image(f, i+1), [s_files, s_idx], outs_load)

    img_in.select(on_select, [s_work, s_pts, s_lbls, mode_radio], [s_pts, s_lbls, img_in])
    undo_btn.click(undo, [s_work, s_pts, s_lbls], [s_pts, s_lbls, img_in])
    reset_btn.click(reset, [s_work], [s_pts, s_lbls, img_in])

    run_btn.click(
        run_inference,
        [s_orig, s_pts, s_lbls, s_resize, s_fname_load],
        [mask_out, over_out, info_out, s_res_mask, s_res_over, s_res_cov, s_fname_save]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)