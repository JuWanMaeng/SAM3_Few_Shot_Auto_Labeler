import gradio as gr
import numpy as np
import cv2
import os

# --- 한글 경로 지원을 위한 헬퍼 함수 ---
def imread_korean(file_path):
    """한글 경로 이미지 읽기"""
    stream = open(file_path.encode("utf-8"), "rb")
    bytes = bytearray(stream.read())
    numpy_array = np.asarray(bytes, dtype=np.uint8)
    return cv2.imdecode(numpy_array, cv2.IMREAD_COLOR)

def imwrite_korean(filename, img, params=None):
    """한글 경로 이미지 저장"""
    try:
        ext = os.path.splitext(filename)[1]
        result, n = cv2.imencode(ext, img, params)
        if result:
            with open(filename, mode='w+b') as f:
                n.tofile(f)
            return True
        return False
    except Exception as e:
        print(f"저장 실패: {e}")
        return False

# ----------------------------------------

def apply_mask_to_editor(image_path, mask_image):
    """
    [마스크 적용] 버튼 로직
    - image_path: 원본 이미지 경로 (str)
    - mask_image: 마스크 이미지 (numpy)
    """
    if image_path is None:
        return None, None, None

    # 1. 파일 경로에서 이미지 읽어오기 (BGR -> RGB 변환 필수)
    original_image = imread_korean(image_path)
    if original_image is None:
        print("이미지를 읽을 수 없습니다.")
        return None, None, None
    
    # Gradio Editor는 RGB를 원하므로 변환
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # 2. 마스크가 없으면 원본만 리턴
    if mask_image is None:
        return {
            "background": original_image,
            "layers": [],
            "composite": original_image
        }, None, image_path

    # 3. 마스크 크기 맞추기
    if original_image.shape[:2] != mask_image.shape[:2]:
        mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 4. 마스크 흑백 변환
    if len(mask_image.shape) == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask_image

    # --- "찍어주기" 로직 ---
    h, w = mask_gray.shape
    rgba_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    # 흰색 부분 칠하기
    mask_indices = mask_gray > 0
    rgba_layer[mask_indices] = [255, 255, 255, 150]

    # Visual용 합성 이미지
    composite_img = original_image.copy()
    overlay = np.full_like(original_image, 255)
    alpha = 0.5
    blended = cv2.addWeighted(original_image, 1-alpha, overlay, alpha, 0)
    composite_img[mask_indices] = blended[mask_indices]

    # 백업 마스크
    backup_mask = np.zeros_like(mask_gray)
    backup_mask[mask_indices] = 255

    print("👉 마스크 레이어를 에디터에 적용했습니다.")

    # [중요] image_path를 State에 저장하기 위해 함께 반환
    return {
        "background": original_image,
        "layers": [rgba_layer],
        "composite": composite_img
    }, backup_mask, image_path


def save_result(editor_content, backup_mask, original_path):
    """
    [저장] 버튼 로직
    - 원본 파일명과 동일하게 PNG로 저장
    """
    # 1. 결과 마스크 생성 로직
    final_mask = None
    
    # 에디터 내용 확인
    if editor_content is not None and editor_content.get("layers"):
        print("💾 편집 내용 반영하여 저장 중...")
        layers = editor_content.get("layers", [])
        h, w = layers[0].shape[:2]
        final_mask = np.zeros((h, w), dtype=np.uint8)
        for layer in layers:
            active = layer[:, :, 3] > 0
            final_mask[active] = 255
    else:
        print("💾 편집 내용 없음 -> 백업본 사용")
        final_mask = backup_mask

    if final_mask is None:
        print("❌ 저장할 마스크 데이터가 없습니다.")
        return None

    # 2. 파일 저장 로직
    if original_path:
        # 경로에서 파일명만 추출 (예: C:/img/test.jpg -> test.jpg)
        filename = os.path.basename(original_path)
        # 확장자 제거 (test.jpg -> test)
        name_only = os.path.splitext(filename)[0]
        # png 확장자 붙이기
        save_name = f"data/masks/{name_only}.png"
        
        # (옵션) 'result' 폴더에 따로 저장하려면 아래 주석 해제
        # if not os.path.exists("result"): os.makedirs("result")
        # save_path = os.path.join("result", save_name)
        
        # 현재 경로에 저장 (원본과 같은 폴더가 아니라 실행 파일 위치)
        save_path = save_name 

        # 저장 실행 (한글 경로 대응)
        imwrite_korean(save_path, final_mask)
        print(f"✅ 저장 완료: {save_path}")
    else:
        print("⚠️ 원본 경로를 찾을 수 없어 'result.png'로 저장합니다.")
        imwrite_korean("result.png", final_mask)

    return final_mask


# --- UI ---
with gr.Blocks() as demo:
    # 데이터 보관소
    state_backup = gr.State()      # 백업 마스크 저장
    state_filepath = gr.State()    # 원본 파일 경로 저장

    gr.Markdown("## 🖌️ 버튼으로 마스크 찍어주기 (파일명 유지 저장)")
    
    with gr.Row():
        with gr.Column(scale=1):
            # [수정됨] type="filepath"로 변경하여 경로를 받아옴
            img_in = gr.Image(label="1. 원본 이미지", type="filepath") 
            mask_in = gr.Image(label="2. 마스크 이미지", type="numpy")
            
            btn_apply = gr.Button("👉 3. 마스크 작업영역에 찍기", variant="primary")
        
        with gr.Column(scale=4):
            editor = gr.ImageEditor(
                label="4. 작업 영역",
                type="numpy",
                brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
                eraser=gr.Eraser(default_size=20),
                interactive=True,
                height=600
            )
            btn_save = gr.Button("✅ 5. 결과 저장 (PNG)", variant="secondary")

    out = gr.Image(label="6. 최종 결과", type="numpy")

    # [적용 버튼] 
    # outputs에 state_filepath 추가 (경로 저장을 위해)
    btn_apply.click(
        fn=apply_mask_to_editor,
        inputs=[img_in, mask_in],
        outputs=[editor, state_backup, state_filepath]
    )

    # [저장 버튼]
    # inputs에 state_filepath 추가 (저장할 때 이름 알기 위해)
    btn_save.click(
        fn=save_result,
        inputs=[editor, state_backup, state_filepath],
        outputs=out
    )

if __name__ == "__main__":
    demo.launch(share=False)