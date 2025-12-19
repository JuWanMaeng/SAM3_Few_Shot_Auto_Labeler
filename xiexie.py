import gradio as gr
import numpy as np
import cv2

def apply_mask_to_editor(original_image, mask_image):
    """
    [마스크 적용] 버튼을 눌렀을 때 실행되는 함수
    1. 마스크의 흰색 부분을 찾음
    2. 투명 레이어에 흰색을 칠함
    3. 에디터에 강제로 밀어넣음
    """
    if original_image is None:
        return None, None

    # 1. 마스크가 없으면 원본만 에디터에 보냄 (레이어 없음)
    if mask_image is None:
        return {
            "background": original_image,
            "layers": [],
            "composite": original_image
        }, None

    # 2. 마스크 크기 맞추기 (안전장치)
    if original_image.shape[:2] != mask_image.shape[:2]:
        mask_image = cv2.resize(mask_image, (original_image.shape[1], original_image.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 3. 마스크 흑백 변환
    if len(mask_image.shape) == 3:
        mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_RGB2GRAY)
    else:
        mask_gray = mask_image

    # --- 여기서부터 "찍어주기" 로직 ---
    
    # (1) 투명 레이어(RGBA) 만들기
    h, w = mask_gray.shape
    rgba_layer = np.zeros((h, w, 4), dtype=np.uint8)
    
    # (2) 흰색 부분(0보다 큰 부분) 찾아서 칠하기
    # 눈에 잘 보이게 흰색(255) + 불투명도(150)
    mask_indices = mask_gray > 0
    rgba_layer[mask_indices] = [255, 255, 255, 150]

    # (3) 화면에 보여줄 합성 이미지 미리 만들기 (이게 있어야 바로 보임)
    composite_img = original_image.copy()
    
    # 원본 위에 흰색 살짝 섞어서 보여줌 (Visual용)
    overlay = np.full_like(original_image, 255)
    alpha = 0.5
    blended = cv2.addWeighted(original_image, 1-alpha, overlay, alpha, 0)
    composite_img[mask_indices] = blended[mask_indices]

    # (4) 백업용 마스크 (저장 오류 방지용)
    backup_mask = np.zeros_like(mask_gray)
    backup_mask[mask_indices] = 255

    print("👉 마스크 레이어를 에디터에 적용했습니다.")

    return {
        "background": original_image,
        "layers": [rgba_layer],   # 편집 가능한 레이어
        "composite": composite_img # 눈에 보이는 이미지
    }, backup_mask


def save_result(editor_content, backup_mask):
    """
    저장 버튼 로직:
    - 에디터가 비어있으면(편집 안함) -> 백업본 리턴
    - 에디터가 있으면 -> 편집본 리턴
    """
    # 1. 에디터 데이터 확인
    if editor_content is None:
        return backup_mask

    layers = editor_content.get("layers", [])
    
    # 2. 레이어 확인
    # 편집을 안 했거나 로딩 오류시 layers가 비어있을 수 있음
    if not layers:
        print("💾 편집 내용 없음 -> 원본 마스크 저장")
        return backup_mask

    # 3. 레이어 합치기 (편집 내용 반영)
    print("💾 편집 내용 있음 -> 수정본 저장")
    h, w = layers[0].shape[:2]
    final_mask = np.zeros((h, w), dtype=np.uint8)

    for layer in layers:
        # 투명도가 있는 부분을 마스크로 인식
        active = layer[:, :, 3] > 0
        final_mask[active] = 255
    
    # 만약 합쳤는데 검은색이다? (지우개로 다 지웠거나 오류) -> 사용자가 다 지운걸 수도 있으니 그대로 반환
    # 단, 사용자가 "아무것도 안 건드린" 경우를 위해 백업본 로직이 필요하다면 아래 주석 해제
    # if np.max(final_mask) == 0 and backup_mask is not None: return backup_mask

    return final_mask


# --- UI ---
with gr.Blocks() as demo:
    # 혹시 모를 상황 대비용 백업 저장소
    state_backup = gr.State()

    gr.Markdown("## 🖌️ 버튼으로 마스크 찍어주기")
    
    with gr.Row():
        # 왼쪽: 입력창
        with gr.Column(scale=1):
            img_in = gr.Image(label="1. 원본 이미지", type="numpy")
            mask_in = gr.Image(label="2. 마스크 이미지", type="numpy")
            
            # 님 아이디어: 마스크 적용 버튼을 따로 뺌
            btn_apply = gr.Button("👉 3. 마스크 작업영역에 찍기", variant="primary")
        
        # 오른쪽: 작업창
        with gr.Column(scale=4):
            editor = gr.ImageEditor(
                label="4. 작업 영역 (여기 마스크가 뜸)",
                type="numpy",
                brush=gr.Brush(colors=["#FFFFFF"], default_size=20),
                eraser=gr.Eraser(default_size=20),
                interactive=True,
                height=600
            )
            btn_save = gr.Button("✅ 5. 결과 저장", variant="secondary")

    out = gr.Image(label="6. 최종 결과", type="numpy")

    # [버튼 클릭] -> 마스크를 에디터 레이어로 변환해서 넣어줌 (+백업)
    btn_apply.click(
        fn=apply_mask_to_editor,
        inputs=[img_in, mask_in],
        outputs=[editor, state_backup]
    )

    # [저장 클릭] -> 에디터 내용 혹은 백업본을 저장
    btn_save.click(
        fn=save_result,
        inputs=[editor, state_backup],
        outputs=out
    )

if __name__ == "__main__":
    demo.launch(share=True)