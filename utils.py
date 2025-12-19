# utils.py
import os
import cv2
import numpy as np
from PIL import Image
import config  # config.py에서 설정값 가져오기

class Visualizer:
    @staticmethod
    def save_visualizations(img_pil, detected_objects, final_mask, reverse_mask, base_name, size):
        """
        검출 결과(Mask, Overlay, BBox)를 저장하는 메인 함수
        """
        # 1. ID Mask & Overlay
        vis_overlay = Visualizer.create_mask_overlay(img_pil, final_mask, color=(255, 0, 0))
        # vis_id = Visualizer.create_overlay_with_id(img_pil, detected_objects, size)
        # vis_bbox = Visualizer.create_bbox_overlay(img_pil, final_mask, color=(0, 255, 0))
        
        vis_overlay.save(os.path.join(config.OVERLAY_OUTPUT_DIR, base_name + "_overlay.jpg"))
        # vis_id.save(os.path.join(config.ID_MASK_OUTPUT_DIR, base_name + "_id_mask.jpg"))
        # vis_bbox.save(os.path.join(config.BBOX_OUTPUT_DIR, base_name + "_bbox.jpg"))
        
        # 2. Reverse Outputs
        # vis_rev_overlay = Visualizer.create_mask_overlay(img_pil, reverse_mask, color=(255, 0, 0))
        # vis_rev_bbox = Visualizer.create_bbox_overlay(img_pil, reverse_mask, color=(0, 255, 0))
        
        # vis_rev_overlay.save(os.path.join(config.REVERSE_OVERLAY_OUTPUT_DIR, base_name + "_rev_overlay.jpg"))
        # vis_rev_bbox.save(os.path.join(config.REVERSE_BBOX_OUTPUT_DIR, base_name + "_rev_bbox.jpg"))

    @staticmethod
    def create_overlay_with_id(original_img_pil, detected_objects, dsize):
        """
        객체별 ID와 확률(Score)을 텍스트로 표시하고 색상을 입힌 오버레이 생성
        """
        img_np = np.array(original_img_pil)
        overlay_np = img_np.copy()
        
        # 1. 마스크 색칠
        for obj in detected_objects:
            resized_mask = cv2.resize(obj['mask'].astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST)
            color = config.COLORS[obj['id'] % 255]
            mask_bool = resized_mask > 0
            overlay_np[mask_bool] = (img_np[mask_bool] * 0.5 + color * 0.5).astype(np.uint8)
        
        vis_img = overlay_np.copy()
        
        # 2. ID 텍스트 표시
        for obj in detected_objects:
            resized_mask = cv2.resize(obj['mask'].astype(np.uint8), dsize, interpolation=cv2.INTER_NEAREST)
            contours, _ = cv2.findContours(resized_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                c = max(contours, key=cv2.contourArea)
                M = cv2.moments(c)
                if M["m00"] != 0:
                    cX, cY = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
                    label = f"ID:{obj['id']}"
                    cv2.putText(vis_img, label, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        return Image.fromarray(vis_img)

    @staticmethod
    def create_mask_overlay(original_img_pil, mask_np, color=(255, 0, 0), alpha=0.5):
        """
        단일 마스크에 대해 반투명 오버레이 생성
        """
        img_np = np.array(original_img_pil)
        overlay_np = img_np.copy()
        mask_bool = mask_np > 0
        rgb_color = np.array(color, dtype=np.uint8)
        overlay_np[mask_bool] = (img_np[mask_bool] * (1 - alpha) + rgb_color * alpha).astype(np.uint8)
        return Image.fromarray(overlay_np)

    @staticmethod
    def create_bbox_overlay(original_img_pil, mask_np, color=(0, 255, 0), thickness=3):
        """
        마스크 영역에 사각형(BBox) 그리기
        """
        img_np = np.array(original_img_pil)
        vis_img = img_np.copy()
        mask_u8 = (mask_np > 0).astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            if cv2.contourArea(cnt) < 10: continue
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(vis_img, (x, y), (x + w, y + h), color, thickness)
        return Image.fromarray(vis_img)