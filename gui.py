# gui.py
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
import matplotlib.patches as patches
import matplotlib.gridspec as gridspec
import numpy as np
import cv2
from config import MPL_COLORS

class SmartSelector:
    def __init__(self, img_arr, title, global_thumbnails, global_counts):
        self.img = img_arr
        self.h, self.w = img_arr.shape[:2]
        self.title = title
        self.global_thumbnails = global_thumbnails
        self.global_counts = global_counts
        
        self.annotations = {} 
        
        # Brush State
        self.brush_active = False
        self.brush_radius = 15
        self.temp_mask = np.zeros((self.h, self.w), dtype=np.uint8) # 현재 그리는 마스크
        
        self.fig = plt.figure(figsize=(16, 9))
        gs = gridspec.GridSpec(4, 5, width_ratios=[6, 1, 1, 1, 1], figure=self.fig)
        
        self.ax_main = self.fig.add_subplot(gs[:, 0])
        self.ax_main.imshow(self.img)
        
        # 투명 오버레이 초기화 (밝기 문제 해결됨)
        empty_overlay = np.zeros((self.h, self.w, 4), dtype=np.float32)
        self.brush_layer = self.ax_main.imshow(empty_overlay)
        
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
        self.is_drawing = False
        
        self.update_title()

    def update_title(self):
        mode_str = "[BRUSH MODE ON]" if self.brush_active else "[BOX MODE]"
        brush_info = f" | Brush Size: {self.brush_radius} (Wheel or [ , ])" if self.brush_active else ""
        
        self.ax_main.set_title(
            f"{self.title}\n"
            f"Current ID: {self.current_obj_id} {mode_str} {brush_info}\n"
            f"Keys: B(Brush Toggle), 1-9(ID), N(Next), D(Delete), Q(Finish)\n"
            f"[Left Drag]: Box/Paint | [Left Click]: Point(+) | [Right Click]: Point(-)"
        , fontsize=10)
        self.fig.canvas.draw_idle()

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

    def _ensure_id_structure(self):
        if self.current_obj_id not in self.annotations:
            # mask 키 추가 (브러쉬용)
            self.annotations[self.current_obj_id] = {'box': None, 'points': [], 'labels': [], 'mask': None, 'artists': []}
            self.global_counts[self.current_obj_id] = self.global_counts.get(self.current_obj_id, 0) + 1
            self.refresh_sidebar_slot(self.current_obj_id)

    # --- Box Logic ---
    def add_box(self, x1, y1, x2, y2):
        if self.brush_active: return # 브러쉬 모드일 땐 박스 생성 안함

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

    # --- Point Logic ---
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

    # --- Brush Logic ---
    def apply_brush(self, x, y):
        if not self.brush_active: return
        self._ensure_id_structure()

        # 현재 ID의 마스크 가져오기 또는 생성
        if self.annotations[self.current_obj_id]['mask'] is None:
            self.annotations[self.current_obj_id]['mask'] = np.zeros((self.h, self.w), dtype=np.uint8)
        
        mask = self.annotations[self.current_obj_id]['mask']
        cv2.circle(mask, (int(x), int(y)), self.brush_radius, 1, -1)
        
        # 화면 업데이트 (Overlay)
        self.update_brush_overlay()

    def update_brush_overlay(self):
        # 전체 마스크 시각화 통합
        combined_vis = np.zeros((self.h, self.w, 4), dtype=np.float32) # RGBA
        
        for oid, data in self.annotations.items():
            if data.get('mask') is not None:
                m = data['mask'] > 0
                color = self.get_color(oid)
                # RGBA 채우기
                combined_vis[m, 0] = color[0] 
                combined_vis[m, 1] = color[1]
                combined_vis[m, 2] = color[2]
                combined_vis[m, 3] = 0.6 # Alpha
        
        self.brush_layer.set_data(combined_vis)
        self.fig.canvas.draw_idle()

    # --- Helpers ---
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

    def delete_nearest(self, x, y):
        if self.current_obj_id not in self.annotations: return
        data = self.annotations[self.current_obj_id]
        
        # 브러쉬 마스크 삭제 기능 추가 (전체 삭제)
        if data.get('mask') is not None:
            if data['mask'].sum() > 0:
                print(f"  [ID: {self.current_obj_id}] Brush Mask Cleared.")
                data['mask'] = None
                self.update_brush_overlay()

        # 기존 로직 (Box/Point 삭제)
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
        
        if min_dist > 100 and data.get('mask') is None: return 
        
        if target_type == 'box':
            self._remove_box_artist()
            print(f"  [ID: {self.current_obj_id}] Box Deleted.")
        elif target_type == 'point':
            data['points'].pop(target_idx)
            data['labels'].pop(target_idx)
            # Re-draw points
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

    def process(self):
        def on_select(eclick, erelease):
            if self.brush_active: return # 브러쉬 모드면 박스 무시
            self.add_box(int(eclick.xdata), int(eclick.ydata), int(erelease.xdata), int(erelease.ydata))

        def on_mouse_press(event):
            if event.inaxes != self.ax_main: return
            self.press_event = event
            self.is_drawing = True
            
            # 브러쉬 시작 (클릭만 해도 점 찍힘)
            if self.brush_active and event.button == 1:
                self.apply_brush(event.xdata, event.ydata)

        def on_mouse_move(event):
            if not self.is_drawing or event.inaxes != self.ax_main: return
            
            # 브러쉬 드래그
            if self.brush_active and self.press_event.button == 1:
                self.apply_brush(event.xdata, event.ydata)

        def on_mouse_release(event):
            self.is_drawing = False
            if self.press_event is None or event.inaxes != self.ax_main: return
            
            # 브러쉬 모드가 아닐 때의 클릭 동작 (Point)
            if not self.brush_active:
                dx = abs(event.xdata - self.press_event.xdata)
                dy = abs(event.ydata - self.press_event.ydata)
                dist = (dx**2 + dy**2)**0.5
                
                if dist < 5: 
                    if event.button == 1:   # 좌클릭 -> Positive
                        self.add_point(event.xdata, event.ydata, 1)
                    elif event.button == 3: # 우클릭 -> Negative
                        self.add_point(event.xdata, event.ydata, 0)
                    elif event.button == 2: # 휠클릭 -> 삭제
                        self.delete_nearest(event.xdata, event.ydata)
            
            self.press_event = None

        # [추가] 마우스 휠 이벤트 핸들러
        def on_scroll(event):
            if not self.brush_active: return
            
            if event.button == 'up':
                self.brush_radius = min(100, self.brush_radius + 2)
            elif event.button == 'down':
                self.brush_radius = max(1, self.brush_radius - 2)
            
            self.update_title()

        def on_key(event):
            if event.key == 'd' and event.inaxes == self.ax_main:
                self.delete_nearest(event.xdata, event.ydata)

            if event.key in ['q', 'Q', 'enter', 'escape'] and self.current_obj_id <= 10: 
                if event.key.lower() == 'q' and event.key != 'Q': pass 
                else: plt.close(self.fig)
            
            # ID 변경
            if event.key in ['n', 'N']:
                self.current_obj_id = (self.current_obj_id % 16) + 1
                self.update_title()
            elif event.key in [str(i) for i in range(1, 10)]:
                self.current_obj_id = int(event.key)
                self.update_title()
            
            # === 브러쉬 관련 키 ===
            if event.key.lower() == 'b':
                self.brush_active = not self.brush_active
                # 브러쉬 모드일 때 RectangleSelector 비활성화 (충돌 방지)
                self.rs.set_active(not self.brush_active)
                self.update_title()
            
            if event.key == '[':
                self.brush_radius = max(1, self.brush_radius - 2)
                self.update_title()
            if event.key == ']':
                self.brush_radius = min(100, self.brush_radius + 2)
                self.update_title()

        self.fig.canvas.mpl_connect('key_press_event', on_key)
        self.fig.canvas.mpl_connect('button_press_event', on_mouse_press)
        self.fig.canvas.mpl_connect('button_release_event', on_mouse_release)
        self.fig.canvas.mpl_connect('motion_notify_event', on_mouse_move)
        self.fig.canvas.mpl_connect('scroll_event', on_scroll) # [추가] 휠 이벤트 연결
        
        self.rs = RectangleSelector(self.ax_main, on_select, useblit=False, button=[1], 
                                    minspanx=5, minspany=5, spancoords='pixels', interactive=True)
        
        plt.tight_layout()
        plt.show(block=True)
        return self.annotations