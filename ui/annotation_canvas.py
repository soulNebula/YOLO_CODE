"""标注画布 —— 纯渲染层，分层渲染优化性能"""
import cv2
import gc
import numpy as np
from PyQt5.QtWidgets import QLabel
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QColor


class ImageCanvas(QLabel):
    """图片显示和标注画布 — 分层渲染架构"""

    bboxDrawn = pyqtSignal(int, int, int, int)
    bboxDeleteRequested = pyqtSignal(int)
    bboxMoveFinished = pyqtSignal(int, int, int, int, int)
    bboxSelected = pyqtSignal(int)
    contextMenuRequested = pyqtSignal(int, object)
    copyRequested = pyqtSignal()
    pasteRequested = pyqtSignal()
    duplicateRequested = pyqtSignal()
    microMoveRequested = pyqtSignal(int, int)
    zoomChanged = pyqtSignal(float)
    mouseMoved = pyqtSignal(int, int)
    panModeChanged = pyqtSignal(bool)

    CORNER_RADIUS = 10
    CORNER_ANCHORS = {
        0: ('x1', 'y1', 'x2', 'y2'),
        1: ('x2', 'y1', 'x1', 'y2'),
        2: ('x1', 'y2', 'x2', 'y1'),
        3: ('x2', 'y2', 'x1', 'y1'),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(
            "background-color: #fafafa; border: 2px solid #e0e0e0; border-radius: 4px;"
        )
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        self.original_image = None
        self.display_image = None
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0

        # 标注视图缓存
        self.annotation_view = []
        self.class_names = []
        self.colors = {}

        # 分层渲染缓存
        self._bg_pixmap = None       # 背景层（图片+缩放），仅缩放/切图时重建
        self._bg_dirty = True        # 背景是否需要重建
        self._need_render = False    # 延迟渲染标志
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(16)  # ~60fps 上限
        self._render_timer.timeout.connect(self._do_render)
        self._render_buffer = None

        # 绘制状态
        self.drawing = False
        self.draw_start = None
        self.draw_end = None
        self.current_class_id = 0

        # 选中状态
        self.selected_bbox_idx = -1
        self.dragging = False
        self.drag_start_pos = None
        self._drag_original_bbox = None

        # 角缩放
        self.corner_dragging = False
        self.drag_corner = -1

        # 拖拽模式
        self._pan_mode = False
        self._panning = False
        self._pan_start = None

        self.setCursor(Qt.ArrowCursor)

    # ═══════════════════════════════════════════════════════════
    # 公共接口
    # ═══════════════════════════════════════════════════════════

    def set_class_names(self, names):
        self.class_names = names

    def set_colors(self, colors):
        self.colors = colors

    def set_current_class(self, class_id):
        self.current_class_id = class_id

    def set_annotations(self, annos):
        """接收标注数据 — 不做深拷贝，直接引用（调用者已拷贝）"""
        self.annotation_view = annos
        self._bg_dirty = True

    def set_pan_mode(self, enabled):
        self._pan_mode = enabled
        self._panning = False
        self._pan_start = None
        self.setCursor(Qt.OpenHandCursor if enabled else Qt.ArrowCursor)

    def is_pan_mode(self):
        return self._pan_mode

    def load_image(self, img_array, annotations=None, zoom_fit=True):
        self.original_image = img_array.copy() if img_array is not None else None
        self.annotation_view = annotations if annotations is not None else []

        if zoom_fit and self.original_image is not None:
            h, w = self.original_image.shape[:2]
            cw = self.width()
            ch = self.height()
            if cw > 0 and ch > 0:
                self.zoom = min(cw / w, ch / h) * 0.9

        self.offset_x = 0
        self.offset_y = 0
        self.selected_bbox_idx = -1
        self.corner_dragging = False
        self.drag_corner = -1
        self._bg_dirty = True
        self._invalidate()

    def zoom_to_fit(self):
        if self.original_image is None:
            return
        h, w = self.original_image.shape[:2]
        cw = self.width()
        ch = self.height()
        if cw > 0 and ch > 0:
            self.zoom = min(cw / w, ch / h) * 0.9
        self.offset_x = 0
        self.offset_y = 0
        self._bg_dirty = True
        self.zoomChanged.emit(self.zoom)
        self._invalidate()

    def zoom_to_width(self):
        if self.original_image is None:
            return
        h, w = self.original_image.shape[:2]
        cw = self.width()
        if cw > 0:
            self.zoom = cw / w * 0.95
        self.offset_x = 0
        self.offset_y = 0
        self._bg_dirty = True
        self.zoomChanged.emit(self.zoom)
        self._invalidate()

    def zoom_reset(self):
        self.zoom = 1.0
        self.offset_x = 0
        self.offset_y = 0
        self._bg_dirty = True
        self.zoomChanged.emit(self.zoom)
        self._invalidate()

    def zoom_percent(self):
        return int(self.zoom * 100)

    def cleanup(self):
        """释放缓存资源"""
        self._bg_pixmap = None
        self._render_buffer = None
        self.display_image = None
        gc.collect()

    # ═══════════════════════════════════════════════════════════
    # 分层渲染
    # ═══════════════════════════════════════════════════════════

    def _invalidate(self):
        """标记需要渲染（合并高频调用）"""
        if not self._need_render:
            self._need_render = True
            self._render_timer.start()

    def _do_render(self):
        """实际执行渲染"""
        self._need_render = False
        self._render()

    def _render(self):
        """分层渲染：背景层 + 标注覆盖层"""
        if self.original_image is None:
            self.display_image = None
            self.setText("请加载图片")
            return

        h, w = self.original_image.shape[:2]
        new_w = int(w * self.zoom)
        new_h = int(h * self.zoom)
        if new_w <= 0 or new_h <= 0:
            return

        # ── 背景层重建（仅缩放/切图/Pan时） ──
        if self._bg_dirty or self._bg_pixmap is None:
            try:
                img = self.original_image.copy()
            except Exception:
                return

            # 使用 INTER_LINEAR 加速（比 INTER_AREA 快很多）
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            img_rgb = np.ascontiguousarray(img)
            self._render_buffer = img_rgb
            h2, w2 = img_rgb.shape[:2]
            q_img = QImage(img_rgb.data, w2, h2, w2 * 3, QImage.Format_RGB888)

            bg_color = QColor("#fafafa")
            self._bg_pixmap = QPixmap(self.width(), self.height())
            self._bg_pixmap.fill(bg_color)
            p = QPainter(self._bg_pixmap)
            draw_x = (self.width() - new_w) // 2 + self.offset_x
            draw_y = (self.height() - new_h) // 2 + self.offset_y
            p.drawImage(draw_x, draw_y, q_img)
            p.end()
            self._bg_dirty = False

        # ── 覆盖层：在背景上绘制标注框 ──
        canvas = QPixmap(self._bg_pixmap)
        painter = QPainter(canvas)

        for idx, ann in enumerate(self.annotation_view):
            cls_id = ann.get('class_id', 0)
            color = self.colors.get(cls_id, (0, 255, 0))

            x1, y1, x2, y2 = self._get_pixel_coords(ann, w, h)
            # 坐标转换到缩放后的画布
            sx1 = int(x1 * self.zoom) + (self.width() - new_w) // 2 + self.offset_x
            sy1 = int(y1 * self.zoom) + (self.height() - new_h) // 2 + self.offset_y
            sx2 = int(x2 * self.zoom) + (self.width() - new_w) // 2 + self.offset_x
            sy2 = int(y2 * self.zoom) + (self.height() - new_h) // 2 + self.offset_y

            qcolor = QColor(color[0], color[1], color[2], 200)
            pen = painter.pen()
            pen.setColor(qcolor)
            pen_width = 3 if idx == self.selected_bbox_idx else 2
            pen.setWidth(pen_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(sx1, sy1, sx2 - sx1, sy2 - sy1)

            # 标签
            cls_name = (
                self.class_names[cls_id]
                if cls_id < len(self.class_names) else f"cls_{cls_id}"
            )
            label = f" {cls_name} "
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label)
            th = fm.height()
            lx = sx1
            ly = sy1 - th - 4
            if ly < 0:
                ly = sy2 + 4

            painter.setBrush(qcolor)
            painter.setPen(Qt.NoPen)
            painter.drawRect(lx, ly, tw, th)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(lx, ly + fm.ascent(), label)

            # 选中框高亮 + 四角手柄
            if idx == self.selected_bbox_idx:
                painter.setBrush(Qt.NoBrush)
                pen.setColor(QColor(255, 255, 0))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.drawRect(sx1 - 2, sy1 - 2, sx2 - sx1 + 4, sy2 - sy1 + 4)

                hs = 6
                painter.setBrush(QColor(255, 255, 0))
                for (cx, cy) in [(sx1, sy1), (sx2, sy1), (sx1, sy2), (sx2, sy2)]:
                    painter.drawRect(cx - hs, cy - hs, hs * 2, hs * 2)

        # 正在绘制的框
        if self.drawing and self.draw_start and self.draw_end:
            color = self.colors.get(self.current_class_id, (0, 107, 255))
            dx1 = int(min(self.draw_start[0], self.draw_end[0]) * self.zoom) + (self.width() - new_w) // 2 + self.offset_x
            dy1 = int(min(self.draw_start[1], self.draw_end[1]) * self.zoom) + (self.height() - new_h) // 2 + self.offset_y
            dx2 = int(max(self.draw_start[0], self.draw_end[0]) * self.zoom) + (self.width() - new_w) // 2 + self.offset_x
            dy2 = int(max(self.draw_start[1], self.draw_end[1]) * self.zoom) + (self.height() - new_h) // 2 + self.offset_y

            pen = painter.pen()
            pen.setColor(QColor(color[0], color[1], color[2]))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(dx1, dy1, dx2 - dx1, dy2 - dy1)

        painter.end()
        self.display_image = None  # 不再使用 OpenCV 图片显示
        self.setPixmap(canvas)

        # 定期清理
        if not self.drawing and not self.corner_dragging and not self.dragging and not self._panning:
            self._cleanup_old_pixmaps()

    def _cleanup_old_pixmaps(self):
        """释放旧的 QPixmap 引用"""
        self._render_buffer = None

    @staticmethod
    def _get_pixel_coords(ann, img_w, img_h):
        if 'x1' in ann and 'y1' in ann:
            return int(ann['x1']), int(ann['y1']), int(ann['x2']), int(ann['y2'])
        x1 = int((ann['cx'] - ann['w'] / 2) * img_w)
        y1 = int((ann['cy'] - ann['h'] / 2) * img_h)
        x2 = int((ann['cx'] + ann['w'] / 2) * img_w)
        y2 = int((ann['cy'] + ann['h'] / 2) * img_h)
        return x1, y1, x2, y2

    # ═══════════════════════════════════════════════════════════
    # 坐标转换
    # ═══════════════════════════════════════════════════════════

    def _to_image_coords(self, pos):
        try:
            if self.original_image is None:
                return None, None
            oh, ow = self.original_image.shape[:2]
            new_w = int(ow * self.zoom)
            new_h = int(oh * self.zoom)
            if new_w <= 0 or new_h <= 0:
                return None, None

            center_x = (self.width() - new_w) // 2 + self.offset_x
            center_y = (self.height() - new_h) // 2 + self.offset_y
            x = int((pos.x() - center_x) / self.zoom)
            y = int((pos.y() - center_y) / self.zoom)
            x = max(0, min(ow - 1, x))
            y = max(0, min(oh - 1, y))
            return x, y
        except Exception:
            return None, None

    def _find_bbox_at(self, img_x, img_y):
        if self.original_image is None:
            return -1
        h, w = self.original_image.shape[:2]
        for idx, ann in reversed(list(enumerate(self.annotation_view))):
            x1, y1, x2, y2 = self._get_pixel_coords(ann, w, h)
            if x1 <= img_x <= x2 and y1 <= img_y <= y2:
                return idx
        return -1

    def _find_corner_at(self, img_x, img_y, bbox_idx):
        if bbox_idx < 0 or self.original_image is None:
            return -1
        h, w = self.original_image.shape[:2]
        ann = self.annotation_view[bbox_idx]
        x1, y1, x2, y2 = self._get_pixel_coords(ann, w, h)
        corners = [(x1, y1), (x2, y1), (x1, y2), (x2, y2)]
        for i, (cx, cy) in enumerate(corners):
            if abs(img_x - cx) <= self.CORNER_RADIUS and abs(img_y - cy) <= self.CORNER_RADIUS:
                return i
        return -1

    # ═══════════════════════════════════════════════════════════
    # 鼠标事件 — 使用 _invalidate 去抖
    # ═══════════════════════════════════════════════════════════

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self.original_image is None:
            return

        if self._pan_mode:
            self._panning = True
            self._pan_start = (event.pos().x(), event.pos().y())
            self.setCursor(Qt.ClosedHandCursor)
            return

        try:
            img_x, img_y = self._to_image_coords(event.pos())
            if img_x is None:
                return

            corner = -1
            if self.selected_bbox_idx >= 0:
                corner = self._find_corner_at(img_x, img_y, self.selected_bbox_idx)
            if corner >= 0:
                self.corner_dragging = True
                self.drag_corner = corner
                self.drag_start_pos = (img_x, img_y)
                self.setCursor(self._corner_cursor(corner))
                return

            hit_idx = self._find_bbox_at(img_x, img_y)
            if hit_idx >= 0:
                self.selected_bbox_idx = hit_idx
                self.drag_start_pos = (img_x, img_y)
                h, w = self.original_image.shape[:2]
                ann = self.annotation_view[hit_idx]
                x1, y1, x2, y2 = self._get_pixel_coords(ann, w, h)
                self._drag_original_bbox = (hit_idx, x1, y1, x2, y2)
                self.setCursor(Qt.ClosedHandCursor)
                self.bboxSelected.emit(hit_idx)
            else:
                self.selected_bbox_idx = -1
                self.bboxSelected.emit(-1)
                self.drawing = True
                self.draw_start = (img_x, img_y)
                self.draw_end = (img_x, img_y)
                self.setCursor(Qt.CrossCursor)

            self._invalidate()
        except Exception:
            pass

    def mouseMoveEvent(self, event):
        if self.original_image is None:
            return

        if self._pan_mode and self._panning and self._pan_start:
            dx = event.pos().x() - self._pan_start[0]
            dy = event.pos().y() - self._pan_start[1]
            self.offset_x += dx
            self.offset_y += dy
            self._pan_start = (event.pos().x(), event.pos().y())
            self._bg_dirty = True
            self._invalidate()
            return

        try:
            img_x, img_y = self._to_image_coords(event.pos())
            if img_x is not None:
                self.mouseMoved.emit(img_x, img_y)

            if self.corner_dragging:
                self._update_corner(img_x, img_y)
                self._invalidate()
            elif self.drawing:
                self.draw_end = (img_x, img_y)
                self._invalidate()
            elif self.selected_bbox_idx >= 0 and self.drag_start_pos:
                dx = img_x - self.drag_start_pos[0]
                dy = img_y - self.drag_start_pos[1]
                ann = self.annotation_view[self.selected_bbox_idx]
                if 'x1' in ann:
                    ann['x1'] += dx
                    ann['y1'] += dy
                    ann['x2'] += dx
                    ann['y2'] += dy
                self.drag_start_pos = (img_x, img_y)
                self._invalidate()
            else:
                hovering = self._find_bbox_at(img_x, img_y) >= 0
                self.setCursor(
                    Qt.PointingHandCursor if hovering else Qt.ArrowCursor
                )
        except Exception:
            pass

    def mouseReleaseEvent(self, event):
        if self._pan_mode and self._panning:
            self._panning = False
            self._pan_start = None
            self.setCursor(Qt.OpenHandCursor)
            return

        try:
            if self.corner_dragging:
                self.corner_dragging = False
                self.drag_corner = -1
                if self.selected_bbox_idx >= 0:
                    h, w = self.original_image.shape[:2]
                    ann = self.annotation_view[self.selected_bbox_idx]
                    x1, y1, x2, y2 = self._get_pixel_coords(ann, w, h)
                    self.bboxMoveFinished.emit(
                        self.selected_bbox_idx, x1, y1, x2, y2
                    )
                self.setCursor(
                    Qt.PointingHandCursor
                    if self.selected_bbox_idx >= 0 else Qt.ArrowCursor
                )
                self.drag_start_pos = None
                gc.collect()
                return

            if self.drawing:
                self.drawing = False
                if self.draw_start and self.draw_end:
                    x1, y1 = self.draw_start
                    x2, y2 = self.draw_end
                    if abs(x2 - x1) > 5 and abs(y2 - y1) > 5:
                        self.bboxDrawn.emit(
                            min(x1, x2), min(y1, y2),
                            max(x1, x2), max(y1, y2)
                        )
                self.draw_start = None
                self.draw_end = None
                self.setCursor(Qt.ArrowCursor)
                self._invalidate()
            elif self.selected_bbox_idx >= 0 and self._drag_original_bbox:
                orig_idx, ox1, oy1, ox2, oy2 = self._drag_original_bbox
                h, w = self.original_image.shape[:2]
                ann = self.annotation_view[self.selected_bbox_idx]
                nx1, ny1, nx2, ny2 = self._get_pixel_coords(ann, w, h)
                if (nx1, ny1, nx2, ny2) != (ox1, oy1, ox2, oy2):
                    self.bboxMoveFinished.emit(
                        self.selected_bbox_idx, nx1, ny1, nx2, ny2
                    )
                self._drag_original_bbox = None
                self.setCursor(
                    Qt.PointingHandCursor
                    if self.selected_bbox_idx >= 0 else Qt.ArrowCursor
                )

            self.drag_start_pos = None
            gc.collect()
        except Exception:
            pass

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.LeftButton or self.original_image is None or self._pan_mode:
            return
        try:
            img_x, img_y = self._to_image_coords(event.pos())
            if img_x is None:
                return
            hit_idx = self._find_bbox_at(img_x, img_y)
            if hit_idx >= 0:
                self.selected_bbox_idx = hit_idx
                self.bboxSelected.emit(hit_idx)
                self._invalidate()
                self.contextMenuRequested.emit(hit_idx, self.mapToGlobal(event.pos()))
        except Exception:
            pass

    def contextMenuEvent(self, event):
        if self.original_image is None or self._pan_mode:
            return
        try:
            img_x, img_y = self._to_image_coords(event.pos())
            if img_x is None:
                return
            hit_idx = self._find_bbox_at(img_x, img_y)
            if hit_idx >= 0:
                self.selected_bbox_idx = hit_idx
                self.bboxSelected.emit(hit_idx)
                self._invalidate()
                self.contextMenuRequested.emit(hit_idx, event.globalPos())
        except Exception:
            pass

    def wheelEvent(self, event):
        delta = event.angleDelta().y()
        if delta > 0:
            self.zoom *= 1.1
        else:
            self.zoom /= 1.1
        self.zoom = max(0.1, min(10.0, self.zoom))
        self._bg_dirty = True
        self.zoomChanged.emit(self.zoom)
        self._invalidate()

    # ═══════════════════════════════════════════════════════════
    # 角缩放
    # ═══════════════════════════════════════════════════════════

    @staticmethod
    def _corner_cursor(corner):
        cursors = {
            0: Qt.SizeFDiagCursor,
            1: Qt.SizeBDiagCursor,
            2: Qt.SizeBDiagCursor,
            3: Qt.SizeFDiagCursor,
        }
        return cursors.get(corner, Qt.SizeAllCursor)

    def _update_corner(self, img_x, img_y):
        idx = self.selected_bbox_idx
        if idx < 0 or idx >= len(self.annotation_view):
            return
        ann = self.annotation_view[idx]
        if 'x1' not in ann or 'x2' not in ann:
            return
        h, w = self.original_image.shape[:2]
        img_x = max(0, min(w - 1, img_x))
        img_y = max(0, min(h - 1, img_y))
        _, _, xa_attr, ya_attr = self.CORNER_ANCHORS[self.drag_corner]

        if self.drag_corner in (0, 2):
            ann['x1'] = min(img_x, ann[xa_attr] - 5)
        else:
            ann['x2'] = max(img_x, ann[xa_attr] + 5)
        if self.drag_corner in (0, 1):
            ann['y1'] = min(img_y, ann[ya_attr] - 5)
        else:
            ann['y2'] = max(img_y, ann[ya_attr] + 5)

    # ═══════════════════════════════════════════════════════════
    # 键盘事件
    # ═══════════════════════════════════════════════════════════

    def keyPressEvent(self, event):
        key = event.key()
        mods = event.modifiers()

        if self._pan_mode:
            super().keyPressEvent(event)
            return

        if key == Qt.Key_Delete and self.selected_bbox_idx >= 0:
            self.bboxDeleteRequested.emit(self.selected_bbox_idx)
            if 0 <= self.selected_bbox_idx < len(self.annotation_view):
                self.annotation_view.pop(self.selected_bbox_idx)
                self.selected_bbox_idx = -1
                self._bg_dirty = True
                self._invalidate()

        elif key == Qt.Key_Escape:
            self.drawing = False
            self.draw_start = None
            self.draw_end = None
            self.selected_bbox_idx = -1
            self.corner_dragging = False
            self.bboxSelected.emit(-1)
            self._invalidate()

        elif key == Qt.Key_C and mods == Qt.ControlModifier:
            self.copyRequested.emit()
        elif key == Qt.Key_V and mods == Qt.ControlModifier:
            self.pasteRequested.emit()
        elif key == Qt.Key_D and mods == Qt.ControlModifier:
            self.duplicateRequested.emit()

        elif key == Qt.Key_F and mods == Qt.ControlModifier:
            if mods & Qt.ShiftModifier:
                self.zoom_to_width()
            else:
                self.zoom_to_fit()
        elif key == Qt.Key_0 and mods == Qt.ControlModifier:
            self.zoom_reset()

        elif key == Qt.Key_H and mods == Qt.ControlModifier:
            if not hasattr(self, '_hidden_annotations'):
                self._hidden_annotations = None
            if self._hidden_annotations is None and self.annotation_view:
                self._hidden_annotations = list(self.annotation_view)
                self.annotation_view = []
            elif self._hidden_annotations is not None:
                self.annotation_view = self._hidden_annotations
                self._hidden_annotations = None
            self._bg_dirty = True
            self._invalidate()

        elif key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
            if self.selected_bbox_idx >= 0:
                dx, dy = 0, 0
                if key == Qt.Key_Left:
                    dx = -1
                elif key == Qt.Key_Right:
                    dx = 1
                elif key == Qt.Key_Up:
                    dy = -1
                elif key == Qt.Key_Down:
                    dy = 1
                if mods & Qt.ShiftModifier:
                    dx *= 10
                    dy *= 10
                self.microMoveRequested.emit(dx, dy)
                ann = self.annotation_view[self.selected_bbox_idx]
                if 'x1' in ann:
                    ann['x1'] += dx
                    ann['y1'] += dy
                    ann['x2'] += dx
                    ann['y2'] += dy
                self._invalidate()
        else:
            super().keyPressEvent(event)
