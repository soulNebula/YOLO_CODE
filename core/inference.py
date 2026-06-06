import os
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal


# ── 固定类别颜色（黄金比例色相，与标注工具一致） ──────────

def _gen_class_color(class_id):
    """为每个类别生成固定颜色"""
    import colorsys
    hue = (class_id * 0.618) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return (int(b * 255), int(g * 255), int(r * 255))  # BGR for OpenCV


_FIXED_COLORS = {}  # 缓存


def _get_color(class_id):
    if class_id not in _FIXED_COLORS:
        _FIXED_COLORS[class_id] = _gen_class_color(class_id)
    return _FIXED_COLORS[class_id]


# ── QThread 视频/摄像头推理线程 ─────────────────────────────

class VideoWorker(QThread):
    frame_ready = pyqtSignal(object, list)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, source, conf=0.25, iou=0.45, max_det=300,
                 agnostic_nms=False, half=False, vid_stride=1, class_names=None):
        super().__init__()
        self._model = model
        self._source = source
        self._conf = conf
        self._iou = iou
        self._max_det = max_det
        self._agnostic_nms = agnostic_nms
        self._half = half
        self._vid_stride = vid_stride
        self._class_names = class_names or []
        self._stop_requested = False

    def request_stop(self):
        self._stop_requested = True

    def run(self):
        try:
            cap = cv2.VideoCapture(self._source)
            if not cap.isOpened():
                self.error.emit(f"无法打开视频源: {self._source}")
                return

            frame_idx = 0
            while not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                if frame_idx % self._vid_stride != 0:
                    continue  # 跳帧

                try:
                    results = self._model(
                        frame,
                        conf=self._conf, iou=self._iou,
                        max_det=self._max_det,
                        agnostic_nms=self._agnostic_nms,
                        half=self._half,
                        verbose=False
                    )
                except Exception as e:
                    self.error.emit(f"推理出错: {e}")
                    break

                annotated, detections = self._process_results(frame, results)
                self.frame_ready.emit(annotated, detections)

            cap.release()
        except Exception as e:
            self.error.emit(f"视频处理异常: {e}")
        finally:
            self.finished.emit()

    @staticmethod
    def _process_results(img, results):
        """处理推理结果 — 用固定颜色画框，显示类别+conf"""
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int) if r.boxes.xyxy is not None else []
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
                clss = r.boxes.cls.cpu().numpy().astype(int) if r.boxes.cls is not None else []
                annotated = img.copy()
                for i in range(len(boxes)):
                    cls_id = int(clss[i])
                    conf = float(confs[i])
                    x1, y1, x2, y2 = int(boxes[i][0]), int(boxes[i][1]), int(boxes[i][2]), int(boxes[i][3])
                    cls_name = r.names.get(cls_id, f"cls_{cls_id}")
                    color = _get_color(cls_id)

                    # 画框
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    # 标签：类别 + conf
                    label = f"{cls_name} {conf:.2f}"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
                    cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

                    detections.append({
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                        'conf': conf, 'class_id': cls_id,
                        'class_name': cls_name, 'color': color
                    })
                return annotated, detections
        return img, detections


# ── InferenceManager ─────────────────────────────────────────

class InferenceManager:

    def __init__(self):
        self.model = None
        self.model_path = None
        self.conf_threshold = 0.25
        self.iou_threshold = 0.45
        self.max_det = 300
        self.agnostic_nms = False
        self.half_prec = False
        self.is_running = False
        self._worker = None
        self.result_callback = None
        self.class_names = []

    def set_result_callback(self, callback):
        self.result_callback = callback

    def load_model(self, model_path):
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.class_names = list(self.model.names.values()) if self.model.names else []
            return True, f"模型加载成功: {Path(model_path).name}"
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"

    def set_params(self, conf_threshold=0.25, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def run_image(self, image_path, imgsz=640, half=False, augment=False,
                  max_det=300, agnostic_nms=False, device=None):
        if self.model is None:
            return None, "请先加载模型"
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, f"无法读取图片: {image_path}"
            results = self.model(
                img, conf=self.conf_threshold, iou=self.iou_threshold,
                imgsz=imgsz, half=half, augment=augment, max_det=max_det,
                agnostic_nms=agnostic_nms, device=device, verbose=False
            )
            annotated, detections = VideoWorker._process_results(img, results)
            return (annotated, detections), None
        except Exception as e:
            return None, str(e)

    def run_video(self, video_path, frame_callback=None, vid_stride=1):
        if self.model is None:
            return "请先加载模型"
        self.is_running = True
        self._worker = VideoWorker(
            self.model, video_path,
            self.conf_threshold, self.iou_threshold,
            self.max_det, self.agnostic_nms,
            self.half_prec, vid_stride,
            self.class_names
        )
        if frame_callback:
            self._worker.frame_ready.connect(frame_callback)
        self._worker.finished.connect(self._on_video_finished)
        self._worker.error.connect(self._on_video_error)
        self._worker.start()

    def run_camera(self, camera_id=0, frame_callback=None):
        self.run_video(camera_id, frame_callback, vid_stride=1)

    def stop(self):
        self.is_running = False
        if self._worker and self._worker.isRunning():
            self._worker.request_stop()
            self._worker.wait(3000)

    def _on_video_finished(self):
        self.is_running = False

    def _on_video_error(self, msg):
        if self.result_callback:
            self.result_callback(None, [{'error': msg}])
        self.is_running = False
