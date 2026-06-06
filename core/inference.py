import os
import cv2
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal


# ── QThread 视频/摄像头推理线程 ─────────────────────────────

class VideoWorker(QThread):
    """QThread 视频推理线程"""
    frame_ready = pyqtSignal(object, list)   # frame (numpy), detections (list)
    finished = pyqtSignal()
    error = pyqtSignal(str)

    def __init__(self, model, source, conf=0.5, iou=0.45, class_names=None):
        super().__init__()
        self._model = model
        self._source = source
        self._conf = conf
        self._iou = iou
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

            while not self._stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                try:
                    results = self._model(
                        frame, conf=self._conf, iou=self._iou, verbose=False
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
        detections = []
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy() if r.boxes.xyxy is not None else []
                confs = r.boxes.conf.cpu().numpy() if r.boxes.conf is not None else []
                clss = r.boxes.cls.cpu().numpy() if r.boxes.cls is not None else []
                annotated = r.plot()
                for i in range(len(boxes)):
                    detections.append({
                        'x1': int(boxes[i][0]), 'y1': int(boxes[i][1]),
                        'x2': int(boxes[i][2]), 'y2': int(boxes[i][3]),
                        'conf': float(confs[i]),
                        'class_id': int(clss[i]),
                        'class_name': r.names.get(int(clss[i]), f"cls_{int(clss[i])}")
                    })
                return annotated, detections
        return img, detections


class InferenceManager:
    """模型推理管理器"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.is_running = False
        self._worker = None
        self.result_callback = None
        self.class_names = []

    def set_result_callback(self, callback):
        self.result_callback = callback

    def load_model(self, model_path):
        """加载模型"""
        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            self.model_path = model_path
            self.class_names = list(self.model.names.values()) if self.model.names else []
            return True, f"模型加载成功: {Path(model_path).name}"
        except Exception as e:
            return False, f"模型加载失败: {str(e)}"

    def set_params(self, conf_threshold=0.5, iou_threshold=0.45):
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold

    def run_image(self, image_path, imgsz=640, half=False, augment=False, max_det=300, device=None):
        """对单张图片进行推理（同步，图片通常很快）"""
        if self.model is None:
            return None, "请先加载模型"

        try:
            img = cv2.imread(image_path)
            if img is None:
                return None, f"无法读取图片: {image_path}"

            results = self.model(
                img, conf=self.conf_threshold, iou=self.iou_threshold,
                imgsz=imgsz, half=half, augment=augment, max_det=max_det,
                device=device, verbose=False
            )
            return self._process_results(img, results), None
        except Exception as e:
            return None, str(e)

    def run_video(self, video_path, frame_callback=None):
        """对视频进行推理（QThread）"""
        if self.model is None:
            return "请先加载模型"

        self.is_running = True
        self._worker = VideoWorker(
            self.model, video_path,
            self.conf_threshold, self.iou_threshold,
            self.class_names
        )
        if frame_callback:
            self._worker.frame_ready.connect(frame_callback)
        self._worker.finished.connect(self._on_video_finished)
        self._worker.error.connect(self._on_video_error)
        self._worker.start()

    def run_camera(self, camera_id=0, frame_callback=None):
        """对摄像头进行实时推理（QThread）"""
        self.run_video(camera_id, frame_callback)

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

    def _process_results(self, img, results):
        """处理推理结果"""
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if result.boxes is not None:
                boxes = result.boxes.xyxy.cpu().numpy() if result.boxes.xyxy is not None else []
                confidences = result.boxes.conf.cpu().numpy() if result.boxes.conf is not None else []
                class_ids = result.boxes.cls.cpu().numpy().astype(int) if result.boxes.cls is not None else []

                for i in range(len(boxes)):
                    detections.append({
                        'bbox': boxes[i].tolist(),
                        'confidence': float(confidences[i]),
                        'class_id': int(class_ids[i]),
                        'class_name': self.class_names[class_ids[i]] if class_ids[i] < len(self.class_names) else f"class_{class_ids[i]}"
                    })

            # 绘制标注结果
            annotated = result.plot()
            return annotated, detections

        return img, []
