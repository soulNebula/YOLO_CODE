import os
import cv2
import numpy as np
from pathlib import Path
import threading

class InferenceManager:
    """模型推理管理器"""

    def __init__(self):
        self.model = None
        self.model_path = None
        self.conf_threshold = 0.5
        self.iou_threshold = 0.45
        self.input_source = None
        self.is_running = False
        self.stop_requested = False
        self.inference_thread = None
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
        """对单张图片进行推理"""
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
        """对视频进行推理"""
        if self.model is None:
            return "请先加载模型"

        self.is_running = True
        self.stop_requested = False
        self.inference_thread = threading.Thread(
            target=self._process_video, args=(video_path, frame_callback), daemon=True
        )
        self.inference_thread.start()

    def run_camera(self, camera_id=0, frame_callback=None):
        """对摄像头进行实时推理"""
        self.run_video(camera_id, frame_callback)  # OpenCV可以将摄像头ID作为视频源

    def stop(self):
        self.stop_requested = True
        self.is_running = False

    def _process_video(self, source, frame_callback=None):
        try:
            cap = cv2.VideoCapture(source)
            while self.is_running and not self.stop_requested:
                ret, frame = cap.read()
                if not ret:
                    break

                results = self.model(frame, conf=self.conf_threshold, iou=self.iou_threshold, verbose=False)
                annotated_frame, detections = self._process_results(frame, results)

                if frame_callback:
                    frame_callback(annotated_frame, detections)

            cap.release()
        except Exception as e:
            pass

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
