import os
import threading
from pathlib import Path
import numpy as np

class EvaluationManager:
    """模型评估管理器"""

    def __init__(self):
        self.is_running = False
        self.results = None
        self.log_callback = None
        self.progress_callback = None

    def set_callbacks(self, log_callback=None, progress_callback=None):
        self.log_callback = log_callback
        self.progress_callback = progress_callback

    def evaluate(self, model_path, data_yaml, conf=0.5, iou=0.45):
        """评估模型"""
        if self.is_running:
            return False

        self.is_running = True
        thread = threading.Thread(
            target=self._run_evaluation,
            args=(model_path, data_yaml, conf, iou),
            daemon=True
        )
        thread.start()
        return True

    def _run_evaluation(self, model_path, data_yaml, conf, iou):
        try:
            from ultralytics import YOLO
        except ImportError:
            self._emit_log("错误: 未安装ultralytics库")
            self.is_running = False
            return

        try:
            self._emit_log(f"加载模型: {model_path}")
            self._emit_progress(10)

            model = YOLO(model_path)
            self._emit_progress(30)
            self._emit_log("开始评估...")

            metrics = model.val(
                data=data_yaml,
                conf=conf,
                iou=iou,
                verbose=False
            )

            self._emit_progress(90)

            # 收集评估指标
            self.results = {
                'mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
                'mAP50-95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
                'precision': float(metrics.box.p[0]) if len(metrics.box.p) > 0 else 0,
                'recall': float(metrics.box.r[0]) if len(metrics.box.r) > 0 else 0,
                'f1_score': float(metrics.box.f1[0]) if len(metrics.box.f1) > 0 else 0,
            }

            self._emit_progress(100)
            self._emit_log("评估完成!")
            self._emit_log(f"mAP@0.5: {self.results['mAP50']:.4f}")
            self._emit_log(f"mAP@0.5:0.95: {self.results['mAP50-95']:.4f}")
            self._emit_log(f"Precision: {self.results['precision']:.4f}")
            self._emit_log(f"Recall: {self.results['recall']:.4f}")

        except Exception as e:
            self._emit_log(f"评估出错: {str(e)}")
        finally:
            self.is_running = False

    def export_model(self, model_path, format='onnx'):
        """导出模型"""
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            save_path = model.export(format=format)
            return True, str(save_path)
        except Exception as e:
            return False, str(e)

    def _emit_log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def _emit_progress(self, value):
        if self.progress_callback:
            self.progress_callback(value)
