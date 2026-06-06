"""模型评估管理器 —— 全面评估 + 混淆矩阵 + 多模型对比"""
import os
import numpy as np
from pathlib import Path
from PyQt5.QtCore import QThread, pyqtSignal


class EvalWorker(QThread):
    """QThread 评估工作线程"""
    progress = pyqtSignal(int)
    log = pyqtSignal(str)
    finished = pyqtSignal(dict)
    error = pyqtSignal(str)

    def __init__(self, model_path, data_yaml, conf=0.25, iou=0.45):
        super().__init__()
        self.model_path = model_path
        self.data_yaml = data_yaml
        self.conf = conf
        self.iou = iou

    def run(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.error.emit("未安装 ultralytics")
            return

        try:
            self.log.emit(f"加载模型: {self.model_path}")
            self.progress.emit(10)
            model = YOLO(self.model_path)
            class_names = list(model.names.values()) if model.names else []

            self.progress.emit(20)
            self.log.emit("开始评估...")

            metrics = model.val(
                data=self.data_yaml,
                conf=self.conf, iou=self.iou,
                verbose=False, plots=True
            )

            self.progress.emit(60)

            # ── 汇总指标 ──
            result = {
                'model_path': self.model_path,
                'model_name': Path(self.model_path).stem,
                'data_yaml': self.data_yaml,
                'conf': self.conf, 'iou': self.iou,
                'mAP50': float(metrics.box.map50) if hasattr(metrics.box, 'map50') else 0,
                'mAP50_95': float(metrics.box.map) if hasattr(metrics.box, 'map') else 0,
            }

            # ── Per-class 指标 ──
            per_class = []
            if hasattr(metrics.box, 'ap_class_index') and metrics.box.ap_class_index is not None:
                ap_indices = metrics.box.ap_class_index.cpu().numpy().astype(int)
                map50_per = metrics.box.map50.cpu().numpy() if hasattr(metrics.box.map50, 'cpu') else None
                map_per = metrics.box.map.cpu().numpy() if hasattr(metrics.box.map, 'cpu') else None

                for i, idx in enumerate(ap_indices):
                    p = float(metrics.box.p[idx]) if idx < len(metrics.box.p) else 0
                    r = float(metrics.box.r[idx]) if idx < len(metrics.box.r) else 0
                    f1 = float(metrics.box.f1[idx]) if idx < len(metrics.box.f1) else 0
                    ap50_v = float(map50_per[i]) if map50_per is not None and i < len(map50_per) else 0
                    ap_v = float(map_per[i]) if map_per is not None and i < len(map_per) else 0
                    name = class_names[idx] if idx < len(class_names) else f"cls_{idx}"
                    per_class.append({
                        'class_id': idx, 'name': name,
                        'Precision': p, 'Recall': r, 'F1': f1,
                        'mAP50': ap50_v, 'mAP50-95': ap_v
                    })

            # 总体 P/R/F1
            p_all = float(metrics.box.p.mean()) if len(metrics.box.p) > 0 else 0
            r_all = float(metrics.box.r.mean()) if len(metrics.box.r) > 0 else 0
            f1_all = float(metrics.box.f1.mean()) if len(metrics.box.f1) > 0 else 0
            result['precision'] = p_all
            result['recall'] = r_all
            result['f1'] = f1_all
            result['per_class'] = per_class
            result['class_names'] = class_names

            # ── 混淆矩阵 ──
            cm = None
            cm_path = os.path.join(model.trainer.save_dir, 'confusion_matrix.png') if hasattr(model, 'trainer') and model.trainer and hasattr(model.trainer, 'save_dir') else None
            if cm_path and os.path.isfile(cm_path):
                result['confusion_matrix'] = cm_path
            try:
                if hasattr(metrics, 'confusion_matrix') and metrics.confusion_matrix is not None:
                    cm = metrics.confusion_matrix.matrix.cpu().numpy() if hasattr(metrics.confusion_matrix, 'matrix') else None
                    if cm is not None:
                        result['confusion_matrix_data'] = cm.tolist()
            except Exception:
                pass

            self.progress.emit(85)

            # ── 漏检/误检分析 ──
            error_samples = _collect_error_samples(model, self.data_yaml, self.conf, self.iou)
            result['error_samples'] = error_samples

            self.progress.emit(100)
            self.log.emit("评估完成!")
            self.log.emit(f"mAP@0.5: {result['mAP50']:.4f}  |  mAP@0.5:0.95: {result['mAP50_95']:.4f}")
            self.log.emit(f"Precision: {p_all:.4f}  |  Recall: {r_all:.4f}  |  F1: {f1_all:.4f}")
            if per_class:
                for pc in per_class:
                    self.log.emit(f"  {pc['name']}: P={pc['Precision']:.3f} R={pc['Recall']:.3f} F1={pc['F1']:.3f} mAP50={pc['mAP50']:.3f}")

            self.finished.emit(result)

        except Exception as e:
            self.error.emit(f"评估出错: {e}")


def _collect_error_samples(model, data_yaml, conf, iou, max_samples=20):
    """收集漏检和误检样例"""
    import cv2
    samples = {'fp': [], 'fn': [], 'tp_low': [], 'fp_high': []}

    try:
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.utils import DATASETS_DIR
    except ImportError:
        return samples

    try:
        import yaml
        with open(data_yaml, 'r') as f:
            cfg = yaml.safe_load(f)

        base = cfg.get('path', os.path.dirname(data_yaml))
        if not os.path.isabs(base):
            base = os.path.join(os.path.dirname(data_yaml), base)
        base = os.path.abspath(base)

        # 从 val 集中取前 N 张
        val_dir = cfg.get('val', '')
        if not os.path.isabs(val_dir):
            val_dir = os.path.join(base, val_dir)
        if not os.path.isdir(val_dir):
            return samples

        ext_set = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = [f for f in sorted(os.listdir(val_dir)) if Path(f).suffix.lower() in ext_set]
        if not images:
            return samples

        model_nc = len(model.names)
        import random
        test_images = random.sample(images, min(max_samples, len(images)))

        for fname in test_images:
            img_path = os.path.join(val_dir, fname)
            img = cv2.imread(img_path)
            if img is None:
                continue
            h, w = img.shape[:2]

            results = model(img, conf=conf, iou=iou, verbose=False)
            r = results[0]
            preds = []
            if r.boxes is not None:
                boxes = r.boxes.xyxy.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                clss = r.boxes.cls.cpu().numpy().astype(int)
                preds = [(clss[i], float(confs[i]), boxes[i]) for i in range(len(boxes))]

            # 读 GT
            labels_dir = os.path.join(base, 'labels')
            label_path = os.path.join(labels_dir, Path(fname).stem + '.txt')
            gts = []
            if os.path.isfile(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                            x1 = int((cx - bw/2) * w)
                            y1 = int((cy - bh/2) * h)
                            x2 = int((cx + bw/2) * w)
                            y2 = int((cy + bh/2) * h)
                            gts.append((cls_id, x1, y1, x2, y2))

            # 简单 IoU
            def iou_box(a, b):
                ax1, ay1, ax2, ay2 = a[1], a[2], a[3], a[4]
                bx1, by1, bx2, by2 = b[1], b[2], b[3], b[4]
                ix1, iy1 = max(ax1, bx1), max(ay1, by1)
                ix2, iy2 = min(ax2, bx2), min(ay2, by2)
                ia = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - ia
                return ia / ua if ua > 0 else 0

            matched_gt = set()
            matched_pred = set()

            for gi, gt in enumerate(gts):
                best_iou = 0
                best_pi = -1
                for pi, pred in enumerate(preds):
                    if pred[0] != gt[0]:
                        continue
                    iou_v = iou_box((gt[0], gt[1], gt[2], gt[3], gt[4]),
                                    (pred[0], pred[1], pred[2], pred[3]))
                    if iou_v > best_iou:
                        best_iou = iou_v
                        best_pi = pi
                if best_iou >= 0.5:
                    matched_gt.add(gi)
                    matched_pred.add(best_pi)

            # FN: GT 未匹配
            for gi, gt in enumerate(gts):
                if gi not in matched_gt:
                    name = model.names.get(gt[0], f"cls_{gt[0]}") if model.names else str(gt[0])
                    samples['fn'].append({
                        'image': img_path, 'class': name,
                        'bbox': [int(gt[1]), int(gt[2]), int(gt[3]), int(gt[4])],
                        'reason': '漏检（未检测到）'
                    })

            # FP: 预测未匹配 + 高/低置信
            for pi, pred in enumerate(preds):
                if pi in matched_pred:
                    continue
                name = model.names.get(pred[0], f"cls_{pred[0]}") if model.names else str(pred[0])
                entry = {
                    'image': img_path, 'class': name,
                    'bbox': [int(pred[2][0]), int(pred[2][1]), int(pred[2][2]), int(pred[2][3])],
                    'conf': pred[1]
                }
                if pred[1] > 0.5:
                    entry['reason'] = '高置信误检'
                    samples['fp_high'].append(entry)
                else:
                    samples['fp'].append(entry)

            # TP low conf
            for pi, pred in enumerate(preds):
                if pi in matched_pred and pred[1] < 0.4:
                    name = model.names.get(pred[0], f"cls_{pred[0]}") if model.names else str(pred[0])
                    samples['tp_low'].append({
                        'image': img_path, 'class': name,
                        'bbox': [int(pred[2][0]), int(pred[2][1]), int(pred[2][2]), int(pred[2][3])],
                        'conf': pred[1], 'reason': '低置信正确检测'
                    })

    except Exception:
        pass

    return samples


# ── 多模型对比 ──────────────────────────────────────────────

class EvalComparisonWorker(QThread):
    """多模型对比工作线程"""
    progress = pyqtSignal(int, int)     # current, total
    log = pyqtSignal(str)
    finished = pyqtSignal(list)          # list of result dicts
    error = pyqtSignal(str)

    def __init__(self, model_paths, data_yaml, conf=0.25, iou=0.45):
        super().__init__()
        self.model_paths = model_paths
        self.data_yaml = data_yaml
        self.conf = conf
        self.iou = iou

    def run(self):
        results = []
        total = len(self.model_paths)
        for i, mp in enumerate(self.model_paths):
            self.log.emit(f"[{i+1}/{total}] 评估: {mp}")
            worker = EvalWorker(mp, self.data_yaml, self.conf, self.iou)
            worker.run()  # 同步（对比工作线程内逐模型串行）
            if hasattr(worker, 'result_cache'):
                results.append(worker.result_cache)
            else:
                self.log.emit(f"  跳过（评估失败）")
            self.progress.emit(i + 1, total)
        self.finished.emit(results)


class EvaluationManager:
    """模型评估管理器"""

    def __init__(self):
        self.is_running = False
        self._worker = None
        self.results = None
        self.all_results = []  # 多模型结果
        self.log_callback = None
        self.progress_callback = None

    def set_callbacks(self, log_callback=None, progress_callback=None):
        self.log_callback = log_callback
        self.progress_callback = progress_callback

    def evaluate(self, model_path, data_yaml, conf=0.25, iou=0.45):
        if self.is_running:
            return False
        self.is_running = True
        self._worker = EvalWorker(model_path, data_yaml, conf, iou)
        self._worker.log.connect(self._on_log)
        self._worker.progress.connect(self._on_progress)
        self._worker.finished.connect(self._on_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        return True

    def compare_models(self, model_paths, data_yaml, conf=0.25, iou=0.45):
        if self.is_running:
            return False
        self.is_running = True
        self.all_results = []
        self._worker = EvalComparisonWorker(model_paths, data_yaml, conf, iou)
        self._worker.log.connect(self._on_log)
        self._worker.progress.connect(lambda c, t: self._on_progress(int(c/t*100)))
        self._worker.finished.connect(self._on_compare_finished)
        self._worker.error.connect(self._on_error)
        self._worker.start()
        return True

    def export_model(self, model_path, format='onnx'):
        try:
            from ultralytics import YOLO
            model = YOLO(model_path)
            save_path = model.export(format=format)
            return True, str(save_path)
        except Exception as e:
            return False, str(e)

    def _on_log(self, msg):
        if self.log_callback:
            self.log_callback(msg)

    def _on_progress(self, val):
        if self.progress_callback:
            self.progress_callback(val)

    def _on_finished(self, result):
        self.results = result
        self.is_running = False

    def _on_compare_finished(self, results):
        self.all_results = results
        self.results = results[0] if results else None
        self.is_running = False

    def _on_error(self, msg):
        if self.log_callback:
            self.log_callback(msg)
        self.is_running = False
