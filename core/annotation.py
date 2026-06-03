import os
import json
import cv2
import numpy as np
from pathlib import Path

class AnnotationManager:
    """标注管理器"""

    def __init__(self):
        self.current_image_path = None
        self.current_image = None
        self.annotations = {}  # {image_path: [bboxes]}
        self.classes = []
        self.class_colors = {}
        self.image_list = []
        self.current_index = -1
        self.dataset_dir = None

    def load_image_dir(self, dir_path):
        """加载目录中的所有图片"""
        self.dataset_dir = dir_path
        self.image_list = []
        ext_set = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        for f in sorted(os.listdir(dir_path)):
            if Path(f).suffix.lower() in ext_set:
                self.image_list.append(os.path.join(dir_path, f))

        if not self.image_list:
            # 也检查 images 子目录
            images_dir = os.path.join(dir_path, 'images')
            if os.path.isdir(images_dir):
                for f in sorted(os.listdir(images_dir)):
                    if Path(f).suffix.lower() in ext_set:
                        self.image_list.append(os.path.join(images_dir, f))

        self.annotations = {}
        self._load_existing_annotations()
        return len(self.image_list)

    def _load_existing_annotations(self):
        """加载已有的YOLO标注"""
        labels_dir = os.path.join(self.dataset_dir, 'labels') if self.dataset_dir else None
        if not labels_dir or not os.path.isdir(labels_dir):
            return

        for img_path in self.image_list:
            label_path = os.path.join(labels_dir, Path(img_path).stem + '.txt')
            if os.path.isfile(label_path):
                bboxes = []
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            cx, cy, w, h = map(float, parts[1:5])
                            bboxes.append({
                                'class_id': cls_id,
                                'cx': cx, 'cy': cy, 'w': w, 'h': h
                            })
                self.annotations[img_path] = bboxes

    def load_image(self, index):
        """加载指定索引的图片"""
        if 0 <= index < len(self.image_list):
            self.current_index = index
            self.current_image_path = self.image_list[index]
            img = cv2.imread(self.current_image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self.current_image = img
                return img
        return None

    def get_current_annotations(self):
        """获取当前图片的标注"""
        if self.current_image_path:
            return self.annotations.get(self.current_image_path, [])
        return []

    def add_bbox(self, class_id, x1, y1, x2, y2):
        """添加一个边界框"""
        if self.current_image is not None and self.current_image_path:
            h, w = self.current_image.shape[:2]
            # 转换为YOLO格式 (中心点归一化)
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h

            bbox = {
                'class_id': class_id,
                'cx': cx, 'cy': cy, 'w': bw, 'h': bh,
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            }
            if self.current_image_path not in self.annotations:
                self.annotations[self.current_image_path] = []
            self.annotations[self.current_image_path].append(bbox)
            return bbox
        return None

    def remove_bbox(self, index):
        """删除指定索引的边界框"""
        if self.current_image_path and self.current_image_path in self.annotations:
            annos = self.annotations[self.current_image_path]
            if 0 <= index < len(annos):
                annos.pop(index)
                return True
        return False

    def update_bbox(self, index, class_id, x1, y1, x2, y2):
        """更新边界框"""
        if self.current_image is not None and self.current_image_path:
            h, w = self.current_image.shape[:2]
            annos = self.annotations.get(self.current_image_path, [])
            if 0 <= index < len(annos):
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                annos[index] = {
                    'class_id': class_id,
                    'cx': cx, 'cy': cy, 'w': bw, 'h': bh,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                }
                return True
        return False

    def save_annotations(self):
        """保存标注为YOLO格式"""
        if not self.dataset_dir:
            return

        labels_dir = os.path.join(self.dataset_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        for img_path, bboxes in self.annotations.items():
            label_path = os.path.join(labels_dir, Path(img_path).stem + '.txt')
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    f.write(f"{bbox['class_id']} {bbox['cx']:.6f} {bbox['cy']:.6f} {bbox['w']:.6f} {bbox['h']:.6f}\n")

        # 也保存classes文件
        if self.classes:
            classes_path = os.path.join(self.dataset_dir, 'classes.txt')
            with open(classes_path, 'w', encoding='utf-8') as f:
                for cls_name in self.classes:
                    f.write(cls_name + '\n')

    def save_classes(self, classes):
        """保存类别列表"""
        self.classes = classes
        if self.dataset_dir:
            classes_path = os.path.join(self.dataset_dir, 'classes.txt')
            with open(classes_path, 'w', encoding='utf-8') as f:
                for cls_name in classes:
                    f.write(cls_name + '\n')

    def get_image_count(self):
        return len(self.image_list)

    def get_class_color(self, class_id):
        """为每个类别生成固定颜色"""
        import colorsys
        if class_id not in self.class_colors:
            hue = (class_id * 0.618) % 1.0
            r, g, b = colorsys.hsv_to_rgb(hue, 0.8, 0.95)
            self.class_colors[class_id] = (int(r), int(g), int(b))
        return self.class_colors[class_id]
