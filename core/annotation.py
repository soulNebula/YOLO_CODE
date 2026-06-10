import os
import cv2
import numpy as np
from pathlib import Path
from colorsys import hsv_to_rgb


class AnnotationManager:
    """标注管理器 —— 单一数据源，所有标注变更的唯一入口"""

    def __init__(self):
        self.current_image_path = None
        self.current_image = None
        self.annotations = {}       # {image_path: [bbox dicts]}
        self.classes = []           # 类别名列表，索引 = class_id
        self.class_colors = {}      # {class_id: (R, G, B)}
        self.image_list = []        # 所有图片路径
        self.current_index = -1
        self.dataset_dir = None

        # 变更追踪
        self.dirty_images = set()
        self.clipboard = []

        # 图片缓存（LRU，最多 20 张常用图）
        self._img_cache = {}        # {path: img_array}
        self._cache_order = []      # MRU 顺序

    # ── 目录加载 ─────────────────────────────────────────────

    def load_image_dir(self, dir_path):
        """加载目录中的所有图片，自动检测数据集根目录"""
        ext_set = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
        self.image_list = []

        # 智能检测数据集根目录
        # 如果打开的是 images/ 子目录，始终以父目录为 dataset_dir
        # （即使是新数据集还没有 labels/ 或 classes.txt，也应该把标注保存到父目录的 labels/ 下）
        parent = os.path.dirname(os.path.abspath(dir_path))
        if os.path.basename(os.path.abspath(dir_path)) == 'images':
            self.dataset_dir = parent
            images_dir = dir_path
        else:
            self.dataset_dir = dir_path
            images_dir = dir_path

        # 收集图片
        for f in sorted(os.listdir(images_dir)):
            if Path(f).suffix.lower() in ext_set:
                self.image_list.append(os.path.join(images_dir, f))

        if not self.image_list:
            alt = os.path.join(dir_path, 'images')
            if os.path.isdir(alt):
                self.dataset_dir = dir_path
                for f in sorted(os.listdir(alt)):
                    if Path(f).suffix.lower() in ext_set:
                        self.image_list.append(os.path.join(alt, f))

        self.annotations = {}
        self.dirty_images = set()
        self._load_existing_annotations()
        self._load_existing_classes()
        return len(self.image_list)

    def _load_existing_annotations(self):
        """加载已有的YOLO标注"""
        labels_dir = (
            os.path.join(self.dataset_dir, 'labels')
            if self.dataset_dir else None
        )
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

    def _load_existing_classes(self):
        """加载已有的 classes.txt"""
        if not self.dataset_dir:
            return
        classes_path = os.path.join(self.dataset_dir, 'classes.txt')
        if os.path.isfile(classes_path):
            with open(classes_path, 'r', encoding='utf-8') as f:
                self.classes = [line.strip() for line in f if line.strip()]
            # 为已加载的类别生成颜色
            for cls_id in range(len(self.classes)):
                if cls_id not in self.class_colors:
                    self.class_colors[cls_id] = self._gen_color(cls_id)

    # ── 图片加载 ─────────────────────────────────────────────

    def load_image(self, index):
        """加载指定索引的图片（带缓存）"""
        if 0 <= index < len(self.image_list):
            self.current_index = index
            self.current_image_path = self.image_list[index]
            # 先查缓存
            img = self._img_cache.get(self.current_image_path)
            if img is not None:
                # 移到 MRU 头部
                if self.current_image_path in self._cache_order:
                    self._cache_order.remove(self.current_image_path)
                self._cache_order.append(self.current_image_path)
                self.current_image = img
                return img
            # 缓存未命中，从磁盘加载
            img = cv2.imread(self.current_image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                self._add_to_cache(self.current_image_path, img)
                self.current_image = img
                return img
        return None

    def _add_to_cache(self, path, img):
        """加入缓存，超出上限时淘汰最旧的"""
        if len(self._cache_order) >= 20:
            old = self._cache_order.pop(0)
            self._img_cache.pop(old, None)
        self._img_cache[path] = img
        self._cache_order.append(path)

    def clear_image_cache(self):
        """清理图片缓存（释放内存）"""
        self._img_cache.clear()
        self._cache_order.clear()
        self.current_image = None
        import gc; gc.collect()

    # ── 标注查询 ─────────────────────────────────────────────

    def get_current_annotations(self):
        """获取当前图片的标注（深拷贝，防止外部直接修改内部数据）"""
        if self.current_image_path:
            annos = self.annotations.get(self.current_image_path, [])
            result = []
            for a in annos:
                bbox = dict(a)
                if 'x1' not in bbox:
                    h, w = self.current_image.shape[:2]
                    bbox.update({
                        'x1': int((bbox['cx'] - bbox['w']/2) * w),
                        'y1': int((bbox['cy'] - bbox['h']/2) * h),
                        'x2': int((bbox['cx'] + bbox['w']/2) * w),
                        'y2': int((bbox['cy'] + bbox['h']/2) * h),
                    })
                result.append(bbox)
            return result
        return []

    def get_bbox_class(self, index):
        """轻量级：获取指定标注的 class_id（不深拷贝整个列表）"""
        annos = self.annotations.get(self.current_image_path, [])
        if 0 <= index < len(annos):
            return annos[index].get('class_id', 0)
        return 0

    def get_bbox_count(self):
        """轻量级：获取当前图片标注数量"""
        annos = self.annotations.get(self.current_image_path, [])
        return len(annos)

    def get_image_count(self):
        return len(self.image_list)

    @staticmethod
    def _yolo_to_pixel(bbox, img_w, img_h):
        cx, cy, bw, bh = bbox['cx'], bbox['cy'], bbox['w'], bbox['h']
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        return x1, y1, x2, y2

    # ── 标注变更（全部调用 mark_dirty） ────────────────────────

    def add_bbox(self, class_id, x1, y1, x2, y2):
        """添加边界框"""
        if self.current_image is not None and self.current_image_path:
            h, w = self.current_image.shape[:2]
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
            self.mark_dirty()
            return bbox
        return None

    def remove_bbox(self, index):
        """删除指定索引的边界框"""
        if self.current_image_path and self.current_image_path in self.annotations:
            annos = self.annotations[self.current_image_path]
            if 0 <= index < len(annos):
                annos.pop(index)
                self.mark_dirty()
                return True
        return False

    def update_bbox(self, index, class_id, x1, y1, x2, y2):
        """更新边界框（坐标 + 类别）"""
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
                self.mark_dirty()
                return True
        return False

    def micro_move(self, index, dx, dy):
        """微移选中框（仅像素坐标）"""
        if self.current_image is not None and self.current_image_path:
            h, w = self.current_image.shape[:2]
            annos = self.annotations.get(self.current_image_path, [])
            if 0 <= index < len(annos):
                ann = annos[index]
                x1, y1, x2, y2 = self._yolo_to_pixel(ann, w, h)
                x1 += dx; y1 += dy; x2 += dx; y2 += dy
                cx = (x1 + x2) / 2.0 / w
                cy = (y1 + y2) / 2.0 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                annos[index].update({
                    'cx': cx, 'cy': cy, 'w': bw, 'h': bh,
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
                })
                self.mark_dirty()
                return True
        return False

    def duplicate_bbox(self, index):
        """复制指定标注框（偏移 20px）"""
        if self.current_image is not None and self.current_image_path:
            h, w = self.current_image.shape[:2]
            annos = self.annotations.get(self.current_image_path, [])
            if 0 <= index < len(annos):
                src = annos[index]
                x1, y1, x2, y2 = self._yolo_to_pixel(src, w, h)
                x1 += 20; y1 += 20; x2 += 20; y2 += 20
                return self.add_bbox(src['class_id'], x1, y1, x2, y2)
        return None

    def copy_annotations(self):
        """复制当前图片全部标注到剪贴板"""
        self.clipboard = [
            dict(a) for a in self.annotations.get(self.current_image_path, [])
        ]

    def paste_annotations(self):
        """从剪贴板粘贴标注到当前图片"""
        if not self.clipboard or self.current_image is None:
            return 0
        h, w = self.current_image.shape[:2]
        count = 0
        for src in self.clipboard:
            if 'x1' in src:
                self.add_bbox(
                    src['class_id'], src['x1'], src['y1'], src['x2'], src['y2']
                )
            else:
                x1, y1, x2, y2 = self._yolo_to_pixel(src, w, h)
                self.add_bbox(src['class_id'], x1, y1, x2, y2)
            count += 1
        return count

    def copy_from_prev_image(self):
        """复制上一张图片的标注到当前图片（视频抽帧标注）"""
        if not self.image_list or self.current_index <= 0:
            return False
        prev_path = self.image_list[self.current_index - 1]
        prev_annos = self.annotations.get(prev_path, [])
        if not prev_annos:
            return False
        for ann in prev_annos:
            bbox = dict(ann)
            if 'x1' not in bbox and self.current_image is not None:
                h, w = self.current_image.shape[:2]
                x1, y1, x2, y2 = self._yolo_to_pixel(bbox, w, h)
                bbox.update({'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2})
            if self.current_image_path not in self.annotations:
                self.annotations[self.current_image_path] = []
            self.annotations[self.current_image_path].append(bbox)
        self.mark_dirty()
        return True

    def discard_current_image(self):
        """标记当前图片为废弃（写入 discarded.txt）"""
        if not self.dataset_dir or not self.current_image_path:
            return False
        discard_file = os.path.join(self.dataset_dir, 'discarded.txt')
        img_name = os.path.basename(self.current_image_path)
        if os.path.isfile(discard_file):
            with open(discard_file, 'r', encoding='utf-8') as f:
                existing = {line.strip() for line in f}
        else:
            existing = set()
        if img_name in existing:
            return False  # 已在废弃列表
        with open(discard_file, 'a', encoding='utf-8') as f:
            f.write(img_name + '\n')
        return True

    def undiscard_current_image(self):
        """取消当前图片的废弃标记（从 discarded.txt 删除）"""
        if not self.dataset_dir or not self.current_image_path:
            return False
        discard_file = os.path.join(self.dataset_dir, 'discarded.txt')
        if not os.path.isfile(discard_file):
            return False
        img_name = os.path.basename(self.current_image_path)
        with open(discard_file, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f if l.strip() != img_name]
        if len(lines) == len([l for l in open(discard_file).read().strip().split('\n') if l]):
            return False  # 不在列表中
        with open(discard_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines) + ('\n' if lines else ''))
        return True

    def is_discarded(self, image_path=None):
        """检查图片是否已被标记为废弃"""
        if not self.dataset_dir:
            return False
        path = image_path or self.current_image_path
        if not path:
            return False
        discard_file = os.path.join(self.dataset_dir, 'discarded.txt')
        if not os.path.isfile(discard_file):
            return False
        img_name = os.path.basename(path)
        with open(discard_file, 'r', encoding='utf-8') as f:
            return img_name in {line.strip() for line in f}

    # ── 类别管理 ─────────────────────────────────────────────

    def delete_class(self, index):
        """删除类别并重排所有 class_id"""
        if not (0 <= index < len(self.classes)):
            return False

        deleted_name = self.classes.pop(index)

        # 重排所有标注中的 class_id
        for img_path, bboxes in self.annotations.items():
            new_bboxes = []
            for b in bboxes:
                if b['class_id'] == index:
                    # 属于被删类别的标注，丢弃
                    continue
                elif b['class_id'] > index:
                    b = dict(b)
                    b['class_id'] -= 1
                new_bboxes.append(b)
            self.annotations[img_path] = new_bboxes

        # 重排 class_colors
        new_colors = {}
        for cls_id, color in self.class_colors.items():
            if cls_id > index:
                new_colors[cls_id - 1] = color
            elif cls_id < index:
                new_colors[cls_id] = color
        self.class_colors = new_colors

        self.mark_dirty_all()
        return True

    # ── 变更追踪 ─────────────────────────────────────────────

    def mark_dirty(self, image_path=None):
        """标记当前图片有未保存变更"""
        path = image_path or self.current_image_path
        if path:
            self.dirty_images.add(path)

    def mark_clean(self, image_path=None):
        """标记图片已保存"""
        path = image_path or self.current_image_path
        self.dirty_images.discard(path)

    def mark_dirty_all(self):
        """标记所有图片为脏"""
        self.dirty_images.update(self.annotations.keys())

    def is_dirty(self, image_path=None):
        """检查是否有未保存变更"""
        path = image_path or self.current_image_path
        return path in self.dirty_images

    def has_any_dirty(self):
        """检查是否有任何未保存变更"""
        return len(self.dirty_images) > 0

    # ── 统计 ─────────────────────────────────────────────────

    def get_stats(self, image_path=None):
        """获取每类标注数量统计 {class_id: count}"""
        path = image_path or self.current_image_path
        annos = self.annotations.get(path, [])
        stats = {}
        for a in annos:
            cls_id = a.get('class_id', 0)
            stats[cls_id] = stats.get(cls_id, 0) + 1
        return stats

    # ── 保存 ─────────────────────────────────────────────────

    def save_annotations(self):
        """保存所有标注为YOLO格式，同时删除已标记为废弃的图片"""
        if not self.dataset_dir:
            return

        # ── 先删除已被标记为废弃的图片和标注 ──
        purged = self._purge_discarded()

        labels_dir = os.path.join(self.dataset_dir, 'labels')
        os.makedirs(labels_dir, exist_ok=True)

        # 只保存脏文件，无脏文件时保存全部（兼容手动点击"保存所有"）
        targets = list(self.dirty_images) if self.dirty_images else list(self.annotations.keys())

        for img_path in targets:
            bboxes = self.annotations.get(img_path, [])
            if not bboxes:
                continue
            label_path = os.path.join(labels_dir, Path(img_path).stem + '.txt')
            with open(label_path, 'w') as f:
                for bbox in bboxes:
                    f.write(
                        f"{bbox['class_id']} "
                        f"{bbox['cx']:.6f} {bbox['cy']:.6f} "
                        f"{bbox['w']:.6f} {bbox['h']:.6f}\n"
                    )
            self.mark_clean(img_path)

        # 保存类别文件
        if self.classes:
            classes_path = os.path.join(self.dataset_dir, 'classes.txt')
            with open(classes_path, 'w', encoding='utf-8') as f:
                for cls_name in self.classes:
                    f.write(cls_name + '\n')

        return purged

    def _purge_discarded(self):
        """删除 discarded.txt 中记录的所有图片和对应标注文件，返回删除数量"""
        discard_file = os.path.join(self.dataset_dir, 'discarded.txt')
        if not os.path.isfile(discard_file):
            return 0

        with open(discard_file, 'r', encoding='utf-8') as f:
            discarded_names = {line.strip() for line in f if line.strip()}

        if not discarded_names:
            return 0

        labels_dir = os.path.join(self.dataset_dir, 'labels')
        purged = 0

        for img_name in discarded_names:
            img_stem = Path(img_name).stem

            # 删除图片（在所有可能的图片目录中查找）
            for search_dir in (os.path.join(self.dataset_dir, 'images'), self.dataset_dir):
                if not os.path.isdir(search_dir):
                    continue
                for root, _, files in os.walk(search_dir):
                    for f in files:
                        if f == img_name or Path(f).stem == img_stem:
                            img_path = os.path.join(root, f)
                            try:
                                os.remove(img_path)
                                purged += 1
                            except OSError:
                                pass

            # 删除对应标注文件
            for search_dir in (labels_dir, os.path.join(self.dataset_dir, 'images', 'labels')):
                if not os.path.isdir(search_dir):
                    continue
                for root, _, files in os.walk(search_dir):
                    lbl_name = img_stem + '.txt'
                    if lbl_name in files:
                        try:
                            os.remove(os.path.join(root, lbl_name))
                        except OSError:
                            pass

            # 清理内存中的记录
            for img_path in list(self.annotations.keys()):
                if Path(img_path).stem == img_stem:
                    del self.annotations[img_path]
                    self.dirty_images.discard(img_path)
            self.image_list = [p for p in self.image_list if Path(p).stem != img_stem]

        # 清空 discarded.txt
        with open(discard_file, 'w', encoding='utf-8') as f:
            f.write('')

        # 清理缓存
        purged_stems = {Path(n).stem for n in discarded_names}
        for img_path in list(self._img_cache.keys()):
            if Path(img_path).stem in purged_stems:
                del self._img_cache[img_path]
                if img_path in self._cache_order:
                    self._cache_order.remove(img_path)

        return purged

    def save_classes(self, classes):
        """保存类别列表"""
        self.classes = classes
        if self.dataset_dir:
            classes_path = os.path.join(self.dataset_dir, 'classes.txt')
            with open(classes_path, 'w', encoding='utf-8') as f:
                for cls_name in classes:
                    f.write(cls_name + '\n')

    # ── 颜色生成 ─────────────────────────────────────────────

    @staticmethod
    def _gen_color(class_id):
        """黄金比例色相生成固定颜色"""
        hue = (class_id * 0.618) % 1.0
        r, g, b = hsv_to_rgb(hue, 0.8, 0.95)
        return (int(r * 255), int(g * 255), int(b * 255))

    def get_class_color(self, class_id):
        """获取类别颜色，按需生成"""
        if class_id not in self.class_colors:
            self.class_colors[class_id] = self._gen_color(class_id)
        return self.class_colors[class_id]
