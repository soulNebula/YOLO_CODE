"""数据集标注验证器 —— 训练前检查标注合法性"""
import os
import yaml
from pathlib import Path
from collections import defaultdict

__all__ = ['validate_dataset', 'validate_yaml_consistency', 'auto_generate_yaml',
           'DatasetValidationResult', 'YamlConsistencyResult']


class Issue:
    """单条问题"""
    __slots__ = ('file_path', 'issue_type', 'detail', 'label_file', 'bbox_index')

    def __init__(self, file_path, issue_type, detail='', label_file='', bbox_index=-1):
        self.file_path = file_path          # 图片或标签路径
        self.issue_type = issue_type        # 问题类型
        self.detail = detail                # 描述
        self.label_file = label_file        # 关联的标签文件
        self.bbox_index = bbox_index        # 标注框索引（-1 表示图片级问题）


class DatasetValidationResult:
    """验证结果"""
    def __init__(self):
        self.errors = []        # Issue 列表
        self.warnings = []      # 警告列表
        self.nc = 0
        self.names = []
        self.total_images = 0
        self.total_labels = 0
        self.image_paths = []   # 所有图片路径（供自动修复用）

    @property
    def has_errors(self):
        return len(self.errors) > 0

    @property
    def error_summary(self):
        """按类型统计"""
        counts = defaultdict(int)
        for e in self.errors:
            counts[e.issue_type] += 1
        return dict(counts)


def validate_dataset(data_yaml_path):
    """验证数据集标注，返回 DatasetValidationResult"""
    result = DatasetValidationResult()

    # 1. 解析 data.yaml
    if not os.path.isfile(data_yaml_path):
        result.errors.append(Issue(data_yaml_path, 'yaml_missing', 'data.yaml 文件不存在'))
        return result

    with open(data_yaml_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    result.nc = nc = int(cfg.get('nc', 0))
    result.names = names = cfg.get('names', [])

    base_path = cfg.get('path', os.path.dirname(data_yaml_path))
    if not os.path.isabs(base_path):
        base_path = os.path.join(os.path.dirname(data_yaml_path), base_path)
    base_path = os.path.abspath(base_path)

    # 2. 收集图片目录
    image_dirs = []
    for key in ('train', 'val'):
        rel = cfg.get(key, '')
        if not rel:
            continue
        full = rel if os.path.isabs(rel) else os.path.join(base_path, rel)
        full = os.path.abspath(full)
        if os.path.isdir(full):
            image_dirs.append(full)

    if not image_dirs:
        result.errors.append(Issue(data_yaml_path, 'no_image_dir', '未找到 train/val 图片目录'))
        return result

    # 3. 收集所有图片
    ext_set = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
    image_map = {}  # {stem: image_path}

    for img_dir in image_dirs:
        for f in sorted(os.listdir(img_dir)):
            if Path(f).suffix.lower() in ext_set:
                full = os.path.join(img_dir, f)
                stem = Path(f).stem
                # 同名图片以第一个为准
                if stem not in image_map:
                    image_map[stem] = full

    result.image_paths = list(image_map.values())
    result.total_images = len(result.image_paths)

    if result.total_images == 0:
        result.errors.append(Issue(data_yaml_path, 'no_images', '图片目录中无图片文件'))
        return result

    # 4. 找标签目录
    labels_dir = os.path.join(base_path, 'labels')
    if not os.path.isdir(labels_dir):
        # 也尝试 image_dir/../labels
        labels_dir = os.path.join(os.path.dirname(image_dirs[0]), 'labels') if image_dirs else None
        if not labels_dir or not os.path.isdir(labels_dir):
            result.warnings.append(Issue(
                base_path, 'no_labels_dir',
                f'未找到 labels 目录（{labels_dir}），所有图片将被视为无标注'
            ))
            return result

    # 5. 逐图片检查
    for stem, img_path in sorted(image_map.items()):
        label_path = os.path.join(labels_dir, stem + '.txt')

        # 5a. 无标签文件
        if not os.path.isfile(label_path):
            result.errors.append(Issue(
                img_path, 'missing_label',
                '缺少标注文件',
                label_file=label_path
            ))
            continue

        # 5b. 读标签内容
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 5c. 空标签
        if len(lines) == 0 or all(not l.strip() for l in lines):
            result.errors.append(Issue(
                img_path, 'empty_label',
                '标注文件为空（无目标）',
                label_file=label_path
            ))
            continue

        result.total_labels += 1

        # 5d. 逐行检查
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                result.errors.append(Issue(
                    img_path, 'bad_format',
                    f'标注行 {i+1} 字段不足：{line[:80]}',
                    label_file=label_path, bbox_index=i
                ))
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            except ValueError:
                result.errors.append(Issue(
                    img_path, 'bad_number',
                    f'标注行 {i+1} 数值非法：{line[:80]}',
                    label_file=label_path, bbox_index=i
                ))
                continue

            # class_id 越界
            if nc > 0 and cls_id >= nc:
                result.errors.append(Issue(
                    img_path, 'class_id_oob',
                    f'class_id={cls_id} >= nc={nc}（{names[cls_id] if cls_id < len(names) else "未知"}）',
                    label_file=label_path, bbox_index=i
                ))

            # 坐标越界
            for name, val in [('cx', cx), ('cy', cy), ('w', bw), ('h', bh)]:
                if not (0 < val <= 1):
                    result.errors.append(Issue(
                        img_path, 'coord_oob',
                        f'{name}={val:.6f} 不在 (0,1] 内',
                        label_file=label_path, bbox_index=i
                    ))
                    break

            # 宽高为0
            if bw <= 0 or bh <= 0:
                result.errors.append(Issue(
                    img_path, 'zero_size',
                    f'w={bw:.6f} h={bh:.6f} 无效',
                    label_file=label_path, bbox_index=i
                ))

    return result


def auto_fix_issues(result):
    """自动修复：删除有问题的标注行/文件，返回修复统计"""
    fixed_lines = 0
    removed_files = 0

    for e in result.errors:
        if e.bbox_index >= 0 and os.path.isfile(e.label_file):
            # 删除单行
            with open(e.label_file, 'r') as f:
                lines = f.readlines()
            new_lines = []
            for i, line in enumerate(lines):
                if i != e.bbox_index or not line.strip():
                    new_lines.append(line)
                else:
                    fixed_lines += 1
            with open(e.label_file, 'w') as f:
                f.writelines(new_lines)

        elif e.issue_type in ('empty_label', 'bad_format', 'bad_number') and os.path.isfile(e.label_file):
            # 整个标签文件有问题 → 删除
            os.remove(e.label_file)
            removed_files += 1

    return {'fixed_lines': fixed_lines, 'removed_files': removed_files}


# ── YAML 一致性校验 ──────────────────────────────────────────

class YamlConsistencyResult:
    """YAML 一致性校验结果"""
    def __init__(self):
        self.ok = True
        self.issues = []        # [(level, message)]
        self.nc = 0
        self.names = []
        self.found_class_ids = set()   # 标注中实际出现的 class_id
        self.missing_classes = []      # names 中声明但标注中未出现的

    def add_error(self, msg):
        self.issues.append(('error', msg))
        self.ok = False

    def add_warning(self, msg):
        self.issues.append(('warning', msg))

    def errors(self):
        return [m for l, m in self.issues if l == 'error']

    def warnings(self):
        return [m for l, m in self.issues if l == 'warning']


def validate_yaml_consistency(data_yaml_path):
    """校验 data.yaml 内部一致性 + names 与标注中实际 class_id 的匹配

    返回 YamlConsistencyResult
    """
    result = YamlConsistencyResult()

    if not os.path.isfile(data_yaml_path):
        result.add_error("文件不存在")
        return result

    # 1. 解析 YAML
    try:
        with open(data_yaml_path, 'r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        result.add_error(f"YAML 解析失败: {e}")
        return result

    nc = cfg.get('nc', 0) if cfg else 0
    names = cfg.get('names', []) if cfg else []
    result.nc = nc if isinstance(nc, int) else 0
    result.names = names if isinstance(names, list) else []

    # 2. nc 必须与 len(names) 一致
    if nc != len(names):
        result.add_error(f"nc={nc} 但 names 有 {len(names)} 个条目（应一致）")

    # 3. 解析路径
    base = cfg.get('path', os.path.dirname(data_yaml_path)) if cfg else ''
    if not os.path.isabs(base):
        base = os.path.join(os.path.dirname(data_yaml_path), base)
    base = os.path.abspath(base) if os.path.isdir(base) else os.path.dirname(os.path.abspath(data_yaml_path))

    # 4. 收集所有标注中出现的 class_id
    labels_dir = os.path.join(base, 'labels')
    if not os.path.isdir(labels_dir):
        # 也尝试父目录
        alt = os.path.join(os.path.dirname(base), 'labels')
        if os.path.isdir(alt):
            labels_dir = alt

    found_ids = set()
    if os.path.isdir(labels_dir):
        for fname in os.listdir(labels_dir):
            if not fname.endswith('.txt'):
                continue
            fpath = os.path.join(labels_dir, fname)
            try:
                with open(fpath, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(parts[0])
                                found_ids.add(cls_id)
                            except ValueError:
                                pass
            except Exception:
                pass

    result.found_class_ids = found_ids

    if not found_ids and os.path.isdir(labels_dir):
        result.add_warning("标注文件存在但未找到有效的 class_id")

    # 5. 标注中出现的 class_id 超出 names 范围
    if names and found_ids:
        oob_ids = found_ids - set(range(len(names)))
        if oob_ids:
            result.add_error(
                f"标注中出现超出范围的 class_id: {sorted(oob_ids)} "
                f"(names 只有 {len(names)} 个类别)"
            )

    # 6. names 中有未在标注中出现的类别
    if names and found_ids:
        result.missing_classes = [i for i in range(len(names)) if i not in found_ids]
        if result.missing_classes:
            missing_names = [names[i] for i in result.missing_classes]
            result.add_warning(
                f"以下类别在标注中从未出现: {missing_names}"
            )

    # 7. 无 names 但有标注
    if not names and found_ids:
        result.add_error(
            f"data.yaml 中未定义 names，但标注中出现了这些 class_id: {sorted(found_ids)}"
        )

    return result


def auto_generate_yaml(dataset_dir, train_rel='images/train', val_rel='images/val',
                       model_type='yolov8', overwrite=False):
    """从数据集目录自动生成 data.yaml

    扫描逻辑：
    1. 读取 classes.txt → 得到类别名
    2. 如果没有 classes.txt，扫描 labels/ 中的 class_id → 生成 cls_0, cls_1...
    3. 自动检测 train/val 路径
    4. 写入 data.yaml

    返回 (yaml_path, YamlConsistencyResult)
    """
    if not os.path.isdir(dataset_dir):
        return None, None

    dataset_dir = os.path.abspath(dataset_dir)
    yaml_path = os.path.join(dataset_dir, 'data.yaml')

    if os.path.exists(yaml_path) and not overwrite:
        # 已存在，只校验
        return yaml_path, validate_yaml_consistency(yaml_path)

    # 1. 读取/推断类别
    classes_txt = os.path.join(dataset_dir, 'classes.txt')
    classes = []
    if os.path.isfile(classes_txt):
        with open(classes_txt, 'r', encoding='utf-8') as f:
            classes = [line.strip() for line in f if line.strip()]

    if not classes:
        # 从 labels/ 扫描最大 class_id
        labels_dir = os.path.join(dataset_dir, 'labels')
        max_id = -1
        if os.path.isdir(labels_dir):
            for fname in os.listdir(labels_dir):
                if not fname.endswith('.txt'):
                    continue
                try:
                    with open(os.path.join(labels_dir, fname), 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                cls_id = int(parts[0])
                                max_id = max(max_id, cls_id)
                except Exception:
                    pass
        if max_id >= 0:
            classes = [f"class_{i}" for i in range(max_id + 1)]
        else:
            classes = ['class_0']  # 默认

    # 2. 检测 train/val 路径
    def _resolve_img_dir(sub):
        full = os.path.join(dataset_dir, sub)
        if os.path.isdir(full):
            return sub
        return None

    train = _resolve_img_dir(train_rel)
    if not train:
        for d in ['images/train', 'images', 'train']:
            t = _resolve_img_dir(d)
            if t:
                train = t
                break

    val = _resolve_img_dir(val_rel)
    if not val:
        for d in ['images/val', 'images', 'val']:
            v = _resolve_img_dir(d)
            if v:
                val = v
                break
    if not val and train:
        val = train  # 没有独立的 val 则复⽤ train

    if not train:
        return None, None  # 没有图片目录，无法生成

    # 3. 写 data.yaml
    config = {
        'path': dataset_dir,
        'train': train,
        'val': val,
        'nc': len(classes),
        'names': classes,
    }

    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    # 4. 校验生成结果
    consistency = validate_yaml_consistency(yaml_path)
    return yaml_path, consistency
