"""全局配置和工作目录管理"""
import os
import json


_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.yolo_code_config.json')

_default_config = {
    'work_dir': '',
    'last_data_yaml': '',
    'last_model': 'yolov8n',
    'last_work_dir': '',
}


def load_config():
    """加载配置"""
    if os.path.exists(_config_path):
        try:
            with open(_config_path, 'r', encoding='utf-8') as f:
                return {**_default_config, **json.load(f)}
        except Exception:
            pass
    return _default_config.copy()


def save_config(config):
    """保存配置"""
    try:
        with open(_config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def get_work_dir():
    """获取工作目录"""
    config = load_config()
    return config.get('work_dir', '') or os.getcwd()


def set_work_dir(path):
    """设置工作目录"""
    config = load_config()
    config['work_dir'] = os.path.abspath(path)
    save_config(config)


def auto_scan(work_dir=None):
    """自动扫描工作目录中的资源"""
    if work_dir is None:
        work_dir = get_work_dir()

    result = {
        'datasets': [],
        'models': [],
        'projects': [],
        'total_images': 0,
        'total_labels': 0,
    }

    if not os.path.isdir(work_dir):
        return result

    ext_set = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}

    # 辅助函数：统计目录中图片/标签
    def _count_media(directory):
        imgs, lbls = 0, 0
        if not os.path.isdir(directory):
            return 0, 0
        for root, _, files in os.walk(directory):
            for f in files:
                if f.lower().endswith(tuple(ext_set)):
                    imgs += 1
                elif f.endswith('.txt') and 'labels' in root.replace('\\', '/'):
                    lbls += 1
        return imgs, lbls

    def _scan_dir(search_dir):
        """扫描一个目录里的数据集和模型"""
        if not os.path.isdir(search_dir):
            return
        for entry in sorted(os.listdir(search_dir)):
            full = os.path.join(search_dir, entry)
            if not os.path.isdir(full):
                if entry.endswith('.pt'):
                    try:
                        size_mb = os.path.getsize(full) / (1024 * 1024)
                    except Exception:
                        size_mb = 0
                    result['models'].append({'name': entry, 'size': f'{size_mb:.1f} MB', 'path': full})
                continue

            has_images = os.path.isdir(os.path.join(full, 'images'))
            has_labels = os.path.isdir(os.path.join(full, 'labels'))
            has_yaml = os.path.exists(os.path.join(full, 'data.yaml'))

            img_count, lbl_count = 0, 0
            if has_images:
                img_count, _ = _count_media(os.path.join(full, 'images'))
                result['total_images'] += img_count
            if has_labels:
                _, lbl_count = _count_media(os.path.join(full, 'labels'))
                result['total_labels'] += lbl_count

            if has_images or has_labels or has_yaml:
                result['datasets'].append({
                    'name': entry,
                    'path': full,
                    'images': img_count,
                    'labels': lbl_count,
                    'has_yaml': has_yaml,
                })

    # 1. 扫描工作目录根级别（用户直接放图或 datasets/ 在根下）
    _scan_dir(work_dir)

    # 2. 扫描 datasets/ 子目录（用户在数据集页创建的都在这里）
    datasets_root = os.path.join(work_dir, 'datasets')
    if os.path.isdir(datasets_root):
        _scan_dir(datasets_root)

    # 3. 扫描 models/ 子目录
    models_root = os.path.join(work_dir, 'models')
    if os.path.isdir(models_root):
        for f in sorted(os.listdir(models_root)):
            if f.endswith('.pt'):
                full = os.path.join(models_root, f)
                if os.path.isfile(full):
                    try:
                        size_mb = os.path.getsize(full) / (1024 * 1024)
                    except Exception:
                        size_mb = 0
                    result['models'].append({'name': f, 'size': f'{size_mb:.1f} MB', 'path': full})

    # 4. 递归扫描 runs/ 中的训练产物（best.pt, last.pt）
    runs_root = os.path.join(work_dir, 'runs')
    seen_paths = {m['path'] for m in result['models']}
    if os.path.isdir(runs_root):
        for root, _, files in os.walk(runs_root):
            for f in sorted(files):
                if f.endswith('.pt'):
                    full = os.path.join(root, f)
                    if os.path.isfile(full) and full not in seen_paths:
                        # 用相对路径作为显示名，如 detect/train/weights/best.pt
                        rel = os.path.relpath(full, work_dir).replace('\\', '/')
                        try:
                            size_mb = os.path.getsize(full) / (1024 * 1024)
                        except Exception:
                            size_mb = 0
                        result['models'].append({
                            'name': rel, 'size': f'{size_mb:.1f} MB', 'path': full
                        })
                        seen_paths.add(full)

    return result


def remember(key, value):
    """记忆任意键值"""
    cfg = load_config()
    cfg[key] = value
    save_config(cfg)


def recall(key, default=None):
    """召回记忆的值"""
    cfg = load_config()
    return cfg.get(key, default)


def remember_last_data_yaml(path):
    remember('last_data_yaml', path)


def recall_last_data_yaml():
    return recall('last_data_yaml', '')


def remember_last_model(name):
    remember('last_model', name)


def recall_last_model():
    return recall('last_model', 'yolov8n')
