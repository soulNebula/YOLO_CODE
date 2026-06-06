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

    for entry in os.listdir(work_dir):
        full = os.path.join(work_dir, entry)
        if not os.path.isdir(full):
            # .pt 模型文件
            if entry.endswith('.pt'):
                size_mb = os.path.getsize(full) / (1024 * 1024)
                result['models'].append({'name': entry, 'size': f'{size_mb:.1f} MB', 'path': full})
            continue

        # 检测数据集目录
        has_images = os.path.isdir(os.path.join(full, 'images'))
        has_labels = os.path.isdir(os.path.join(full, 'labels'))
        has_yaml = os.path.exists(os.path.join(full, 'data.yaml'))

        img_count = 0
        lbl_count = 0

        if has_images:
            img_dir = os.path.join(full, 'images')
            # 递归统计图片
            for root, _, files in os.walk(img_dir):
                for f in files:
                    if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_count += 1
                        result['total_images'] += 1

        if has_labels:
            lbl_dir = os.path.join(full, 'labels')
            for root, _, files in os.walk(lbl_dir):
                for f in files:
                    if f.endswith('.txt'):
                        lbl_count += 1
                        result['total_labels'] += 1

        if has_images or has_labels or has_yaml:
            result['datasets'].append({
                'name': entry,
                'path': full,
                'images': img_count,
                'labels': lbl_count,
                'has_yaml': has_yaml,
            })

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
