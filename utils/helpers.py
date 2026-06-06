import os
import yaml
from pathlib import Path

def get_yolo_classes_from_dataset(dataset_path):
    """从YOLO数据集的data.yaml读取类别"""
    yaml_path = None
    if os.path.isfile(os.path.join(dataset_path, 'data.yaml')):
        yaml_path = os.path.join(dataset_path, 'data.yaml')
    elif os.path.isfile(dataset_path) and dataset_path.endswith('.yaml'):
        yaml_path = dataset_path

    if yaml_path:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        return data.get('names', []), yaml_path
    return [], None

def save_yolo_dataset_config(dataset_path, train_path, val_path, classes, model_type='yolov8'):
    """保存YOLO数据集配置"""
    config = {
        'path': os.path.abspath(dataset_path),
        'train': train_path,
        'val': val_path,
        'nc': len(classes),
        'names': classes
    }
    yaml_path = os.path.join(dataset_path, 'data.yaml')
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
    return yaml_path

def get_supported_models():
    """返回支持的模型列表"""
    return [
        'yolov8n', 'yolov8s', 'yolov8m', 'yolov8l', 'yolov8x',
        'yolov8n-cls', 'yolov8s-cls', 'yolov8m-cls', 'yolov8l-cls',
        'yolov8n-seg', 'yolov8s-seg', 'yolov8m-seg', 'yolov8l-seg',
        'yolov8n-pose', 'yolov8s-pose', 'yolov8m-pose', 'yolov8l-pose',
        'yolov11n', 'yolov11s', 'yolov11m', 'yolov11l', 'yolov11x',
        'yolov5nu', 'yolov5su', 'yolov5mu', 'yolov5lu',
    ]

def get_default_training_params():
    """返回默认训练参数"""
    return {
        'epochs': 100,
        'batch_size': 16,
        'img_size': 640,
        'learning_rate': 0.01,
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 3,
        'patience': 50,
        'device': 'auto',
        'workers': 4,
        'pretrained': True,
        'augment': True,
        'cos_lr': False,
        'close_mosaic': 10,
        'optimizer': 'auto',
    }

def format_time(seconds):
    """格式化时间显示"""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{int(seconds // 60)}m {int(seconds % 60)}s"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def find_latest_checkpoint(project='runs', name='train'):
    """查找最近的 last.pt 和 best.pt

    返回 (last_pt_path, best_pt_path, run_dir) 或 (None, None, None)
    自动递增查找: train/ train2/ train3/ ...
    """
    import os
    base = os.path.abspath(project)

    # 从 train/ 开始递增查找
    for suffix in ['', '2', '3', '4', '5', '6', '7', '8', '9'] + [str(i) for i in range(10, 50)]:
        run_dir = os.path.join(base, f'{name}{suffix}')
        weights_dir = os.path.join(run_dir, 'weights')
        last_pt = os.path.join(weights_dir, 'last.pt')
        if os.path.isfile(last_pt):
            best_pt = os.path.join(weights_dir, 'best.pt')
            return last_pt, best_pt if os.path.isfile(best_pt) else None, run_dir
        if not os.path.isdir(run_dir):
            # 该目录不存在 → 可用于新训练
            return None, None, run_dir
    return None, None, os.path.join(base, f'{name}50')


def get_next_run_dir(project='runs', name='train'):
    """返回下一个可用的运行目录（自动递增）"""
    _, _, run_dir = find_latest_checkpoint(project, name)
    return run_dir
