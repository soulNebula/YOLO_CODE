# YOLO CODE API 文档

## 核心模块

### `core.annotation.AnnotationManager`

标注管理——单一数据源，所有标注变更的唯一入口。

```python
from core.annotation import AnnotationManager

mgr = AnnotationManager()

# 加载图片目录
count = mgr.load_image_dir("/path/to/dataset")

# 加载图片
img = mgr.load_image(0)  # 加载索引0的图片

# 获取当前标注
annos = mgr.get_current_annotations()

# 添加标注框（像素坐标）
mgr.add_bbox(class_id=0, x1=100, y1=100, x2=300, y2=300)

# 更新标注框
mgr.update_bbox(index=0, class_id=1, x1=110, y1=100, x2=290, y2=310)

# 删除标注框
mgr.remove_bbox(index=0)

# 微移标注框
mgr.micro_move(index=0, dx=5, dy=0)

# 复制上一帧标注
mgr.copy_from_prev_image()

# 保存标注
mgr.save_annotations()

# 类别管理
mgr.classes.append("car")
mgr.delete_class(0)  # 删除并重排 class_id
```

### `core.training.TrainingManager`

训练管理——管理 QThread 训练线程生命周期。

```python
from core.training import TrainingManager, TrainingWorker

mgr = TrainingManager()

# 设置回调
mgr.set_callbacks(
    progress_callback=my_progress_fn,
    log_callback=my_log_fn,
    metrics_callback=my_metrics_fn
)

# 开始训练
mgr.start_training({
    'model': 'yolov8n',
    'data_yaml': '/path/to/data.yaml',
    'epochs': 100,
    'batch_size': 16,
    'img_size': 640,
    'learning_rate': 0.01,
    'device': 'auto',
    'workers': 4,
    'optimizer': 'auto',
    'close_mosaic': 10,
    'rect': False,
    'seed': 0,
    'project': 'runs',
    'name': 'train',
    'resume': False,
})

# 停止训练
mgr.stop_training()

# 获取指标
metrics = mgr.get_training_metrics()
```

### `core.inference.InferenceManager`

推理管理——图片/视频/摄像头推理。

```python
from core.inference import InferenceManager, VideoWorker

mgr = InferenceManager()

# 加载模型
success, msg = mgr.load_model("/path/to/best.pt")

# 设置参数
mgr.conf_threshold = 0.25
mgr.iou_threshold = 0.45
mgr.max_det = 300
mgr.agnostic_nms = False
mgr.half_prec = True

# 图片推理
result, error = mgr.run_image(
    "/path/to/image.jpg",
    imgsz=640, half=True, augment=False, max_det=300,
    agnostic_nms=False, device=None
)
annotated_img, detections = result  # if no error

# 视频推理
mgr.run_video("/path/to/video.mp4", frame_callback, vid_stride=1)

# 摄像头推理
mgr.run_camera(0, frame_callback)

# 停止
mgr.stop()
```

### `core.evaluation.EvalWorker`

评估工作线程。

```python
from core.evaluation import EvalWorker, EvalComparisonWorker

worker = EvalWorker(
    model_path="/path/to/best.pt",
    data_yaml="/path/to/data.yaml",
    conf=0.25, iou=0.45
)
worker.progress.connect(my_progress_fn)
worker.log.connect(my_log_fn)
worker.finished.connect(my_result_fn)
worker.error.connect(my_error_fn)
worker.start()
```

## 工具模块

### `utils.validator`

```python
from utils.validator import (
    validate_dataset,         # 验证数据集标注
    validate_yaml_consistency, # 校验 data.yaml
    auto_generate_yaml,        # 自动生成 data.yaml
    check_annotation_quality,  # 标注质量检查
)
```

### `utils.config`

```python
from utils.config import (
    get_work_dir, set_work_dir,
    remember, recall,
    remember_last_data_yaml, recall_last_data_yaml,
    remember_last_model, recall_last_model,
)
```

### `utils.helpers`

```python
from utils.helpers import (
    get_yolo_classes_from_dataset,
    save_yolo_dataset_config,
    get_supported_models,
    get_default_training_params,
    find_latest_checkpoint,
    format_time,
)
```
