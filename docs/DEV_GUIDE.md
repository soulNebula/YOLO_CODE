# YOLO CODE 二次开发指南

## 项目架构

```
YOLO_CODE/
├── main.py                # 程序入口（PyQt5 初始化 + main window）
├── start.py               # 跨平台启动器
├── requirements.txt       # 依赖锁定版本
├── README.md
├── docs/                  # 文档
│   ├── API.md
│   ├── FAQ.md
│   ├── INSTALL.md
│   ├── TROUBLESHOOTING.md
│   └── DEV_GUIDE.md
├── ui/
│   ├── __init__.py
│   ├── main_window.py     # 所有 UI 组件（Dashboard/Annotation/Training/etc）
│   └── annotation_canvas.py # ImageCanvas 标注画布（纯渲染层）
├── core/
│   ├── __init__.py
│   ├── annotation.py      # AnnotationManager 标注数据模型
│   ├── training.py        # TrainingManager + TrainingWorker(QThread)
│   ├── inference.py       # InferenceManager + VideoWorker(QThread)
│   └── evaluation.py      # EvalWorker + EvalComparisonWorker
└── utils/
    ├── __init__.py
    ├── config.py           # 全局配置和工作目录管理
    ├── helpers.py          # 辅助函数（YAML/数据集/checkpoint）
    └── validator.py        # 数据集验证 + 标注质量检查
```

## 架构原则

### 1. 单一数据源
- `AnnotationManager` 是所有标注数据的唯一入口
- `ImageCanvas` 不直接修改数据，只通过信号通知控制器
- 数据流：Canvas → Signal → Widget → Manager

### 2. QThread 模式
- 训练/推理/评估均在 `QThread` 中运行
- 通过 `pyqtSignal` 跨线程安全通信
- 绝不从工作线程直接操作 Qt Widget

### 3. 信号驱动
```python
# Canvas（视图层）
class ImageCanvas(QLabel):
    bboxDrawn = pyqtSignal(int, int, int, int)
    bboxDeleteRequested = pyqtSignal(int)
    bboxMoveFinished = pyqtSignal(int, int, int, int, int)
    # ...

# Widget（控制器层）
class AnnotationWidget(QWidget):
    def _init_ui(self):
        self.canvas.bboxDrawn.connect(self._on_bbox_drawn)
        self.canvas.bboxDeleteRequested.connect(self._on_bbox_delete_requested)
```

## 如何添加新功能

### 添加新的 Widget 页面

1. 在 `ui/main_window.py` 中创建 Widget 类
2. 在 `MainWindow._init_ui` 中实例化
3. 添加到 `self.content_stack` 并分配索引
4. 在侧边栏添加导航按钮

### 添加新的 core 模块

1. 在 `core/` 下创建 `.py` 文件
2. 长时间运行的操作放在 `QThread` 子类中
3. 用 `pyqtSignal` 暴露进度和结果
4. 在对应的 Widget 中连接信号

### 添加新的工具函数

1. 在 `utils/` 下创建 `.py` 文件
2. 保持无状态、纯函数风格
3. 在需要的地方 import

## QSS 样式

### 添加新样式

```python
# 在 main_window.py 顶部定义
MY_STYLE = """
QPushButton#myButton {
    background-color: #ff6b00;
    color: white;
}
"""
```

### 主题系统

```python
# 获取当前主题样式
from ui.main_window import get_current_style, toggle_theme

style = get_current_style()  # 'light' 或 'dark' 对应的完整 QSS
new_theme = toggle_theme()   # 切换并返回新主题名
```

## 测试

### 手动测试清单

- [ ] 启动：`python start.py`
- [ ] 标注：导入图片 → 画框 → 保存 → 确认 .txt 文件
- [ ] 训练：选择 yaml → 选模型 → 开始训练 → 查看监控
- [ ] 推理：加载模型 → 选图片 → 推理 → 查看结果
- [ ] 评估：选模型 → 选数据集 → 评估 → 查看指标
- [ ] 主题：Ctrl+T 切换深色主题

### 常见开发错误

| 错误 | 原因 | 修复 |
|------|------|------|
| `Set changed size during iteration` | 迭代 set 时修改了它 | 用 `list(set)` 拷贝后再迭代 |
| `QWidget: Must construct...` | 在工作线程操作 UI | 用信号回主线程 |
| `colorsys.hsv_to_rgb` 返回黑色 | 返回值是 0-1 范围 | 乘以 255 |
| `QShortcut` 不触发 | 子控件拦截事件 | 使用 `WidgetWithChildren` 上下文 |
