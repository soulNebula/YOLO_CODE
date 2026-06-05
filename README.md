<p align="center">
  <img src="screenshots/logo.png" width="80" alt="YOLO CODE Logo" />
</p>

<h1 align="center">YOLO CODE 模型训练平台</h1>

<p align="center">
  <strong>基于 Ultralytics YOLO 的一体化深度学习平台</strong>
</p>

<p align="center">
  数据集管理 → 手动标注 → 模型训练 → 模型推理 → 模型评估 → 模型导出
</p>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.2-orange" alt="version" />
  <img src="https://img.shields.io/badge/python-3.10+-blue" alt="python" />
  <img src="https://img.shields.io/badge/license-MIT-green" alt="license" />
  <img src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey" alt="platform" />
</p>

---

## 📋 目录

- [功能模块](#功能模块)
- [系统要求](#系统要求)
- [安装指南](#安装指南)
- [使用流程](#使用流程)
- [界面预览](#界面预览)
- [标注快捷键](#标注快捷键)
- [模型导出格式](#模型导出格式)
- [项目结构](#项目结构)
- [常见问题](#常见问题)
- [联系方式](#联系方式)

---

## 📦 功能模块

| 模块 | 描述 |
|------|------|
| 📊 **仪表盘** | 项目概览、工作目录管理、系统状态、快捷操作 |
| 📝 **手动标注** | YOLO 格式标注，四角拖拽缩放，拖拽模式，自动保存 |
| 📁 **数据集** | 创建/导入数据集，训练/验证/测试集分割，多格式导出 |
| 🚀 **模型训练** | 环境检测修复、YOLO 训练、实时曲线、指标监控 |
| 🔍 **模型推理** | 图片/视频/摄像头检测，结果导出 CSV/JSON |
| 📈 **模型评估** | mAP/Precision/Recall 评估，模型对比分析 |
| 📤 **模型导出** | ONNX / TensorRT / OpenVINO / TFLite / TorchScript / CoreML |
| 💻 **终端** | Linux 风格命令行，支持 pip / Python / 系统命令 |

---

## 💻 系统要求

| 项目 | 要求 |
|------|------|
| **操作系统** | Windows 10+ / Linux / macOS |
| **Python** | 3.10 或更高 |
| **推荐硬件** | NVIDIA GPU（CUDA 支持）或 CPU |
| **内存** | 8GB 以上 |

---

## 🚀 安装指南

### 1. 克隆仓库

```bash
git clone https://github.com/HFGDHX/YOLO-CODE.git
cd YOLO-CODE
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 GPU 版 PyTorch（可选）

有 NVIDIA GPU 时推荐，也可在软件内通过 **模型训练 → 安装 PyTorch** 完成。

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. 启动

**三端通用** — Windows / Linux / macOS 统一入口：

```bash
python start.py
```

启动器会自动检测 Python 版本、检查依赖并安装缺失项，然后启动软件。

---

## 📖 使用流程

从零开始完成一个目标检测项目的完整步骤：

### 第一步：设置工作目录

启动软件后进入 **仪表盘**，点击蓝色 🏠 按钮设置工作目录。所有数据集、模型、标注、推理结果都将保存在此目录下。

### 第二步：准备数据集

![数据集模块](screenshots/数据集.png)

1. 切换到 **数据集** 模块
2. 输入数据集名称，点击 "创建数据集"
3. 点击 "导入数据" 选择图片文件夹，或直接将图片放入 `datasets/<名称>/images/` 下
4. 如需分割训练/验证集，设置比例后点击 "执行分割"

### 第三步：标注数据

![标注模块](screenshots/标注数据.png)

1. 切换到 **手动标注** 模块
2. 点击 "浏览" 选择数据集中的图片目录
3. 添加类别名称（如 `dog`、`cat`、`person`）
4. **左键拖拽** 绘制标注框
5. **选中标注框** 后拖拽四角手柄可缩放框大小
6. 按 **M** 进入拖拽模式拖移画面，按 **N** 回到标注模式
7. 右键或双击标注框可编辑类别
8. 按 **A / D** 或 **← / →** 切换图片，支持自动保存
9. 点击 "保存所有标注"，标注文件保存为 YOLO 格式到 `labels/` 目录

### 第四步：训练模型

![训练模块](screenshots/模型训练.png)

1. 切换到 **模型训练** 模块
2. 点击 "检测环境" 确认系统状态
3. 选择数据集的 `data.yaml` 文件
4. 选择模型（推荐 `yolov8n` 入门）
5. 调整参数后点击 "开始训练"
6. 在 **训练监控** 页查看实时曲线和指标
7. 训练完成后模型保存在 `runs/train/` 下

### 第五步：推理测试

![推理模块](screenshots/模型推理.png)

1. 切换到 **模型推理** 模块
2. 浏览选择训练好的 `.pt` 模型并加载
3. 选择输入源（图片 / 视频 / 摄像头）
4. 调整置信度阈值
5. 点击 "开始推理" 查看检测结果
6. 可导出结果为 CSV 或 JSON

### 第六步：评估与部署

![评估模块](screenshots/模型评估.png)

1. 在 **模型评估** 模块评估模型性能
2. 在 **模型导出** 模块将模型转换为部署格式（推荐 ONNX）

---

## 🖥 界面预览

### 仪表盘

![仪表盘](screenshots/仪表盘.png)

> 状态指示灯：🟢 绿色就绪 / 🔴 红色训练中

### 训练监控

训练页面包含三个子标签页：

| 子页 | 内容 |
|------|------|
| **训练配置** | 环境检测修复、数据集模型选择、训练参数 |
| **训练监控** | 实时 Loss/mAP 曲线、指标表格（当前值 / 最佳值） |
| **训练日志** | 终端风格全屏日志，支持导出和自动滚动 |

---

## ⌨ 标注快捷键

| 快捷键 | 功能 |
|--------|------|
| 鼠标左键拖拽 | 绘制标注框 |
| 拖拽四角手柄 | 缩放标注框 |
| 鼠标左键拖拽框 | 移动标注框 |
| 鼠标滚轮 | 缩放图片 |
| **M** / **N** | 拖拽模式 / 标注模式 |
| **A** / **D** 或 **←** / **→** | 上一张 / 下一张图片 |
| **Delete** | 删除选中标注框 |
| **Esc** | 取消当前绘制 / 取消选中 |
| 右键 / 双击框 | 编辑标注类别 |
| **Ctrl + C** / **Ctrl + V** | 复制 / 粘贴标注 |
| **Ctrl + D** | 复制选中标注框 |
| **Ctrl + F** / **Ctrl + Shift + F** | 适应窗口 / 适应宽度 |
| **Ctrl + 0** | 重置缩放至 100% |
| **Ctrl + H** | 隐藏 / 显示标注框 |
| **方向键** | 微移选中框（Shift 加速 ×10） |

---

## 📤 模型导出格式

| 格式 | 适用场景 |
|------|---------|
| **ONNX** | 跨平台推理（推荐） |
| **TensorRT** | NVIDIA GPU 极致加速 |
| **OpenVINO** | Intel CPU / GPU 优化 |
| **TFLite** | 移动端 / 嵌入式设备 |
| **TorchScript** | PyTorch 生态部署 |
| **CoreML** | Apple 设备部署 |

---

## 📁 项目结构

```
YOLO_CODE/
├── main.py                  # 程序入口
├── start.py                 # 跨平台启动器（自动检查依赖）
├── requirements.txt         # Python 依赖列表
├── README.md                # 项目文档
├── screenshots/             # 界面截图
├── ui/
│   ├── __init__.py
│   ├── main_window.py       # 主窗口 + 全部页面组件
│   └── annotation_canvas.py # 标注画布（渲染 + 交互）
├── core/
│   ├── __init__.py
│   ├── annotation.py        # 标注数据管理器
│   ├── training.py          # 训练管理 + 环境检测
│   ├── inference.py         # 推理管理
│   └── evaluation.py        # 评估管理
└── utils/
    ├── __init__.py
    ├── config.py             # 工作目录 + 自动扫描
    └── helpers.py            # 通用工具函数
```

---

## ❓ 常见问题

<details>
<summary><strong>PyTorch 加载失败 (c10.dll)</strong></summary>

软件内操作：**模型训练 → 环境操作 → 修复 PyTorch**
</details>

<details>
<summary><strong>GPU 不显示</strong></summary>

- 安装 NVIDIA 驱动
- **模型训练 → 安装 PyTorch** 选择 CUDA 版本
</details>

<details>
<summary><strong>训练显存不足</strong></summary>

- 减小批次大小（如 8 或 4）
- 减小图片尺寸（如 320）
- 使用更小模型（如 `yolov8n`）
</details>

<details>
<summary><strong>标注颜色异常</strong></summary>

图片颜色通道已校准，如仍有问题请检查图片格式。
</details>

<details>
<summary><strong>下载速度慢</strong></summary>

**模型训练 → 镜像源配置** 切换清华或阿里云镜像。
</details>

---

## 📧 联系方式

如有问题或建议，欢迎联系：

📮 **Email**: [2807087688@qq.com](mailto:2807087688@qq.com)

---

<p align="center">
  <strong>YOLO CODE</strong> v1.2 — Built with Ultralytics YOLO + PyQt5 + Matplotlib
</p>
