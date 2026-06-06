# YOLO CODE 安装配置指南

## 系统要求

| 项目 | 最低 | 推荐 |
|------|------|------|
| 操作系统 | Windows 10 / Linux / macOS | Windows 11 |
| Python | 3.10+ | 3.11 或 3.12 |
| 内存 | 8 GB | 16 GB+ |
| GPU | 无（CPU 训练可行） | NVIDIA GPU 6GB+ 显存（CUDA） |
| 磁盘 | 2 GB | 10 GB+（存放数据集和模型） |

## 安装步骤

### 1. 安装 Python

从 [python.org](https://python.org) 下载 Python 3.10~3.12。

**注意**：Python 3.13+ 部分库兼容性仍在完善，建议 3.11/3.12。

勾选 "Add Python to PATH" 后安装。

验证：
```bash
python --version
```

### 2. 下载项目

```bash
git clone <repo-url>
cd YOLO_CODE
```

或直接解压项目压缩包到目标目录。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 安装 GPU 版 PyTorch（可选，强烈推荐）

```bash
# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

也可在软件内 "模型训练 → 环境操作 → 安装PyTorch" 完成。

### 5. 启动

```bash
python start.py
```

启动器自动检测 Python 版本、检查依赖并安装缺失项。

## 配置指南

### 工作目录

启动后进入仪表盘，点击蓝色 🏠 按钮设置工作目录。

所有数据集、模型、标注、推理结果保存在工作目录下。

推荐结构：
```
D:/YOLO_Projects/
├── datasets/
│   ├── my_project/
│   │   ├── images/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── labels/
│   │   │   ├── train/
│   │   │   └── val/
│   │   ├── classes.txt
│   │   └── data.yaml
├── models/
│   └── yolov8n.pt
└── inference_results/
```

### 镜像源配置

训练页 → 镜像源配置 → 选择清华或阿里云镜像。

加速 PyTorch 和 ultralytics 下载。

### 记住上次设置

软件自动记住上次使用的 data.yaml 和模型文件路径，重启后恢复。

## 验证安装

启动后检查状态栏：
```
🟢 就绪 | GPU: NVIDIA GeForce RTX 4050 | Python 3.12 | PyTorch 2.1 | CUDA 12.1
```

训练页 → 检测环境 → 确认全部 OK。

## 常见安装问题

参见 [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
