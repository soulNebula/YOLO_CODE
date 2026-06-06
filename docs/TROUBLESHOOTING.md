# YOLO CODE 故障排查指南

## 启动问题

### `python start.py` 闪退

```bash
python main.py  # 直接启动查看完整错误信息
```

常见原因：
- Python 版本 < 3.10
- 缺少依赖，执行 `pip install -r requirements.txt`

### ImportError: No module named 'PyQt5'

```bash
pip install PyQt5
```

### ImportError: No module named 'ultralytics'

```bash
pip install ultralytics
```

## PyTorch 问题

### c10.dll 加载失败

软件内：模型训练 → 环境操作 → 修复PyTorch。

或手动：
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### GPU 不显示

1. 安装 NVIDIA 驱动：[nvidia.com/drivers](https://nvidia.com/drivers)
2. 安装 CUDA 版 PyTorch（见安装指南）
3. 验证：`python -c "import torch; print(torch.cuda.is_available())"`

### 训练显存不足 (OOM)

```
RuntimeError: CUDA out of memory
```

解决方案：
- 减小批次大小：`batch=8` 或 `batch=4`
- 减小图片尺寸：`imgsz=320`
- 使用更小模型：`yolov8n` 而非 `yolov8x`
- 关闭数据增强：取消勾选"数据增强"

## 标注问题

### 标注颜色异常

已修复：`colorsys.hsv_to_rgb` 返回值需乘 255。v1.2+ 正常。

如仍有问题，检查图片是否为 RGB（非 BGR）格式。

### A/D 键无反应

v1.3+ 已修复。使用 QShortcut 确保键盘事件不被子控件拦截。

### 标注保存后坐标错误

v1.3+ 已修复数据同步 Bug。确保使用最新版本。

## 训练问题

### 训练不收敛 / loss 为 NaN

1. 运行标注验证（自动提示）
2. 检查 `class_id >= nc`
3. 检查坐标是否在 (0,1] 范围内
4. 降低学习率（如 0.001）
5. 检查数据集是否正常

### 训练突然停止

- 检查显存：`nvidia-smi` 查看 GPU 内存使用
- 检查日志：训练页日志面板显示详细错误

### 数据增强预览不显示

- 确保已选择有效的 data.yaml
- 确保训练图片目录存在且包含图片

## 推理问题

### 无检测结果

软件会自动弹窗提示降低 confidence 阈值。

默认 conf=0.25，尝试降为 0.1 或 0.05。

### 推理速度慢

- 启用半精度（FP16）：勾选"半精度"
- 使用 ONNX 导出后推理（2-3x 加速）
- 检查设备：确认使用 GPU 而非 CPU

## 评估问题

### 评估卡住不动

- 检查 data.yaml 路径是否正确
- 检查 val 目录是否存在且包含图片
- 确保模型和数据集类别数匹配

## 其他

### 下载速度慢

模型训练 → 镜像源配置 → 切换清华/阿里云镜像。

### 中文路径乱码

避免在路径中使用中文字符。Python 在 Windows 上对非 ASCII 路径有已知兼容性问题。

### 配置文件损坏

删除项目根目录的 `.yolo_code_config.json`，软件会自动重建。
