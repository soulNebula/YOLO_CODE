#!/usr/bin/env python3
"""YOLO CODE 跨平台启动器 — Windows / Linux / macOS 通用"""

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def check_python():
    v = sys.version_info
    if v.major < 3 or (v.major == 3 and v.minor < 10):
        print(f"[错误] 需要 Python 3.10+，当前: {v.major}.{v.minor}.{v.micro}")
        input("按 Enter 退出...")
        sys.exit(1)
    print(f"[OK] Python {v.major}.{v.minor}.{v.micro}")

    # Python 3.13+ 警告
    if v.minor >= 13:
        print("[警告] Python 3.13+ 部分库兼容性仍在完善，建议使用 3.11 或 3.12")


def check_deps():
    """检查并安装缺失依赖，失败时给出明确操作指引"""
    deps = {
        'PyQt5':        ('PyQt5',        'pip install PyQt5'),
        'cv2':          ('opencv-python', 'pip install opencv-python'),
        'numpy':        ('numpy',         'pip install numpy'),
        'yaml':         ('PyYAML',        'pip install PyYAML'),
        'psutil':       ('psutil',        'pip install psutil'),
        'matplotlib':   ('matplotlib',    'pip install matplotlib'),
        'ultralytics':  ('ultralytics',   'pip install ultralytics'),
    }

    missing = []
    for mod, (pkg, install_cmd) in deps.items():
        try:
            __import__(mod)
            print(f"[OK] {mod}")
        except ImportError:
            print(f"[缺失] {mod}")
            missing.append((pkg, install_cmd))

    if not missing:
        print()
        return

    print()
    print("─" * 45)
    print("  以下依赖缺失:")
    for pkg, cmd in missing:
        print(f"    {pkg}")
    print()
    print("  自动安装中...")
    print("─" * 45)

    r = subprocess.run(
        [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
        capture_output=False
    )
    if r.returncode != 0:
        print()
        print("=" * 45)
        print("  安装失败，请手动执行以下命令:")
        print()
        for pkg, cmd in missing:
            print(f"    {cmd}")
        print()
        print("  或一键安装全部依赖:")
        print("    pip install -r requirements.txt")
        print("=" * 45)
        input("\n按 Enter 退出...")
        sys.exit(1)

    print("\n[完成] 依赖安装成功\n")


def check_torch():
    """检查 PyTorch 和 CUDA"""
    try:
        import torch
        ver = torch.__version__
        cuda = "可用" if torch.cuda.is_available() else "不可用"
        cuda_ver = torch.version.cuda if torch.cuda.is_available() else "N/A"
        print(f"[OK] PyTorch {ver}  |  CUDA: {cuda} ({cuda_ver})")
    except ImportError:
        print("[提示] 未安装 PyTorch（GPU 训练需要）")
        print("      安装命令: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")


def launch():
    print("\n正在启动 YOLO CODE...")
    r = subprocess.run([sys.executable, 'main.py'])
    if r.returncode != 0:
        print()
        print("=" * 45)
        print("  启动异常")
        print()
        print("  排查步骤:")
        print("    1. 检查依赖: pip install -r requirements.txt")
        print("    2. 修复 PyTorch: pip install torch --force-reinstall")
        print("    3. 查看详细错误: python main.py")
        print("=" * 45)
        input("\n按 Enter 退出...")


if __name__ == '__main__':
    print("=" * 45)
    print("  YOLO CODE 模型训练平台 — 启动器")
    print("=" * 45)
    check_python()
    check_deps()
    check_torch()
    launch()
