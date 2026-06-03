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


def check_deps():
    deps = ['PyQt5', 'ultralytics', 'cv2', 'numpy', 'matplotlib']
    missing = []
    for d in deps:
        try:
            __import__(d)
            print(f"[OK] {d}")
        except ImportError:
            print(f"[缺失] {d}")
            missing.append(d)

    if missing:
        print("\n正在安装缺失依赖...")
        r = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
            capture_output=False
        )
        if r.returncode != 0:
            print("[错误] 安装失败，请手动执行: pip install -r requirements.txt")
            input("按 Enter 退出...")
            sys.exit(1)
        print("[完成] 依赖安装成功\n")
    else:
        print()


def launch():
    print("正在启动 YOLO CODE...")
    subprocess.run([sys.executable, 'main.py'])


if __name__ == '__main__':
    print("=" * 45)
    print("  YOLO CODE 模型训练平台 — 启动器")
    print("=" * 45)
    check_python()
    check_deps()
    launch()
