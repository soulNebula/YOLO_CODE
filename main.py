#!/usr/bin/env python3
"""
YOLO CODE 模型训练平台
基于YOLO模型的标注+训练+推理一体化平台
"""

import sys
import os

# 将项目根目录添加到path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Windows DLL预加载：在导入torch之前配置DLL搜索路径
if sys.platform == 'win32' and hasattr(os, 'add_dll_directory'):
    _torch_dirs = []
    for _p in sys.path:
        _lib = os.path.join(_p, 'torch', 'lib')
        if os.path.isdir(_lib):
            _torch_dirs.append(_lib)
    if not _torch_dirs:
        try:
            import subprocess as _sp
            _res = _sp.run([sys.executable, '-m', 'pip', 'show', 'torch'],
                          capture_output=True, text=True, timeout=15)
            for _line in _res.stdout.split('\n'):
                if _line.startswith('Location:'):
                    _loc = _line.split(':', 1)[1].strip()
                    _lib = os.path.join(_loc, 'torch', 'lib')
                    if os.path.isdir(_lib):
                        _torch_dirs.append(_lib)
                    break
        except Exception:
            pass
    for _d in _torch_dirs:
        try:
            os.add_dll_directory(_d)
        except Exception:
            pass

# matplotlib 后端必须在 QApplication 之前设置
import matplotlib
matplotlib.use('Qt5Agg')

from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import Qt
from ui.main_window import MainWindow


def main():
    # 高DPI支持
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication(sys.argv)

    # 全局异常拦截 — 防止打包后反复弹窗
    def _global_exception_hook(exc_type, exc_value, exc_tb):
        import traceback
        msg = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
        print(msg, file=sys.stderr)
        # 只弹一次，不循环
        from PyQt5.QtWidgets import QMessageBox
        short = msg[-500:] if len(msg) > 500 else msg
        QMessageBox.critical(None, "运行错误",
            f"发生错误:\n{short}\n\n详细信息已输出到控制台。")
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _global_exception_hook

    app.setApplicationName("YOLO CODE")
    app.setOrganizationName("YOLO CODE")

    # 设置应用程序图标 (如果存在)
    icon_path = os.path.join(os.path.dirname(__file__), "icon.png")
    if os.path.exists(icon_path):
        from PyQt5.QtGui import QIcon
        app.setWindowIcon(QIcon(icon_path))

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
