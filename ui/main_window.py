import sys
import os
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QPushButton, QFileDialog, QMessageBox,
    QStatusBar, QAction, QMenuBar, QMenu, QSplitter, QFrame,
    QGroupBox, QCheckBox, QProgressBar, QTextEdit, QLineEdit,
    QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QTreeWidget,
    QTreeWidgetItem, QTableWidget, QTableWidgetItem, QListWidget,
    QListWidgetItem, QScrollArea, QToolBar, QToolButton, QSizePolicy,
    QHeaderView, QRadioButton, QButtonGroup, QGridLayout, QDialog,
    QStackedWidget, QShortcut
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt5.QtGui import (
    QPixmap, QImage, QPainter, QPen, QColor, QFont,
    QIcon, QFontDatabase, QPalette, QCursor
)
import numpy as np
import cv2
import threading
from pathlib import Path
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from core.annotation import AnnotationManager
from ui.annotation_canvas import ImageCanvas
from core.training import TrainingManager
from core.inference import InferenceManager
from core.evaluation import EvaluationManager
from utils.helpers import (
    get_yolo_classes_from_dataset, save_yolo_dataset_config,
    get_supported_models, get_default_training_params,
    find_latest_checkpoint
)
from utils.validator import validate_dataset, auto_fix_issues, \
    validate_yaml_consistency, auto_generate_yaml
from utils.config import remember_last_data_yaml, recall_last_data_yaml, \
    remember_last_model, recall_last_model, get_work_dir
from core.training import get_pip_mirrors


ORANGE_WHITE_STYLE = """
QMainWindow {
    background-color: #fafafa;
}
QWidget {
    background-color: #fafafa;
    color: #333333;
    font-size: 15px;
    font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
}
QTabWidget::pane {
    border: 1px solid #e0e0e0;
    background-color: #ffffff;
}
QTabBar::tab {
    background-color: #f5f5f5;
    color: #666666;
    padding: 12px 20px;
    margin-right: 1px;
    border: 1px solid #e0e0e0;
    font-weight: bold;
}
QTabBar::tab:selected {
    background-color: #ffffff;
    color: #ff6b00;
    border-bottom: 3px solid #ff6b00;
}
QTabBar::tab:hover:!selected {
    background-color: #fff3e6;
    color: #ff8c00;
}
QGroupBox {
    border: 1px solid #e8e8e8;
    border-radius: 6px;
    margin-top: 10px;
    padding: 14px 10px 10px 10px;
    font-weight: normal;
    font-size: 15px;
    color: #555555;
    background-color: #ffffff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: #ff6b00;
    font-size: 14px;
}
QPushButton {
    background-color: #ffffff;
    color: #ff6b00;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    padding: 5px 14px;
    font-size: 14px;
    font-weight: normal;
    min-width: 60px;
}
QPushButton:hover {
    background-color: #fff3e6;
    color: #e65c00;
    border-color: #ff6b00;
}
QPushButton:pressed {
    background-color: #ffe0b3;
}
QPushButton:disabled {
    background-color: #f5f5f5;
    color: #cccccc;
    border-color: #e0e0e0;
}
QPushButton#btnTrain {
    background-color: #ff6b00;
    color: #ffffff;
    border: none;
    font-size: 15px;
    padding: 7px 22px;
}
QPushButton#btnTrain:hover {
    background-color: #e65c00;
    border: none;
}
QPushButton#btnStop {
    background-color: #999999;
    color: #ffffff;
    border: none;
}
QPushButton#btnStop:hover {
    background-color: #777777;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    border-radius: 3px;
    padding: 5px 8px;
    color: #333333;
}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
    border-color: #ff6b00;
}
QComboBox::drop-down {
    border: none;
    padding-right: 8px;
}
QComboBox QAbstractItemView {
    background-color: #ffffff;
    border: 1px solid #d0d0d0;
    color: #333333;
    selection-background-color: #fff3e6;
    selection-color: #333333;
}
QTreeWidget, QListWidget, QTableWidget {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    color: #333333;
    alternate-background-color: #fafafa;
}
QTreeWidget::item:selected, QListWidget::item:selected, QTableWidget::item:selected {
    background-color: #fff3e6;
    color: #ff6b00;
}
QTreeWidget::item:hover, QListWidget::item:hover {
    background-color: #fff9f2;
}
QHeaderView::section {
    background-color: #ff8c00;
    color: #ffffff;
    padding: 5px;
    border: 1px solid #e0e0e0;
    font-weight: bold;
}
QScrollBar:vertical {
    background-color: #f5f5f5;
    width: 10px;
    border-radius: 5px;
}
QScrollBar::handle:vertical {
    background-color: #d0d0d0;
    border-radius: 5px;
    min-height: 30px;
}
QScrollBar::handle:vertical:hover {
    background-color: #ff8c00;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
QScrollBar:horizontal {
    background-color: #f5f5f5;
    height: 10px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal {
    background-color: #d0d0d0;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background-color: #ff8c00;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
    width: 0px;
}
QProgressBar {
    background-color: #f0f0f0;
    border: 1px solid #e0e0e0;
    border-radius: 4px;
    text-align: center;
    color: #333333;
    height: 20px;
}
QProgressBar::chunk {
    background-color: #ff6b00;
    border-radius: 3px;
}
QTextEdit, QPlainTextEdit {
    background-color: #ffffff;
    border: 1px solid #e0e0e0;
    color: #333333;
    font-family: "Consolas", "Courier New", monospace;
}
QSlider::groove:horizontal {
    background: #e0e0e0;
    height: 6px;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #ff6b00;
    width: 16px;
    margin: -5px 0;
    border-radius: 8px;
}
QSlider::handle:horizontal:hover {
    background: #e65c00;
}
QSlider::sub-page:horizontal {
    background: #ff8c00;
    border-radius: 3px;
}
QStatusBar {
    background-color: #fff3e6;
    color: #666666;
    border-top: 1px solid #ffcc80;
}
QMenuBar {
    background-color: #ffffff;
    color: #555555;
    border-bottom: 1px solid #e0e0e0;
}
QMenuBar::item:selected {
    background-color: #fff3e6;
    color: #ff6b00;
}
QMenu {
    background-color: #ffffff;
    color: #333333;
    border: 1px solid #e0e0e0;
}
QMenu::item:selected {
    background-color: #fff3e6;
    color: #ff6b00;
}
QToolBar {
    background-color: #ffffff;
    border-bottom: 1px solid #e0e0e0;
    spacing: 5px;
    padding: 3px;
}
QToolButton {
    background-color: transparent;
    color: #555555;
    border: 1px solid transparent;
    border-radius: 3px;
    padding: 5px 10px;
}
QToolButton:hover {
    background-color: #fff3e6;
    border-color: #ffcc80;
}
QCheckBox {
    spacing: 5px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 2px solid #d0d0d0;
    border-radius: 3px;
    background-color: #ffffff;
}
QCheckBox::indicator:checked {
    background-color: #ff6b00;
    border-color: #ff6b00;
}
QRadioButton::indicator {
    width: 16px;
    height: 16px;
    border-radius: 8px;
    border: 2px solid #d0d0d0;
    background-color: #ffffff;
}
QRadioButton::indicator:checked {
    background-color: #ff6b00;
    border-color: #ff6b00;
}
QSplitter::handle {
    background-color: #e0e0e0;
    width: 2px;
    height: 2px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    background-color: #f5f5f5;
    border-radius: 2px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    background-color: #f5f5f5;
    border-radius: 2px;
}
QLabel#titleLabel {
    font-size: 18px;
    font-weight: bold;
    color: #ff6b00;
    padding: 5px;
}
QLabel#subtitleLabel {
    font-size: 14px;
    color: #999999;
}
"""

SIDEBAR_STYLE = """
QWidget#sidebar {
    background-color: #2c2c2c;
}
QWidget#logoWidget {
    background-color: #2c2c2c;
}
QLabel#logoTitle {
    font-size: 20px;
    font-weight: bold;
    color: #ffffff;
    padding: 2px 0;
    background: transparent;
}
QLabel#logoSub {
    font-size: 13px;
    color: #999999;
    padding: 0;
    background: transparent;
}
QFrame#sidebarSep {
    color: #3a3a3a;
    background-color: #3a3a3a;
    max-height: 1px;
    margin: 6px 15px;
}
QFrame#sidebarBorder {
    color: #3a3a3a;
    background-color: #3a3a3a;
    max-width: 1px;
}
QWidget#contentStack {
    background-color: #fafafa;
}
QLabel#versionLabel {
    font-size: 12px;
    color: #666666;
    background: transparent;
    padding: 4px;
}
NavButton {
    background-color: transparent;
    color: #bbbbbb;
    border: none;
    border-radius: 0px;
    text-align: left;
    padding: 9px 18px;
    font-size: 14px;
    font-weight: normal;
    min-width: 0px;
}
NavButton:hover {
    background-color: #383838;
    color: #ffffff;
}
NavButton:checked {
    background-color: #ff6b00;
    color: #ffffff;
    font-weight: bold;
}
"""



class AnnotationWidget(QWidget):
    """标注页面 —— 控制器层，所有数据变更通过 AnnotationManager"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = AnnotationManager()
        self.auto_save_enabled = True
        self._init_ui()

    # ── UI 初始化 ────────────────────────────────────────────

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # ── 左侧面板 ──
        left_panel = QWidget()
        left_panel.setFixedWidth(280)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 数据集目录
        dir_group = QGroupBox("数据集目录")
        dir_layout = QVBoxLayout(dir_group)
        dir_btn_layout = QHBoxLayout()
        self.dir_label = QLineEdit()
        self.dir_label.setReadOnly(True)
        self.dir_label.setPlaceholderText("请选择图片目录...")
        dir_btn = QPushButton("浏览")
        dir_btn.clicked.connect(self._open_dir)
        dir_btn_layout.addWidget(self.dir_label)
        dir_btn_layout.addWidget(dir_btn)
        dir_layout.addLayout(dir_btn_layout)

        self.file_tree = QTreeWidget()
        self.file_tree.setHeaderLabels(["图片列表"])
        self.file_tree.itemClicked.connect(self._on_file_selected)
        dir_layout.addWidget(self.file_tree)

        self.img_count_label = QLabel("图片数量: 0")
        self.img_count_label.setObjectName("subtitleLabel")
        dir_layout.addWidget(self.img_count_label)
        left_layout.addWidget(dir_group)

        # 类别管理
        class_group = QGroupBox("类别管理")
        class_layout = QVBoxLayout(class_group)

        cls_input_layout = QHBoxLayout()
        self.class_input = QLineEdit()
        self.class_input.setPlaceholderText("输入类别名称...")
        add_cls_btn = QPushButton("添加")
        add_cls_btn.clicked.connect(self._add_class)
        cls_input_layout.addWidget(self.class_input)
        cls_input_layout.addWidget(add_cls_btn)
        class_layout.addLayout(cls_input_layout)

        self.class_list = QListWidget()
        self.class_list.currentRowChanged.connect(self._on_class_selected)
        class_layout.addWidget(self.class_list)

        cls_btn_layout = QHBoxLayout()
        del_cls_btn = QPushButton("删除类别")
        del_cls_btn.clicked.connect(self._del_class)
        save_cls_btn = QPushButton("保存类别")
        save_cls_btn.clicked.connect(self._save_classes)
        cls_btn_layout.addWidget(del_cls_btn)
        cls_btn_layout.addWidget(save_cls_btn)
        class_layout.addLayout(cls_btn_layout)
        left_layout.addWidget(class_group)

        # 保存按钮
        save_anno_btn = QPushButton("保存所有标注")
        save_anno_btn.clicked.connect(self._save_annotations)
        left_layout.addWidget(save_anno_btn)
        left_layout.addStretch()

        # ── 中间面板 ──
        center_panel = QWidget()
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(0, 0, 0, 0)

        # 工具栏
        toolbar = QHBoxLayout()

        fit_btn = QPushButton("适应窗口")
        fit_btn.setToolTip("适应窗口 (Ctrl+F)")
        fit_btn.clicked.connect(lambda: self.canvas.zoom_to_fit())

        fitw_btn = QPushButton("适应宽度")
        fitw_btn.setToolTip("适应宽度 (Ctrl+Shift+F)")
        fitw_btn.clicked.connect(lambda: self.canvas.zoom_to_width())

        self.zoom_label = QPushButton("100%")
        self.zoom_label.setFlat(True)
        self.zoom_label.setToolTip("点击重置缩放 (Ctrl+0)")
        self.zoom_label.clicked.connect(lambda: self.canvas.zoom_reset())

        # 长宽比锁定
        ratio_label = QLabel("比例:")
        self.ratio_combo = QComboBox()
        self.ratio_combo.addItems(["自由", "1:1", "4:3", "16:9", "3:4", "9:16", "2:1", "1:2"])
        self.ratio_combo.setToolTip("绘制时锁定长宽比（车牌/人脸/表计等固定比例目标）")
        self.ratio_combo.currentIndexChanged.connect(self._on_ratio_changed)

        self.mode_label = QLabel("📝 标注模式")

        self.auto_save_cb = QCheckBox("自动保存")
        self.auto_save_cb.setChecked(True)
        self.auto_save_cb.stateChanged.connect(self._toggle_auto_save)

        toolbar.addWidget(fit_btn)
        toolbar.addWidget(fitw_btn)
        toolbar.addWidget(self.zoom_label)
        toolbar.addWidget(ratio_label)
        toolbar.addWidget(self.ratio_combo)
        toolbar.addSpacing(10)
        toolbar.addWidget(self.mode_label)
        toolbar.addStretch()
        toolbar.addWidget(self.auto_save_cb)
        center_layout.addLayout(toolbar)

        # 画布
        self.canvas = ImageCanvas()
        self.canvas.bboxDrawn.connect(self._on_bbox_drawn)
        self.canvas.bboxDeleteRequested.connect(self._on_bbox_delete_requested)
        self.canvas.bboxMoveFinished.connect(self._on_bbox_move_finished)
        self.canvas.bboxSelected.connect(self._on_bbox_selected)
        self.canvas.contextMenuRequested.connect(self._show_context_menu)
        self.canvas.copyRequested.connect(self._copy_bboxes)
        self.canvas.pasteRequested.connect(self._paste_bboxes)
        self.canvas.duplicateRequested.connect(self._duplicate_bbox)
        self.canvas.microMoveRequested.connect(self._micro_move)
        self.canvas.zoomChanged.connect(self._on_zoom_changed)
        self.canvas.panModeChanged.connect(self._on_pan_mode_changed)
        self.canvas.copyPrevRequested.connect(self._copy_from_prev)
        self.canvas.discardRequested.connect(self._discard_current)
        center_layout.addWidget(self.canvas)

        # 底部导航
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("◀ 上一张 (A)")
        self.prev_btn.clicked.connect(self._prev_image)
        self.next_btn = QPushButton("下一张 (D) ▶")
        self.next_btn.clicked.connect(self._next_image)
        self.img_info_label = QLabel("0 / 0")
        self.img_info_label.setAlignment(Qt.AlignCenter)

        nav_layout.addWidget(self.prev_btn)
        nav_layout.addWidget(self.img_info_label)
        nav_layout.addWidget(self.next_btn)
        center_layout.addLayout(nav_layout)

        # ── 右侧面板 ──
        right_panel = QWidget()
        right_panel.setFixedWidth(280)
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 当前类别
        cur_cls_group = QGroupBox("当前标注类别")
        cur_cls_layout = QVBoxLayout(cur_cls_group)
        self.cur_cls_combo = QComboBox()
        self.cur_cls_combo.currentIndexChanged.connect(self._on_cur_class_changed)
        cur_cls_layout.addWidget(self.cur_cls_combo)

        self.single_class_label = QLabel()
        self.single_class_label.setObjectName("subtitleLabel")
        self.single_class_label.hide()
        cur_cls_layout.addWidget(self.single_class_label)
        right_layout.addWidget(cur_cls_group)

        # 标注列表
        anno_group = QGroupBox("当前图片标注")
        anno_layout = QVBoxLayout(anno_group)

        # 标签过滤
        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("显示:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("全部", -1)
        self.filter_combo.currentIndexChanged.connect(self._on_filter_changed)
        filter_layout.addWidget(self.filter_combo)
        anno_layout.addLayout(filter_layout)

        self.stats_label = QLabel("统计: 无")
        self.stats_label.setObjectName("subtitleLabel")
        anno_layout.addWidget(self.stats_label)

        self.save_indicator = QLabel("🟢 已保存")
        anno_layout.addWidget(self.save_indicator)

        self.anno_table = QTableWidget()
        self.anno_table.setColumnCount(3)
        self.anno_table.setHorizontalHeaderLabels(["类别", "宽", "高"])
        self.anno_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.anno_table.itemClicked.connect(self._on_anno_selected)
        anno_layout.addWidget(self.anno_table)

        del_anno_btn = QPushButton("删除选中标注")
        del_anno_btn.clicked.connect(self._del_annotation)
        anno_layout.addWidget(del_anno_btn)
        right_layout.addWidget(anno_group)

        # 操作提示
        tip_group = QGroupBox("操作提示")
        tip_layout = QVBoxLayout(tip_group)
        tips = [
            "左键拖拽: 绘制  四角拖拽: 缩放框",
            "左键点击: 选中/拖拽  右键/DblClick: 编辑类别",
            "M: 拖拽模式  N: 标注模式  X: 废弃图片",
            "A/D: 切换图片  Delete: 删除框",
            "Ctrl+F/0: 适应窗口/重置缩放",
            "方向键: 微移  Ctrl+C/V/D: 复制/粘贴",
            "Ctrl+Shift+C: 复制上一帧框",
        ]
        for tip in tips:
            tip_layout.addWidget(QLabel(tip))
        right_layout.addWidget(tip_group)
        right_layout.addStretch()

        # ── 组合 ──
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(center_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)

        layout.addWidget(splitter)

        # ── 快捷键 ──
        self._prev_sc = QShortcut(Qt.Key_A, self)
        self._prev_sc.activated.connect(self._prev_image)
        self._prev_sc2 = QShortcut(Qt.Key_Left, self)
        self._prev_sc2.activated.connect(self._prev_image)
        self._next_sc = QShortcut(Qt.Key_D, self)
        self._next_sc.activated.connect(self._next_image)
        self._next_sc2 = QShortcut(Qt.Key_Right, self)
        self._next_sc2.activated.connect(self._next_image)

        self._pan_m_sc = QShortcut(Qt.Key_M, self)
        self._pan_m_sc.activated.connect(self._enter_pan_mode)
        self._pan_n_sc = QShortcut(Qt.Key_N, self)
        self._pan_n_sc.activated.connect(self._enter_anno_mode)

        self._copy_prev_sc = QShortcut(Qt.Key_C | Qt.SHIFT | Qt.CTRL, self)
        self._copy_prev_sc.activated.connect(self._copy_from_prev)

    # ── 数据同步 ─────────────────────────────────────────────

    def _sync_canvas(self):
        """将 Manager 数据推送到 Canvas"""
        annos = self.manager.get_current_annotations()
        self.canvas.set_annotations(annos)
        self.canvas.set_class_names(self.manager.classes)
        self.canvas.set_colors(self.manager.class_colors)
        self._update_save_indicator()

    def _update_save_indicator(self):
        dirty = self.manager.is_dirty()
        self.save_indicator.setText("🟡 未保存" if dirty else "🟢 已保存")

    # ── 文件操作 ─────────────────────────────────────────────

    def _open_dir(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if not dir_path:
            return
        self.dir_label.setText(dir_path)
        count = self.manager.load_image_dir(dir_path)
        self._refresh_file_list()
        self.img_count_label.setText(f"图片数量: {count}")
        self._update_nav()
        self._refresh_class_list()

    def _refresh_file_list(self):
        self.file_tree.clear()
        for path in self.manager.image_list:
            item = QTreeWidgetItem([Path(path).name])
            item.setData(0, Qt.UserRole, path)
            self.file_tree.addTopLevelItem(item)

    def _on_file_selected(self, item, col):
        try:
            path = item.data(0, Qt.UserRole)
            idx = self.manager.image_list.index(path)
        except (ValueError, AttributeError):
            return
        self._navigate_to(idx)

    # ── 导航 ─────────────────────────────────────────────────

    def _prev_image(self):
        idx = self.manager.current_index - 1
        if idx >= 0:
            self._navigate_to(idx)

    def _next_image(self):
        idx = self.manager.current_index + 1
        if idx < len(self.manager.image_list):
            self._navigate_to(idx)

    def _navigate_to(self, idx):
        """导航到指定图片，处理自动保存"""
        if self.auto_save_enabled and self.manager.is_dirty():
            self._auto_save_current()
        elif not self.auto_save_enabled and self.manager.is_dirty():
            reply = QMessageBox.question(
                self, "未保存的修改",
                "当前图片有未保存的标注修改。\n是否保存后再切换？",
                QMessageBox.Save | QMessageBox.Discard | QMessageBox.Cancel
            )
            if reply == QMessageBox.Save:
                self._auto_save_current()
            elif reply == QMessageBox.Cancel:
                return
        self._load_image_at(idx)

    def _load_image_at(self, idx):
        try:
            img = self.manager.load_image(idx)
            if img is not None:
                self.canvas.load_image(img, self.manager.get_current_annotations())
                self._update_nav()
                self._refresh_anno_table()
        except Exception:
            pass

    def _update_nav(self):
        total = self.manager.get_image_count()
        cur = self.manager.current_index + 1 if self.manager.current_index >= 0 else 0
        self.img_info_label.setText(f"{cur} / {total}")
        self.prev_btn.setEnabled(cur > 1)
        self.next_btn.setEnabled(cur < total)

    # ── 自动保存 ─────────────────────────────────────────────

    def _auto_save_current(self):
        self.manager.save_annotations()
        self._update_save_indicator()

    def _toggle_auto_save(self, state):
        self.auto_save_enabled = (state == Qt.Checked)

    # ── 类别管理 ─────────────────────────────────────────────

    def _add_class(self):
        name = self.class_input.text().strip()
        if not name:
            return
        if name in self.manager.classes:
            return
        self.manager.classes.append(name)
        cls_id = len(self.manager.classes) - 1
        self.manager.class_colors[cls_id] = self.manager._gen_color(cls_id)
        self._refresh_class_list()
        self.class_input.clear()

    def _del_class(self):
        row = self.class_list.currentRow()
        if 0 <= row < len(self.manager.classes):
            self.manager.delete_class(row)
            self._refresh_class_list()
            self._sync_canvas()
            self._refresh_anno_table()

    def _save_classes(self):
        self.manager.save_classes(self.manager.classes)

    def _on_class_selected(self, row):
        if row >= 0:
            self.canvas.set_current_class(row)
            self.cur_cls_combo.setCurrentIndex(row)

    def _on_cur_class_changed(self, idx):
        if idx >= 0:
            self.canvas.set_current_class(idx)

    def _refresh_class_list(self):
        self.class_list.clear()
        self.cur_cls_combo.clear()
        for cls_name in self.manager.classes:
            self.class_list.addItem(cls_name)
            self.cur_cls_combo.addItem(cls_name)
        self.canvas.set_class_names(self.manager.classes)

        # 更新过滤下拉框
        cur_filter = self.filter_combo.currentData()
        self.filter_combo.blockSignals(True)
        self.filter_combo.clear()
        self.filter_combo.addItem("全部", -1)
        for i, cls_name in enumerate(self.manager.classes):
            self.filter_combo.addItem(cls_name, i)
        # 恢复之前的过滤选择
        idx = self.filter_combo.findData(cur_filter)
        self.filter_combo.setCurrentIndex(max(0, idx))
        self.filter_combo.blockSignals(False)

        # 单类别模式
        if len(self.manager.classes) == 1:
            self.cur_cls_combo.hide()
            self.single_class_label.setText(
                f"单类别模式: {self.manager.classes[0]} (自动分配)"
            )
            self.single_class_label.show()
            self.canvas.set_current_class(0)
        else:
            self.cur_cls_combo.show()
            self.single_class_label.hide()

    # ── 标注操作（信号处理） ──────────────────────────────────

    def _on_bbox_drawn(self, x1, y1, x2, y2):
        self.manager.add_bbox(self.canvas.current_class_id, x1, y1, x2, y2)
        self._sync_canvas()
        self._refresh_anno_table()

    def _on_bbox_delete_requested(self, idx):
        self.manager.remove_bbox(idx)
        self._sync_canvas()
        self.canvas.selected_bbox_idx = -1
        self._refresh_anno_table()

    def _on_bbox_move_finished(self, idx, x1, y1, x2, y2):
        class_id = self.manager.get_bbox_class(idx)
        self.manager.update_bbox(idx, class_id, x1, y1, x2, y2)
        self._sync_canvas()
        self._refresh_anno_table()

    def _on_bbox_selected(self, idx):
        if idx >= 0:
            self.anno_table.selectRow(idx)
        else:
            self.anno_table.clearSelection()

    def _copy_bboxes(self):
        self.manager.copy_annotations()

    def _paste_bboxes(self):
        count = self.manager.paste_annotations()
        if count > 0:
            self._sync_canvas()
            self._refresh_anno_table()

    def _duplicate_bbox(self):
        if self.canvas.selected_bbox_idx >= 0:
            self.manager.duplicate_bbox(self.canvas.selected_bbox_idx)
            self._sync_canvas()
            self._refresh_anno_table()

    def _micro_move(self, dx, dy):
        if self.canvas.selected_bbox_idx >= 0:
            self.manager.micro_move(self.canvas.selected_bbox_idx, dx, dy)
            self._sync_canvas()

    def _del_annotation(self):
        row = self.anno_table.currentRow()
        if row >= 0:
            self.manager.remove_bbox(row)
            self._sync_canvas()
            self.canvas.selected_bbox_idx = -1
            self._refresh_anno_table()

    # ── 右键菜单 ─────────────────────────────────────────────

    def _show_context_menu(self, bbox_idx, global_pos):
        menu = QMenu(self)
        edit_act = menu.addAction("编辑类别")
        dup_act = menu.addAction("复制标注框")
        menu.addSeparator()
        del_act = menu.addAction("删除标注框")
        action = menu.exec_(global_pos)

        if action == edit_act:
            self._edit_bbox_label(bbox_idx)
        elif action == dup_act:
            self.canvas.selected_bbox_idx = bbox_idx
            self._duplicate_bbox()
        elif action == del_act:
            self.canvas.selected_bbox_idx = bbox_idx
            self._on_bbox_delete_requested(bbox_idx)

    def _edit_bbox_label(self, bbox_idx):
        """编辑标注框的类别"""
        annos = self.manager.get_current_annotations()
        if not (0 <= bbox_idx < len(annos)):
            return
        old_cls_id = annos[bbox_idx].get('class_id', 0)
        menu = QMenu(self)
        for cls_id, cls_name in enumerate(self.manager.classes):
            act = menu.addAction(cls_name)
            act.setData(cls_id)
            if cls_id == old_cls_id:
                act.setCheckable(True)
                act.setChecked(True)
        action = menu.exec_(QCursor.pos())
        if action is not None and action.data() != old_cls_id:
            h, w = self.manager.current_image.shape[:2]
            ann = annos[bbox_idx]
            x1, y1, x2, y2 = self.manager._yolo_to_pixel(ann, w, h)
            self.manager.update_bbox(bbox_idx, action.data(), x1, y1, x2, y2)
            self._sync_canvas()
            self._refresh_anno_table()

    # ── 标注表格 ─────────────────────────────────────────────

    def _refresh_anno_table(self):
        annos = self.manager.get_current_annotations()
        self.anno_table.setRowCount(len(annos))
        for i, ann in enumerate(annos):
            cls_id = ann.get('class_id', 0)
            cls_name = (
                self.manager.classes[cls_id]
                if cls_id < len(self.manager.classes) else str(cls_id)
            )
            h, w = 0, 0
            if self.manager.current_image is not None:
                h, w = self.manager.current_image.shape[:2]
            bw = int(ann.get('w', 0) * w)
            bh = int(ann.get('h', 0) * h)
            self.anno_table.setItem(i, 0, QTableWidgetItem(cls_name))
            self.anno_table.setItem(i, 1, QTableWidgetItem(str(bw)))
            self.anno_table.setItem(i, 2, QTableWidgetItem(str(bh)))
        self._update_stats()

    def _update_stats(self):
        stats = self.manager.get_stats()
        if not stats:
            self.stats_label.setText("统计: 无")
            return
        parts = []
        for cls_id, count in sorted(stats.items()):
            name = (
                self.manager.classes[cls_id]
                if cls_id < len(self.manager.classes) else f"cls_{cls_id}"
            )
            parts.append(f"{name}×{count}")
        self.stats_label.setText("统计: " + "  ".join(parts))

    def _on_anno_selected(self, item):
        row = item.row()
        self.canvas.selected_bbox_idx = row
        self.canvas._invalidate()

    # ── 保存 ─────────────────────────────────────────────────

    def _save_annotations(self):
        self.manager.save_annotations()
        self._update_save_indicator()
        QMessageBox.information(
            self, "提示",
            f"标注已保存到:\n{self.manager.dataset_dir}/labels/"
        )

    # ── 模式/缩放回调 ────────────────────────────────────────

    def _on_zoom_changed(self, zoom):
        pct = int(zoom * 100)
        self.zoom_label.setText(f"{pct}%")

    def _enter_pan_mode(self):
        self.canvas.set_pan_mode(True)
        self.mode_label.setText("🖐 拖拽模式")

    def _enter_anno_mode(self):
        self.canvas.set_pan_mode(False)
        self.mode_label.setText("📝 标注模式")

    def _on_pan_mode_changed(self, enabled):
        self.mode_label.setText("🖐 拖拽模式" if enabled else "📝 标注模式")

    def _on_ratio_changed(self, idx):
        """长宽比锁定"""
        ratios = [None, 1.0, 4/3, 16/9, 3/4, 9/16, 2.0, 0.5]
        if idx < len(ratios):
            self.canvas.set_aspect_ratio(ratios[idx])

    def _on_filter_changed(self, idx):
        """标签过滤"""
        class_id = self.filter_combo.currentData()
        self.canvas.set_filter_class(class_id)

    def _copy_from_prev(self):
        """复制上一帧标注"""
        if self.manager.copy_from_prev_image():
            self._sync_canvas()
            self.canvas._invalidate()
            self._refresh_anno_table()

    def _discard_current(self):
        """标记当前图片为废弃"""
        self.manager.discard_current_image()
        QMessageBox.information(
            self, "已标记",
            f"当前图片已加入废弃列表: discarded.txt"
        )


# ── 数据集验证对话框 ────────────────────────────────────────

_TYPE_LABELS = {
    'class_id_oob': 'class_id 越界',
    'coord_oob': '坐标超出范围',
    'zero_size': '宽/高为零',
    'bad_format': '格式错误',
    'bad_number': '数值非法',
    'missing_label': '缺少标注文件',
    'empty_label': '空标注文件',
}

class ValidationDialog(QDialog):
    def __init__(self, result, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据集标注验证")
        self.resize(850, 500)
        self.setMinimumSize(600, 350)

        layout = QVBoxLayout(self)

        # 摘要
        summary = QLabel(
            f"发现 <b>{len(result.errors)}</b> 个问题 "
            f"（{', '.join(f'{_TYPE_LABELS.get(k, k)}×{v}' for k, v in result.error_summary.items())}）"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        # 表格
        table = QTableWidget()
        table.setColumnCount(4)
        table.setHorizontalHeaderLabels(["文件", "类型", "详情", "位置"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeToContents)
        table.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeToContents)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setEditTriggers(QTableWidget.NoEditTriggers)

        table.setRowCount(len(result.errors))
        for i, e in enumerate(result.errors):
            fname = Path(e.file_path).name if e.file_path else '—'
            table.setItem(i, 0, QTableWidgetItem(fname))
            table.setItem(i, 1, QTableWidgetItem(_TYPE_LABELS.get(e.issue_type, e.issue_type)))
            table.setItem(i, 2, QTableWidgetItem(e.detail))
            loc = f"行 {e.bbox_index + 1}" if e.bbox_index >= 0 else "—"
            table.setItem(i, 3, QTableWidgetItem(loc))

        layout.addWidget(table)

        # 按钮
        btn_layout = QHBoxLayout()
        fix_btn = QPushButton("自动修复 (删除有问题的标注行)")
        fix_btn.setStyleSheet("color: #e67e22;")
        fix_btn.clicked.connect(lambda: self.done(2))

        skip_btn = QPushButton("跳过 (继续训练)")
        skip_btn.setStyleSheet("color: #666;")
        skip_btn.clicked.connect(lambda: self.done(1))

        cancel_btn = QPushButton("取消训练")
        cancel_btn.clicked.connect(self.reject)

        btn_layout.addWidget(fix_btn)
        btn_layout.addStretch()
        btn_layout.addWidget(skip_btn)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)


class TrainingWidget(QWidget):
    """模型训练页面 - 含环境检测、镜像源配置、训练模式选择"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = TrainingManager()
        self.manager.set_callbacks(
            progress_callback=self._on_progress,
            log_callback=self._on_log,
            metrics_callback=self._on_metrics
        )
        self.mirrors = get_pip_mirrors()
        self.status_callback = None  # 由MainWindow设置
        self._init_ui()

        # ── 恢复上次设置 ──
        last_yaml = recall_last_data_yaml()
        if last_yaml and os.path.exists(last_yaml):
            self.data_yaml_edit.setText(last_yaml)
            self._validate_current_yaml(last_yaml)
        last_model = recall_last_model()
        idx = self.model_combo.findText(last_model)
        if idx >= 0:
            self.model_combo.setCurrentIndex(idx)

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        # ================= 顶部：环境配置区 =================
        env_group = QGroupBox("环境配置")
        env_layout = QHBoxLayout(env_group)
        env_layout.setSpacing(15)

        # ----- 训练模式 -----
        mode_group = QGroupBox("训练模式")
        mode_layout = QVBoxLayout(mode_group)
        self.cpu_radio = QRadioButton("CPU 训练")
        self.gpu_radio = QRadioButton("GPU 训练")
        self.cpu_radio.setChecked(True)
        mode_layout.addWidget(self.cpu_radio)
        mode_layout.addWidget(self.gpu_radio)
        mode_layout.addStretch()
        env_layout.addWidget(mode_group)

        # ----- 镜像源配置 -----
        mirror_group = QGroupBox("镜像源配置")
        mirror_layout = QVBoxLayout(mirror_group)
        mirror_row = QHBoxLayout()
        mirror_row.addWidget(QLabel("PyPI 镜像:"))
        self.mirror_combo = QComboBox()
        self.mirror_combo.addItems(list(self.mirrors.keys()))
        self.mirror_combo.setCurrentText('默认 (PyPI官方)')
        mirror_row.addWidget(self.mirror_combo)
        mirror_layout.addLayout(mirror_row)

        self.mirror_url_label = QLabel(self.mirrors['默认 (PyPI官方)'])
        self.mirror_url_label.setObjectName("subtitleLabel")
        self.mirror_url_label.setWordWrap(True)
        mirror_layout.addWidget(self.mirror_url_label)
        self.mirror_combo.currentTextChanged.connect(
            lambda t: self.mirror_url_label.setText(self.mirrors.get(t, ''))
        )
        mirror_layout.addStretch()
        env_layout.addWidget(mirror_group)

        # ----- CUDA环境 -----
        cuda_group = QGroupBox("CUDA 环境配置")
        cuda_layout = QVBoxLayout(cuda_group)
        cuda_layout.addWidget(QLabel("CUDA 版本:"))
        self.cuda_combo = QComboBox()
        self.cuda_combo.addItems(['cpu', 'cu118', 'cu121'])
        self.cuda_combo.setCurrentText('cpu')
        cuda_layout.addWidget(self.cuda_combo)
        cuda_layout.addStretch()
        env_layout.addWidget(cuda_group)

        # ----- 环境操作按钮 -----
        ops_group = QGroupBox("环境操作")
        ops_layout = QVBoxLayout(ops_group)
        self.detect_env_btn = QPushButton("🔍 检测环境")
        self.detect_env_btn.clicked.connect(self._detect_environment)
        ops_layout.addWidget(self.detect_env_btn)

        self.install_pt_btn = QPushButton("📦 安装 PyTorch")
        self.install_pt_btn.clicked.connect(self._install_pytorch)
        ops_layout.addWidget(self.install_pt_btn)

        self.install_ultra_btn = QPushButton("📦 安装 Ultralytics")
        self.install_ultra_btn.clicked.connect(self._install_ultralytics)
        ops_layout.addWidget(self.install_ultra_btn)

        self.fix_pt_btn = QPushButton("🔧 修复 PyTorch")
        self.fix_pt_btn.setToolTip("检测到DLL错误时，强制重装PyTorch")
        self.fix_pt_btn.clicked.connect(self._fix_pytorch)
        ops_layout.addWidget(self.fix_pt_btn)
        ops_layout.addStretch()
        env_layout.addWidget(ops_group)

        outer.addWidget(env_group)

        # ================= 子标签页：训练配置 | 训练监控 =================
        self.sub_tabs = QTabWidget()
        self.sub_tabs.setStyleSheet("""
            QTabWidget::pane { border: 1px solid #e0e0e0; background: #fafafa; }
            QTabBar::tab { padding: 10px 20px; font-size: 15px; }
            QTabBar::tab:selected { color: #ff6b00; border-bottom: 3px solid #ff6b00; font-weight: bold; }
        """)

        # ---- Tab 0: 训练配置 ----
        config_tab = QWidget()
        config_outer = QVBoxLayout(config_tab)
        config_outer.setContentsMargins(0, 0, 0, 0)

        mid_splitter = QSplitter(Qt.Horizontal)

        # 左侧：环境信息面板
        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.setContentsMargins(0, 0, 0, 0)

        sys_group = QGroupBox("系统环境信息")
        sys_layout = QGridLayout(sys_group)

        self.env_labels = {}
        env_fields = [
            ('python_version', 'Python 版本'),
            ('python_arch', 'Python 架构'),
            ('pytorch_version', 'PyTorch 版本'),
            ('pytorch_error', 'PyTorch 状态'),
            ('cuda_version', 'CUDA 版本'),
            ('cuda_available', 'CUDA 可用'),
            ('gpu_count', 'GPU 数量'),
            ('gpu_names', 'GPU 型号'),
            ('nvidia_driver', 'NVIDIA 驱动'),
            ('nvidia_smi_cuda', 'Driver CUDA'),
            ('vc_redist', 'VC++ Redist'),
            ('ultralytics_version', 'Ultralytics'),
            ('opencv_version', 'OpenCV'),
            ('platform', '系统平台'),
        ]
        for i, (key, label) in enumerate(env_fields):
            lbl = QLabel(f"{label}:")
            lbl.setStyleSheet("font-weight: bold; color: #555;")
            val = QLabel("---")
            val.setObjectName("subtitleLabel")
            val.setWordWrap(True)
            self.env_labels[key] = val
            sys_layout.addWidget(lbl, i, 0)
            sys_layout.addWidget(val, i, 1)

        info_layout.addWidget(sys_group)
        info_layout.addStretch()
        mid_splitter.addWidget(info_panel)

        # 右侧：训练配置
        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(0, 0, 0, 0)

        ds_model_group = QGroupBox("数据集 & 模型")
        ds_model_layout = QGridLayout(ds_model_group)

        ds_model_layout.addWidget(QLabel("数据集:"), 0, 0)
        ds_row = QHBoxLayout()
        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setReadOnly(True)
        self.data_yaml_edit.setPlaceholderText("选择或生成 data.yaml...")
        ds_browse = QPushButton("浏览")
        ds_browse.clicked.connect(self._browse_data)
        ds_gen = QPushButton("自动生成")
        ds_gen.setToolTip("从数据集目录自动生成 data.yaml")
        ds_gen.clicked.connect(self._generate_yaml)
        ds_row.addWidget(self.data_yaml_edit)
        ds_row.addWidget(ds_browse)
        ds_row.addWidget(ds_gen)
        ds_model_layout.addLayout(ds_row, 0, 1)

        self.data_info_label = QLabel("未加载数据集")
        self.data_info_label.setObjectName("subtitleLabel")
        self.data_info_label.setWordWrap(True)
        ds_model_layout.addWidget(self.data_info_label, 1, 1)

        ds_model_layout.addWidget(QLabel("模型:"), 2, 0)
        model_row = QHBoxLayout()
        self.model_combo = QComboBox()
        self.model_combo.addItems(get_supported_models())
        self.model_combo.setCurrentText(recall_last_model())
        self.model_combo.currentTextChanged.connect(remember_last_model)
        model_row.addWidget(self.model_combo)
        self.pretrained_check = QCheckBox("预训练")
        self.pretrained_check.setChecked(True)
        model_row.addWidget(self.pretrained_check)
        ds_model_layout.addLayout(model_row, 2, 1)

        config_layout.addWidget(ds_model_group)

        params_group = QGroupBox("训练参数")
        params_layout = QGridLayout(params_group)

        defaults = get_default_training_params()

        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 1000)
        self.epochs_spin.setValue(defaults['epochs'])
        params_layout.addWidget(QLabel("训练轮数:"), 0, 0)
        params_layout.addWidget(self.epochs_spin, 0, 1)

        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 256)
        self.batch_spin.setValue(defaults['batch_size'])
        params_layout.addWidget(QLabel("批次大小:"), 0, 2)
        params_layout.addWidget(self.batch_spin, 0, 3)

        # img size: combo + custom
        img_label = QLabel("图片尺寸:")
        self.imgsz_combo = QComboBox()
        self.imgsz_combo.addItems(['640', '960', '1280', '自定义'])
        self.imgsz_combo.setCurrentText(str(defaults['img_size']))
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 2048)
        self.imgsz_spin.setValue(defaults['img_size'])
        self.imgsz_spin.setVisible(False)
        self.imgsz_combo.currentTextChanged.connect(
            lambda t: self.imgsz_spin.setVisible(t == '自定义')
        )
        img_row = QHBoxLayout()
        img_row.addWidget(self.imgsz_combo)
        img_row.addWidget(self.imgsz_spin)
        params_layout.addWidget(img_label, 1, 0)
        params_layout.addLayout(img_row, 1, 1)

        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.0001, 1.0)
        self.lr_spin.setValue(defaults['learning_rate'])
        self.lr_spin.setSingleStep(0.001)
        self.lr_spin.setDecimals(4)
        params_layout.addWidget(QLabel("学习率:"), 1, 2)
        params_layout.addWidget(self.lr_spin, 1, 3)

        self.warmup_spin = QSpinBox()
        self.warmup_spin.setRange(0, 50)
        self.warmup_spin.setValue(defaults['warmup_epochs'])
        params_layout.addWidget(QLabel("预热轮数:"), 2, 0)
        params_layout.addWidget(self.warmup_spin, 2, 1)

        self.device_combo = QComboBox()
        self.device_combo.addItems(['自动选择', 'GPU 0', 'GPU 1', 'CPU'])
        params_layout.addWidget(QLabel("训练设备:"), 2, 2)
        params_layout.addWidget(self.device_combo, 2, 3)

        # 优化器
        params_layout.addWidget(QLabel("优化器:"), 3, 0)
        self.optimizer_combo = QComboBox()
        self.optimizer_combo.addItems(['auto', 'SGD', 'Adam', 'AdamW'])
        self.optimizer_combo.setCurrentText(defaults.get('optimizer', 'auto'))
        params_layout.addWidget(self.optimizer_combo, 3, 1)

        self.workers_spin = QSpinBox()
        self.workers_spin.setRange(0, 32)
        self.workers_spin.setValue(defaults['workers'])
        params_layout.addWidget(QLabel("工作线程:"), 3, 2)
        params_layout.addWidget(self.workers_spin, 3, 3)

        # close_mosaic + seed
        params_layout.addWidget(QLabel("关闭Mosaic:"), 4, 0)
        self.mosaic_spin = QSpinBox()
        self.mosaic_spin.setRange(0, 100)
        self.mosaic_spin.setValue(defaults.get('close_mosaic', 10))
        self.mosaic_spin.setToolTip("最后 N 轮关闭 Mosaic 增强（0=不关闭）")
        params_layout.addWidget(self.mosaic_spin, 4, 1)

        params_layout.addWidget(QLabel("随机种子:"), 4, 2)
        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 99999)
        self.seed_spin.setValue(0)
        self.seed_spin.setToolTip("固定随机种子（0=随机），确保可复现结果")
        self.seed_spin.setSpecialValueText("随机")
        params_layout.addWidget(self.seed_spin, 4, 3)

        # rect + augment + cos_lr
        self.augment_check = QCheckBox("数据增强")
        self.augment_check.setChecked(True)
        params_layout.addWidget(self.augment_check, 5, 0)

        self.cos_lr_check = QCheckBox("余弦学习率衰减")
        params_layout.addWidget(self.cos_lr_check, 5, 1)

        self.rect_check = QCheckBox("矩形训练")
        self.rect_check.setToolTip("矩形推理训练（减少填充，加速训练）")
        params_layout.addWidget(self.rect_check, 5, 2)

        config_layout.addWidget(params_group)

        # 高级设置（可折叠/展开）
        self.advanced_group = QGroupBox("高级设置（自定义 args）")
        self.advanced_group.setCheckable(True)
        self.advanced_group.setChecked(False)
        adv_layout = QVBoxLayout(self.advanced_group)
        self.advanced_args_edit = QTextEdit()
        self.advanced_args_edit.setMaximumHeight(80)
        self.advanced_args_edit.setPlaceholderText(
            "自定义 YOLO 参数，每行一个 key=value:\n"
            "例:  lr_factor=0.01  label_smoothing=0.1\n"
            "留空表示不传入额外参数"
        )
        adv_layout.addWidget(self.advanced_args_edit)
        config_layout.addWidget(self.advanced_group)

        btn_row = QHBoxLayout()
        self.train_btn = QPushButton("▶ 开始训练")
        self.train_btn.setObjectName("btnTrain")
        self.train_btn.clicked.connect(self._start_training)
        self.stop_btn = QPushButton("■ 停止训练")
        self.stop_btn.setObjectName("btnStop")
        self.stop_btn.clicked.connect(self._stop_training)
        self.stop_btn.setEnabled(False)
        btn_row.addWidget(self.train_btn)
        btn_row.addWidget(self.stop_btn)
        config_layout.addLayout(btn_row)

        config_layout.addStretch()
        mid_splitter.addWidget(config_panel)

        mid_splitter.setStretchFactor(0, 3)
        mid_splitter.setStretchFactor(1, 4)
        config_outer.addWidget(mid_splitter)

        self.sub_tabs.addTab(config_tab, "⚙ 训练配置")

        # ---- Tab 1: 训练监控 ----
        monitor_tab = QWidget()
        monitor_layout = QVBoxLayout(monitor_tab)
        monitor_layout.setContentsMargins(5, 5, 5, 5)

        # 顶部：训练数据集 + 模型信息
        info_row = QHBoxLayout()
        self.monitor_ds_label = QLabel("数据集: ---")
        self.monitor_ds_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        self.monitor_model_label = QLabel("模型: ---")
        self.monitor_model_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #333;")
        info_row.addWidget(self.monitor_ds_label)
        info_row.addWidget(self.monitor_model_label)
        info_row.addStretch()
        monitor_layout.addLayout(info_row)

        # 中部：训练曲线 + 指标表格
        mid_row = QHBoxLayout()

        # 训练曲线 (matplotlib)
        curve_group = QGroupBox("训练曲线")
        curve_layout = QVBoxLayout(curve_group)
        self.training_figure = Figure(figsize=(6, 4), dpi=80, facecolor='#fafafa')
        self.loss_ax = self.training_figure.add_subplot(121)
        self.loss_ax.set_title('Loss', fontsize=10, color='#333')
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True, alpha=0.3)
        self.loss_ax.set_facecolor('#ffffff')
        self.metric_ax = self.training_figure.add_subplot(122)
        self.metric_ax.set_title('mAP', fontsize=10, color='#333')
        self.metric_ax.set_xlabel('Epoch')
        self.metric_ax.set_ylabel('mAP')
        self.metric_ax.grid(True, alpha=0.3)
        self.metric_ax.set_facecolor('#ffffff')
        self.training_figure.tight_layout(pad=2)
        self.curve_canvas = FigureCanvas(self.training_figure)
        curve_layout.addWidget(self.curve_canvas)
        mid_row.addWidget(curve_group, 3)

        # 指标表格 (参数为行，值+最佳为列)
        table_group = QGroupBox("训练指标")
        table_layout = QVBoxLayout(table_group)

        info_row = QHBoxLayout()
        self.metrics_epoch_label = QLabel("Epoch: ---")
        self.metrics_epoch_label.setStyleSheet("font-size: 14px; font-weight: bold; color: #ff6b00;")
        info_row.addWidget(self.metrics_epoch_label)
        info_row.addStretch()
        table_layout.addLayout(info_row)

        self.metrics_table = QTableWidget()
        self.metrics_table.setRowCount(5)
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(['当前值', '最佳值'])
        self.metrics_table.setVerticalHeaderLabels(['box_loss', 'cls_loss', 'dfl_loss', 'mAP50', 'mAP50-95'])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.verticalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.metrics_table.setAlternatingRowColors(True)
        self.metrics_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.metrics_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.metrics_table.setMaximumHeight(200)

        # 初始化空值
        for r in range(5):
            for c in range(2):
                item = QTableWidgetItem('---')
                item.setTextAlignment(Qt.AlignCenter)
                self.metrics_table.setItem(r, c, item)

        table_layout.addWidget(self.metrics_table)
        mid_row.addWidget(table_group, 2)

        monitor_layout.addLayout(mid_row)

        # 底部：进度 + 日志
        bottom_row = QHBoxLayout()

        progress_group = QGroupBox("训练进度")
        progress_layout = QVBoxLayout(progress_group)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(True)
        progress_layout.addWidget(self.progress_bar)

        prog_info = QHBoxLayout()
        self.progress_label = QLabel("就绪")
        self.eta_label = QLabel("")
        prog_info.addWidget(self.progress_label)
        prog_info.addStretch()
        prog_info.addWidget(self.eta_label)
        progress_layout.addLayout(prog_info)
        bottom_row.addWidget(progress_group)
        monitor_layout.addLayout(bottom_row)

        self.sub_tabs.addTab(monitor_tab, "📈 训练监控")

        # ---- Tab 2: 训练日志 ----
        log_tab = QWidget()
        log_tab.setStyleSheet("background: #1e1e1e;")
        log_layout_full = QVBoxLayout(log_tab)
        log_layout_full.setContentsMargins(0, 0, 0, 0)
        log_layout_full.setSpacing(0)

        # 日志内容区 — 终端风格（先创建，工具栏按钮引用它）
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet("""
            QTextEdit {
                background: #1e1e1e;
                color: #d4d4d4;
                font-family: 'Consolas', 'Courier New', 'Microsoft YaHei', monospace;
                font-size: 15px;
                border: none;
                padding: 8px 12px;
                selection-background-color: #264f78;
            }
        """)

        # 工具栏
        toolbar = QWidget()
        toolbar.setStyleSheet("background: #2d2d2d; border-bottom: 1px solid #3d3d3d;")
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(12, 6, 12, 6)

        title = QLabel("📋 训练日志")
        title.setStyleSheet("color: #ddd; font-size: 15px; font-weight: bold; border: none; background: transparent;")
        toolbar_layout.addWidget(title)
        self.log_line_count = QLabel("0 行")
        self.log_line_count.setStyleSheet("color: #888; font-size: 13px; border: none; background: transparent;")
        toolbar_layout.addWidget(self.log_line_count)
        toolbar_layout.addStretch()

        clear_btn = QPushButton("清空")
        clear_btn.setStyleSheet("background: #3d3d3d; color: #ccc; border: none; border-radius: 3px; "
                                "padding: 4px 12px; font-size: 13px;")
        clear_btn.clicked.connect(self.log_text.clear)
        toolbar_layout.addWidget(clear_btn)

        save_log_btn = QPushButton("导出")
        save_log_btn.setStyleSheet("background: #3d3d3d; color: #ccc; border: none; border-radius: 3px; "
                                   "padding: 4px 12px; font-size: 13px;")
        save_log_btn.clicked.connect(self._export_log)
        toolbar_layout.addWidget(save_log_btn)

        auto_scroll_check = QCheckBox("自动滚动")
        auto_scroll_check.setChecked(True)
        auto_scroll_check.setStyleSheet("color: #aaa; font-size: 13px; border: none; background: transparent;")
        auto_scroll_check.stateChanged.connect(lambda s: setattr(self, '_auto_scroll', s == Qt.Checked))
        self._auto_scroll = True
        toolbar_layout.addWidget(auto_scroll_check)

        log_layout_full.addWidget(toolbar)
        log_layout_full.addWidget(self.log_text)

        # 连接行数统计和自动滚动
        self.log_text.textChanged.connect(
            lambda: self.log_line_count.setText(f"{self.log_text.document().blockCount()} 行"))
        self.log_text.textChanged.connect(self._scroll_log_if_needed)

        self.sub_tabs.addTab(log_tab, "📋 训练日志")

        outer.addWidget(self.sub_tabs)

    # ========== 环境操作 ==========

    def _detect_environment(self):
        """检测环境并更新显示"""
        self._on_log("=== 开始环境检测 ===")
        self.detect_env_btn.setEnabled(False)
        QApplication.processEvents()

        info = self.manager.detect_env()

        def set_val(key, value):
            if key in self.env_labels:
                self.env_labels[key].setText(str(value))

        set_val('python_version', info.get('python_version', '---'))
        set_val('python_arch', info.get('python_arch', '---'))
        set_val('pytorch_version', info.get('pytorch_version', '---'))

        # PyTorch 错误/状态显示
        pytorch_err = info.get('pytorch_error', '')
        if pytorch_err:
            set_val('pytorch_error', pytorch_err)
            self.env_labels['pytorch_error'].setStyleSheet("color: #e74c3c; font-size: 14px;")
        else:
            set_val('pytorch_error', '正常' if info.get('pytorch_installed') else '未安装')

        set_val('cuda_version', info.get('cuda_version', '---'))
        set_val('cuda_available', '是' if info.get('cuda_available') else '否')

        gpu_count = info.get('gpu_count', 0)
        set_val('gpu_count', str(gpu_count))
        gpu_names = info.get('gpu_names', [])
        set_val('gpu_names', ', '.join(gpu_names) if gpu_names else '---')

        set_val('nvidia_driver', info.get('nvidia_driver', '---'))
        set_val('nvidia_smi_cuda', info.get('nvidia_smi_cuda', '---'))
        set_val('vc_redist', info.get('vc_redist', '---'))
        set_val('ultralytics_version', info.get('ultralytics_version', '---'))
        set_val('opencv_version', info.get('opencv_version', '---'))
        set_val('platform', info.get('platform', '---'))

        # 自动判断GPU可用性
        if info.get('cuda_available'):
            self.gpu_radio.setChecked(True)
            self.device_combo.setCurrentText('auto')
        else:
            self.cpu_radio.setChecked(True)
            self.device_combo.setCurrentText('cpu')

        # 日志输出错误详情
        if pytorch_err:
            self._on_log(f"⚠ PyTorch 问题: {pytorch_err}")

        self.detect_env_btn.setEnabled(True)
        self._on_log("=== 环境检测完成 ===")

    def _install_pytorch(self):
        """安装PyTorch"""
        mirror_name = self.mirror_combo.currentText()
        mirror_url = self.mirrors.get(mirror_name, '')
        cuda_ver = self.cuda_combo.currentText()

        self.install_pt_btn.setEnabled(False)
        self._on_log(f"开始安装 PyTorch (CUDA: {cuda_ver}, 镜像: {mirror_name})...")

        thread = threading.Thread(target=self._run_install_pt, args=(mirror_url, cuda_ver), daemon=True)
        thread.start()

    def _run_install_pt(self, mirror_url, cuda_ver):
        success, msg = self.manager.install_pytorch(mirror_url, cuda_ver)
        self._on_log(msg)
        self.install_pt_btn.setEnabled(True)
        if success:
            self._detect_environment()

    def _install_ultralytics(self):
        """安装Ultralytics"""
        mirror_name = self.mirror_combo.currentText()
        mirror_url = self.mirrors.get(mirror_name, '')

        self.install_ultra_btn.setEnabled(False)
        self._on_log("开始安装 Ultralytics...")

        thread = threading.Thread(target=self._run_install_ultra, args=(mirror_url,), daemon=True)
        thread.start()

    def _run_install_ultra(self, mirror_url):
        success, msg = self.manager.install_ultralytics(mirror_url)
        self._on_log(msg)
        self.install_ultra_btn.setEnabled(True)
        if success:
            self._detect_environment()

    def _fix_pytorch(self):
        """强制重装修复PyTorch DLL问题"""
        self.fix_pt_btn.setEnabled(False)
        self._on_log("=== 开始修复 PyTorch ===")
        self._on_log("1. 卸载现有版本...")
        self._on_log("2. 清理缓存...")
        self._on_log("3. 重新安装...")
        thread = threading.Thread(target=self._run_fix_pt, daemon=True)
        thread.start()

    def _run_fix_pt(self):
        mirror_name = self.mirror_combo.currentText()
        mirror_url = self.mirrors.get(mirror_name, '')
        success, msg = self.manager.fix_pytorch(mirror_url)
        self._on_log(msg)
        self.fix_pt_btn.setEnabled(True)
        if success:
            self._on_log("修复完成，重新检测环境...")
            self._detect_environment()

    # ========== 训练操作 ==========

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据集配置文件", "", "YAML (*.yaml *.yml);;All (*)")
        if path:
            self.data_yaml_edit.setText(path)
            remember_last_data_yaml(path)
            self._validate_current_yaml(path)

    def _generate_yaml(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择数据集根目录")
        if not dir_path:
            return
        yaml_path, consistency = auto_generate_yaml(dir_path)
        if yaml_path is None:
            QMessageBox.warning(self, "生成失败", "未找到图片目录，无法生成 data.yaml。\n请确保数据集包含 images/ 目录。")
            return
        self.data_yaml_edit.setText(yaml_path)
        remember_last_data_yaml(yaml_path)
        if consistency and not consistency.ok:
            self._show_yaml_issues(consistency)
        else:
            self._show_yaml_ok(consistency)

    def _validate_current_yaml(self, path):
        consistency = validate_yaml_consistency(path)
        if consistency.ok and not consistency.warnings():
            self._show_yaml_ok(consistency)
        else:
            self._show_yaml_issues(consistency)

    def _show_yaml_ok(self, consistency):
        if consistency and consistency.names:
            self.data_info_label.setText(
                f"✅ nc={consistency.nc} 类别: {', '.join(consistency.names[:8])}"
                f"{'...' if len(consistency.names) > 8 else ''}"
            )
            self.data_info_label.setStyleSheet("color: #27ae60; font-weight: bold;")
            self._update_dataset_status(consistency)
        else:
            self.data_info_label.setText("✅ 已加载")
            self.data_info_label.setStyleSheet("color: #27ae60;")

    def _update_dataset_status(self, consistency):
        """更新主窗口状态栏数据集信息"""
        try:
            mw = self.window()
            if hasattr(mw, 'dataset_info_label'):
                # 统计图片数
                import os, yaml
                data_yaml = self.data_yaml_edit.text()
                img_count = 0
                img_size = '?'
                if os.path.isfile(data_yaml):
                    with open(data_yaml, 'r') as f:
                        cfg = yaml.safe_load(f)
                    base = cfg.get('path', os.path.dirname(data_yaml))
                    if not os.path.isabs(base):
                        base = os.path.join(os.path.dirname(data_yaml), base)
                    for key in ('train', 'val'):
                        d = cfg.get(key, '')
                        if not os.path.isabs(d):
                            d = os.path.join(base, d)
                        if os.path.isdir(d):
                            for f in os.listdir(d):
                                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                                    img_count += 1
                                    if img_count == 1 and img_size == '?':
                                        try:
                                            import cv2
                                            sample = cv2.imread(os.path.join(d, f))
                                            if sample is not None:
                                                img_size = str(max(sample.shape[:2]))
                                        except Exception:
                                            pass
                nc = consistency.nc if consistency else 0
                mw.dataset_info_label.setText(
                    f"Dataset: {img_count} imgs | {nc} classes | img sz {img_size}"
                )
        except Exception:
            pass

    def _show_yaml_issues(self, consistency):
        parts = [f"nc={consistency.nc}, names×{len(consistency.names)}"]
        for msg in consistency.errors():
            parts.append(f"❌ {msg}")
        for msg in consistency.warnings():
            parts.append(f"⚠️ {msg}")
        self.data_info_label.setText('\n'.join(parts))
        self.data_info_label.setStyleSheet(
            "color: #e74c3c; font-weight: bold;" if consistency.errors()
            else "color: #e67e22; font-weight: bold;"
        )

    def _start_training(self):
        data_yaml = self.data_yaml_edit.text()
        if not data_yaml or not Path(data_yaml).exists():
            QMessageBox.warning(self, "警告", "请先选择数据集配置!")
            return

        device_map = {'自动选择': 'auto', 'GPU 0': '0', 'GPU 1': '1', 'CPU': 'cpu'}
        device = device_map.get(self.device_combo.currentText(), 'auto')
        if self.cpu_radio.isChecked():
            device = 'cpu'

        # 图片尺寸
        img_sz_text = self.imgsz_combo.currentText()
        img_size = self.imgsz_spin.value() if img_sz_text == '自定义' else int(img_sz_text)

        params = {
            'model': self.model_combo.currentText(),
            'data_yaml': data_yaml,
            'epochs': self.epochs_spin.value(),
            'batch_size': self.batch_spin.value(),
            'img_size': img_size,
            'learning_rate': self.lr_spin.value(),
            'warmup_epochs': self.warmup_spin.value(),
            'device': device,
            'workers': self.workers_spin.value(),
            'pretrained': self.pretrained_check.isChecked(),
            'augment': self.augment_check.isChecked(),
            'cos_lr': self.cos_lr_check.isChecked(),
            'optimizer': self.optimizer_combo.currentText(),
            'close_mosaic': self.mosaic_spin.value(),
            'rect': self.rect_check.isChecked(),
            'seed': self.seed_spin.value(),
            'project': 'runs',
            'name': 'train',
        }

        # 解析自定义 args
        custom_text = self.advanced_args_edit.toPlainText().strip()
        if custom_text:
            extra = {}
            for line in custom_text.split('\n'):
                line = line.strip()
                if '=' in line and not line.startswith('#'):
                    k, v = line.split('=', 1)
                    k, v = k.strip(), v.strip()
                    try:
                        v = float(v)
                        if v == int(v):
                            v = int(v)
                    except ValueError:
                        pass
                    extra[k] = v
            if extra:
                params['extra_args'] = extra

        # ── 检测可继续的训练 ──
        last_pt, best_pt, prev_dir = find_latest_checkpoint('runs', 'train')
        if last_pt:
            reply = QMessageBox.question(
                self, "发现未完成的训练",
                f"检测到上一次训练记录:\n{prev_dir}\n\n"
                f"是否从上次中断处继续训练？\n"
                f"（选择「否」将创建新训练目录 {params['name']}2/）",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
            )
            if reply == QMessageBox.Yes:
                params['resume'] = True
                params['resume_checkpoint'] = last_pt
                params['name'] = os.path.basename(prev_dir)
                self.log_text.append(f"从 {prev_dir} 继续训练...")
            # 否 → 自动递增到 train2/ train3/ ...

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setValue(0)
        self.log_text.clear()

        # 更新状态栏
        if self.status_callback:
            self.status_callback(True)

        # 更新监控页信息
        run_name = params.get('name', 'train')
        self.monitor_ds_label.setText(f"数据集: {Path(data_yaml).stem}")
        self.monitor_model_label.setText(f"模型: {params['model']} → runs/{run_name}/")

        # 清空指标表格
        for r in range(5):
            for c in range(2):
                self.metrics_table.setItem(r, c, QTableWidgetItem('---'))
                self.metrics_table.item(r, c).setTextAlignment(Qt.AlignCenter)
                self.metrics_table.item(r, c).setBackground(QColor('#ffffff'))
        self.metrics_epoch_label.setText("Epoch: ---")

        # 重置曲线
        self.loss_ax.clear()
        self.loss_ax.set_title('Loss', fontsize=10, color='#333')
        self.loss_ax.set_xlabel('Epoch')
        self.loss_ax.set_ylabel('Loss')
        self.loss_ax.grid(True, alpha=0.3)
        self.metric_ax.clear()
        self.metric_ax.set_title('mAP', fontsize=10, color='#333')
        self.metric_ax.set_xlabel('Epoch')
        self.metric_ax.set_ylabel('mAP')
        self.metric_ax.grid(True, alpha=0.3)
        self.curve_canvas.draw()

        # ── YAML 一致性校验 ──
        consistency = validate_yaml_consistency(data_yaml)
        if consistency.errors():
            msgs = '\n'.join(f"  • {m}" for m in consistency.errors())
            reply = QMessageBox.warning(
                self, "data.yaml 配置问题",
                f"data.yaml 存在以下问题:\n\n{msgs}\n\n建议修复后再训练。\n是否仍然继续？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply == QMessageBox.No:
                self.train_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return

        # ── 数据集验证 ──
        result = validate_dataset(data_yaml)
        if result.has_errors:
            dlg = ValidationDialog(result, self)
            choice = dlg.exec_()
            if choice == QDialog.Rejected:  # 取消
                self.train_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
            elif choice == 2:  # 自动修复
                fix_stats = auto_fix_issues(result)
                QMessageBox.information(
                    self, "修复完成",
                    f"已清理 {fix_stats['fixed_lines']} 行非法标注，"
                    f"删除 {fix_stats['removed_files']} 个无效标注文件。\n"
                    f"建议重新验证后再训练。"
                )
                self.train_btn.setEnabled(True)
                self.stop_btn.setEnabled(False)
                return
            # choice == 1 → 跳过，继续训练

        # 自动切换到监控页
        self.sub_tabs.setCurrentIndex(1)

        self.manager.start_training(params)

    def _stop_training(self):
        self.manager.stop_training()
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if self.status_callback:
            self.status_callback(False)
        self._on_log("训练已停止")

    def _on_progress(self, current, total, loss):
        pct = int(current / total * 100) if total > 0 else 0
        self.progress_bar.setValue(pct)
        self.progress_label.setText(f"Epoch: {current}/{total}" if total > 0 else f"进度: {current}")
        if loss > 0:
            self.eta_label.setText(f"Loss: {loss:.4f}")

    def _on_log(self, message):
        self.log_text.append(message)

    def _scroll_log_if_needed(self):
        """自动滚动到日志底部"""
        if getattr(self, '_auto_scroll', True):
            scrollbar = self.log_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    def _export_log(self):
        """导出训练日志到文件（含参数 + 结果快照）"""
        text = self.log_text.toPlainText()
        if not text.strip():
            return
        import datetime
        d = os.path.join(get_work_dir(), 'training_logs')
        os.makedirs(d, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        p = os.path.join(d, f'train_log_{ts}.txt')
        with open(p, 'w', encoding='utf-8') as f:
            f.write(f"# YOLO CODE 训练日志 — {ts}\n")
            f.write(f"# 数据集: {self.data_yaml_edit.text()}\n")
            img_sz_text = self.imgsz_combo.currentText()
            img_sz = self.imgsz_spin.value() if img_sz_text == '自定义' else img_sz_text
            f.write(f"# 模型: {self.model_combo.currentText()}\n")
            f.write(f"# Epochs: {self.epochs_spin.value()} | "
                    f"Batch: {self.batch_spin.value()} | "
                    f"ImgSz: {img_sz} | "
                    f"LR: {self.lr_spin.value()} | "
                    f"Opt: {self.optimizer_combo.currentText()} | "
                    f"Seed: {self.seed_spin.value()}\n")
            f.write(f"# 设备: {self.device_combo.currentText()} | "
                    f"Workers: {self.workers_spin.value()} | "
                    f"Mosaic: {self.mosaic_spin.value()} | "
                    f"Rect: {self.rect_check.isChecked()}\n")
            metrics = self.manager.get_training_metrics()
            best = metrics.get('best_metrics', {})
            if best:
                f.write(f"# 结果: mAP50={best.get('mAP50',0):.4f} "
                        f"mAP50-95={best.get('mAP50-95',0):.4f} "
                        f"box_loss={best.get('box_loss',0):.4f}\n")
            f.write("#" + "─" * 50 + "\n\n")
            f.write(text)
        QMessageBox.information(self, "导出完成", f"日志已保存至:\n{p}")

    def _on_metrics(self, metrics):
        if not metrics:
            return

        epoch = metrics.get('epoch', 0)
        vals = {
            'box_loss': metrics.get('box_loss', 0),
            'cls_loss': metrics.get('cls_loss', 0),
            'dfl_loss': metrics.get('dfl_loss', 0),
            'mAP50': metrics.get('mAP50', 0),
            'mAP50-95': metrics.get('mAP50-95', 0),
        }

        # 获取最佳值
        best = self.manager.best_metrics
        best_vals = {
            'box_loss': best.get('box_loss', 0),
            'cls_loss': best.get('cls_loss', 0),
            'dfl_loss': best.get('dfl_loss', 0),
            'mAP50': best.get('mAP50', 0),
            'mAP50-95': best.get('mAP50-95', 0),
        }

        # 判断方向：loss越低越好，mAP越高越好
        def is_current_best(key, cur, bst):
            if cur <= 0 or bst <= 0:
                return False
            if key.startswith('mAP'):
                return cur >= bst
            return cur <= bst

        # 更新表格（5行 × 2列：当前值 | 最佳值）
        metric_keys = ['box_loss', 'cls_loss', 'dfl_loss', 'mAP50', 'mAP50-95']
        for row, key in enumerate(metric_keys):
            cur = vals[key]
            bst = best_vals[key]
            best_flag = is_current_best(key, cur, bst)

            # 当前值单元格
            cur_item = QTableWidgetItem(f'{cur:.4f}')
            cur_item.setTextAlignment(Qt.AlignCenter)
            if best_flag and cur > 0:
                cur_item.setBackground(QColor('#ffe6cc'))
                cur_item.setForeground(QColor('#ff6b00'))
                font = cur_item.font()
                font.setBold(True)
                cur_item.setFont(font)
            self.metrics_table.setItem(row, 0, cur_item)

            # 最佳值单元格
            bst_item = QTableWidgetItem(f'{bst:.4f}')
            bst_item.setTextAlignment(Qt.AlignCenter)
            if bst > 0:
                bst_item.setBackground(QColor('#f9f9f9'))
            self.metrics_table.setItem(row, 1, bst_item)

        # 更新epoch标签
        total_epochs = self.epochs_spin.value()
        self.metrics_epoch_label.setText(f"Epoch: {epoch}/{total_epochs}")

        # 更新曲线
        all_metrics = self.manager.epoch_metrics
        if len(all_metrics) > 0:
            epochs = [m['epoch'] for m in all_metrics]
            box_vals = [m['box_loss'] for m in all_metrics]
            cls_vals = [m['cls_loss'] for m in all_metrics]
            dfl_vals = [m['dfl_loss'] for m in all_metrics]
            map50_vals = [m['mAP50'] for m in all_metrics]
            map50_95_vals = [m['mAP50-95'] for m in all_metrics]

            self.loss_ax.clear()
            self.loss_ax.plot(epochs, box_vals, '#e74c3c', label='box_loss', linewidth=1.2)
            self.loss_ax.plot(epochs, cls_vals, '#3498db', label='cls_loss', linewidth=1.2)
            self.loss_ax.plot(epochs, dfl_vals, '#2ecc71', label='dfl_loss', linewidth=1.2)
            self.loss_ax.set_title('Loss', fontsize=10, color='#333')
            self.loss_ax.set_xlabel('Epoch')
            self.loss_ax.set_ylabel('Loss')
            self.loss_ax.legend(fontsize=7, loc='upper right')
            self.loss_ax.grid(True, alpha=0.3)

            self.metric_ax.clear()
            self.metric_ax.plot(epochs, map50_vals, '#ff6b00', label='mAP50', linewidth=1.5)
            self.metric_ax.plot(epochs, map50_95_vals, '#996633', label='mAP50-95', linewidth=1.5)
            self.metric_ax.set_title('mAP', fontsize=10, color='#333')
            self.metric_ax.set_xlabel('Epoch')
            self.metric_ax.set_ylabel('mAP')
            self.metric_ax.legend(fontsize=7, loc='upper left')
            self.metric_ax.grid(True, alpha=0.3)

            self.training_figure.tight_layout(pad=2)
            self.curve_canvas.draw()

        QApplication.processEvents()


class InferenceWidget(QWidget):
    """模型推理页面"""
    frame_signal = pyqtSignal(object, list)
    INFERENCE_STYLE = """
        QGroupBox { font-size: 14px; margin-top: 8px; padding-top: 16px; }
        QGroupBox::title { color: #ff6b00; }
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = InferenceManager()
        self.current_results = None
        self.video_running = False
        self._init_ui()
        self.frame_signal.connect(self._show_result)

    def _init_ui(self):
        outer = QVBoxLayout(self)
        outer.setContentsMargins(12, 10, 12, 10)
        outer.setSpacing(8)

        # ========== 第一行：模型选择 ==========
        row1 = QHBoxLayout()
        row1.setSpacing(10)

        model_grp = QGroupBox("模型配置")
        m_layout = QHBoxLayout(model_grp)
        m_layout.setSpacing(6)
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("选择 .pt 模型文件...")
        m_layout.addWidget(self.model_path_edit, 1)
        m_br = QPushButton("浏览")
        m_br.clicked.connect(self._browse_model)
        m_layout.addWidget(m_br)
        m_load = QPushButton("加载模型")
        m_load.clicked.connect(self._load_model)
        m_layout.addWidget(m_load)
        self.model_info_label = QLabel("未加载模型")
        self.model_info_label.setStyleSheet("font-size: 14px; color: #999; padding: 0 5px;")
        m_layout.addWidget(self.model_info_label)
        row1.addWidget(model_grp, 3)

        # 推理设备
        dev_grp = QGroupBox("推理设备")
        d_layout = QHBoxLayout(dev_grp)
        self.infer_device_combo = QComboBox()
        self.infer_device_combo.addItems(['自动选择', 'CPU', 'GPU 0', 'GPU 1'])
        self.infer_device_combo.setCurrentText('自动选择')
        d_layout.addWidget(QLabel("设备:"))
        d_layout.addWidget(self.infer_device_combo)
        self.half_prec_check = QCheckBox("半精度(FP16)")
        d_layout.addWidget(self.half_prec_check)
        row1.addWidget(dev_grp, 1)

        outer.addLayout(row1)

        # ========== 第二行：推理参数 ==========
        row2 = QHBoxLayout()
        row2.setSpacing(10)

        # 参数区
        param_grp = QGroupBox("检测参数")
        p_layout = QGridLayout(param_grp)
        p_layout.setHorizontalSpacing(18)
        p_layout.setVerticalSpacing(4)

        # 置信度
        p_layout.addWidget(QLabel("置信度阈值:"), 0, 0)
        conf_row = QHBoxLayout()
        self.conf_label = QLabel("0.50")
        self.conf_label.setMinimumWidth(35)
        self.conf_label.setStyleSheet("font-weight: bold; color: #ff6b00;")
        self.conf_label.setAlignment(Qt.AlignCenter)
        conf_row.addWidget(self.conf_label)
        cr1 = QLabel("0.01")
        cr1.setStyleSheet("color: #bbb; font-size: 12px;")
        cr2 = QLabel("0.99")
        cr2.setStyleSheet("color: #bbb; font-size: 12px;")
        conf_row.addWidget(cr1)
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setRange(1, 99)
        self.conf_slider.setValue(50)
        self.conf_slider.valueChanged.connect(self._on_conf_changed)
        conf_row.addWidget(self.conf_slider)
        conf_row.addWidget(cr2)
        p_layout.addLayout(conf_row, 0, 1)

        # IoU
        p_layout.addWidget(QLabel("IoU 阈值:"), 1, 0)
        iou_row = QHBoxLayout()
        self.iou_label = QLabel("0.45")
        self.iou_label.setMinimumWidth(35)
        self.iou_label.setStyleSheet("font-weight: bold; color: #ff6b00;")
        self.iou_label.setAlignment(Qt.AlignCenter)
        iou_row.addWidget(self.iou_label)
        ir1 = QLabel("0.01")
        ir1.setStyleSheet("color: #bbb; font-size: 12px;")
        ir2 = QLabel("0.99")
        ir2.setStyleSheet("color: #bbb; font-size: 12px;")
        iou_row.addWidget(ir1)
        self.iou_slider = QSlider(Qt.Horizontal)
        self.iou_slider.setRange(1, 99)
        self.iou_slider.setValue(45)
        self.iou_slider.valueChanged.connect(self._on_iou_changed)
        iou_row.addWidget(self.iou_slider)
        iou_row.addWidget(ir2)
        p_layout.addLayout(iou_row, 1, 1)

        # 最大检测数
        p_layout.addWidget(QLabel("最大检测数:"), 2, 0)
        max_row = QHBoxLayout()
        self.max_det_spin = QSpinBox()
        self.max_det_spin.setRange(1, 1000)
        self.max_det_spin.setValue(300)
        max_row.addWidget(self.max_det_spin)
        max_row.addWidget(QLabel("个"))
        max_row.addStretch()
        p_layout.addLayout(max_row, 2, 1)

        row2.addWidget(param_grp, 3)

        # 预处理参数
        pre_grp = QGroupBox("预处理")
        pre_layout = QGridLayout(pre_grp)
        pre_layout.setHorizontalSpacing(12)

        pre_layout.addWidget(QLabel("推理尺寸:"), 0, 0)
        imgsz_row = QHBoxLayout()
        self.infer_imgsz_spin = QSpinBox()
        self.infer_imgsz_spin.setRange(32, 2048)
        self.infer_imgsz_spin.setValue(640)
        self.infer_imgsz_spin.setSingleStep(32)
        imgsz_row.addWidget(self.infer_imgsz_spin)
        imgsz_row.addWidget(QLabel("px"))
        imgsz_row.addStretch()
        pre_layout.addLayout(imgsz_row, 0, 1)

        pre_layout.addWidget(QLabel("批次大小:"), 0, 2)
        bs_row = QHBoxLayout()
        self.infer_batch_spin = QSpinBox()
        self.infer_batch_spin.setRange(1, 128)
        self.infer_batch_spin.setValue(1)
        bs_row.addWidget(self.infer_batch_spin)
        bs_row.addStretch()
        pre_layout.addLayout(bs_row, 0, 3)

        self.augment_infer_check = QCheckBox("测试增强(TTA)")
        pre_layout.addWidget(self.augment_infer_check, 1, 0)
        self.verbose_check = QCheckBox("详细输出")
        self.verbose_check.setChecked(True)
        pre_layout.addWidget(self.verbose_check, 1, 2)

        row2.addWidget(pre_grp, 2)

        outer.addLayout(row2)

        # ========== 第三行：输入源 ==========
        row3 = QHBoxLayout()
        row3.setSpacing(10)

        src_grp = QGroupBox("输入源")
        src_layout = QHBoxLayout(src_grp)
        src_layout.setSpacing(10)

        self.input_type_group = QButtonGroup()
        self.image_radio = QRadioButton("图片")
        self.video_radio = QRadioButton("视频")
        self.camera_radio = QRadioButton("摄像头")
        self.image_radio.setChecked(True)
        self.input_type_group.addButton(self.image_radio, 0)
        self.input_type_group.addButton(self.video_radio, 1)
        self.input_type_group.addButton(self.camera_radio, 2)
        src_layout.addWidget(self.image_radio)
        src_layout.addWidget(self.video_radio)
        src_layout.addWidget(self.camera_radio)

        src_layout.addWidget(QLabel("|"))
        self.input_path_edit = QLineEdit()
        self.input_path_edit.setReadOnly(True)
        self.input_path_edit.setPlaceholderText("选择图片文件...")
        src_layout.addWidget(self.input_path_edit, 1)

        self.camera_combo = QComboBox()
        self.camera_combo.setVisible(False)
        self.camera_combo.setMinimumWidth(140)
        self.camera_combo.setToolTip("选择摄像头设备（选择摄像头模式时自动检测）")
        self.camera_combo.addItem("选择摄像头后自动检测", -1)
        src_layout.addWidget(self.camera_combo)
        self.refresh_cam_btn = QPushButton("↻")
        self.refresh_cam_btn.setFixedWidth(28)
        self.refresh_cam_btn.setToolTip("刷新摄像头列表")
        self.refresh_cam_btn.setVisible(False)
        self.refresh_cam_btn.clicked.connect(self._refresh_cameras)
        src_layout.addWidget(self.refresh_cam_btn)
        self.debug_cam_btn = QPushButton("🔧")
        self.debug_cam_btn.setFixedWidth(28)
        self.debug_cam_btn.setToolTip("摄像头调试预览")
        self.debug_cam_btn.setVisible(False)
        self.debug_cam_btn.clicked.connect(self._debug_camera)
        src_layout.addWidget(self.debug_cam_btn)

        self.browse_input_btn = QPushButton("浏览")
        self.browse_input_btn.clicked.connect(self._browse_input)
        src_layout.addWidget(self.browse_input_btn)

        self.save_result_check = QCheckBox("保存")
        src_layout.addWidget(self.save_result_check)

        self.run_btn = QPushButton("▶ 开始推理")
        self.run_btn.setObjectName("btnTrain")
        self.run_btn.clicked.connect(self._run_inference)
        src_layout.addWidget(self.run_btn)

        self.stop_video_btn = QPushButton("■ 停止")
        self.stop_video_btn.setObjectName("btnStop")
        self.stop_video_btn.setVisible(False)
        self.stop_video_btn.clicked.connect(self._stop_video)
        src_layout.addWidget(self.stop_video_btn)

        self.image_radio.toggled.connect(lambda checked: checked and self._input_type_changed(0))
        self.video_radio.toggled.connect(lambda checked: checked and self._input_type_changed(1))
        self.camera_radio.toggled.connect(lambda checked: checked and self._input_type_changed(2))

        row3.addWidget(src_grp)
        outer.addLayout(row3)

        # ========== 第四行：结果显示 ==========
        mid_split = QSplitter(Qt.Horizontal)

        # 左侧：结果画布
        canvas_cont = QWidget()
        canvas_layout = QVBoxLayout(canvas_cont)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(4)

        self.result_canvas = QLabel()
        self.result_canvas.setMinimumSize(400, 300)
        self.result_canvas.setAlignment(Qt.AlignCenter)
        self.result_canvas.setStyleSheet(
            "background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;")
        canvas_layout.addWidget(self.result_canvas, 1)

        # 结果统计栏
        stat_bar = QWidget()
        stat_bar.setStyleSheet("background: #f9f9f9; border-radius: 4px; padding: 6px;")
        stat_layout = QHBoxLayout(stat_bar)
        stat_layout.setContentsMargins(10, 4, 10, 4)

        self.detect_count_label = QLabel("检测到 0 个目标")
        self.detect_count_label.setStyleSheet("font-size: 15px; font-weight: bold; color: #ff6b00; border: none;")
        self.inference_time_label = QLabel("")
        self.inference_time_label.setStyleSheet("color: #999; font-size: 14px; border: none;")

        class_stats = QLabel("")
        class_stats.setStyleSheet("color: #666; font-size: 13px; border: none;")

        stat_layout.addWidget(self.detect_count_label)
        stat_layout.addWidget(class_stats)
        stat_layout.addStretch()
        stat_layout.addWidget(self.inference_time_label)
        self.class_stats_label = class_stats

        canvas_layout.addWidget(stat_bar)
        mid_split.addWidget(canvas_cont)

        # 右侧：检测列表
        result_grp = QGroupBox("检测结果")
        rp_layout = QVBoxLayout(result_grp)
        rp_layout.setSpacing(10)

        # 头部统计 + 导出
        rp_head = QWidget()
        rp_head_layout = QHBoxLayout(rp_head)
        rp_head_layout.setContentsMargins(0, 0, 0, 6)
        self.result_summary = QLabel("就绪，等待推理...")
        self.result_summary.setStyleSheet("font-size: 14px; color: #666;")
        rp_head_layout.addWidget(self.result_summary)
        rp_head_layout.addStretch()
        self.export_csv_btn = QPushButton("导出 CSV")
        self.export_csv_btn.clicked.connect(self._export_results)
        self.export_csv_btn.setStyleSheet("padding: 4px 10px; font-size: 13px;")
        rp_head_layout.addWidget(self.export_csv_btn)
        self.export_json_btn = QPushButton("导出 JSON")
        self.export_json_btn.clicked.connect(self._export_json)
        self.export_json_btn.setStyleSheet("padding: 4px 10px; font-size: 13px;")
        rp_head_layout.addWidget(self.export_json_btn)
        rp_layout.addWidget(rp_head)

        # 表格
        self.detect_table = QTableWidget()
        self.detect_table.setColumnCount(7)
        self.detect_table.setHorizontalHeaderLabels(["#", "类别", "置信度", "X1", "Y1", "X2", "Y2"])
        # 列宽 — 给足空间
        self.detect_table.setColumnWidth(0, 40)
        self.detect_table.setColumnWidth(1, 120)
        self.detect_table.setColumnWidth(2, 90)
        self.detect_table.setColumnWidth(3, 80)
        self.detect_table.setColumnWidth(4, 80)
        self.detect_table.setColumnWidth(5, 80)
        self.detect_table.setColumnWidth(6, 80)
        self.detect_table.horizontalHeader().setStretchLastSection(True)
        self.detect_table.setAlternatingRowColors(True)
        self.detect_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.detect_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.detect_table.verticalHeader().setVisible(False)
        self.detect_table.setShowGrid(True)
        # 加大行高和字体
        self.detect_table.verticalHeader().setDefaultSectionSize(32)
        self.detect_table.setStyleSheet("""
            QTableWidget {
                font-size: 15px;
                gridline-color: #e8e8e8;
            }
            QTableWidget::item {
                padding: 8px 10px;
            }
            QHeaderView::section {
                padding: 8px 6px;
                font-size: 14px;
                font-weight: bold;
                color: #222222;
                background: #f5f5f5;
                border: none;
                border-bottom: 2px solid #ddd;
            }
        """)
        rp_layout.addWidget(self.detect_table, 1)

        mid_split.addWidget(result_grp)
        mid_split.setStretchFactor(0, 3)
        mid_split.setStretchFactor(1, 2)

        outer.addWidget(mid_split, 1)

    # ========== 事件处理 ==========

    def _on_conf_changed(self, v):
        self.conf_label.setText(f"{v / 100:.2f}")
        self.manager.conf_threshold = v / 100

    def _on_iou_changed(self, v):
        self.iou_label.setText(f"{v / 100:.2f}")
        self.manager.iou_threshold = v / 100

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型", "", "PyTorch (*.pt);;All (*)")
        if path:
            self.model_path_edit.setText(path)

    def _load_model(self):
        path = self.model_path_edit.text()
        if not path:
            QMessageBox.warning(self, "提示", "请先选择模型文件")
            return
        import time
        t0 = time.time()
        success, msg = self.manager.load_model(path)
        el = (time.time() - t0) * 1000
        if success:
            self.model_info_label.setText(f"✅ {Path(path).name} ({el:.0f}ms)")
            self.model_info_label.setStyleSheet("font-size: 14px; color: #2ecc71; padding: 0 5px;")
            # 更新预处理尺寸为模型原生尺寸
            try:
                sz = self.manager.model.model.args.get('imgsz', 640) if self.manager.model else 640
                self.infer_imgsz_spin.setValue(int(sz))
            except Exception:
                pass
        else:
            QMessageBox.critical(self, "错误", msg)

    def _input_type_changed(self, mode):
        self.stop_video_btn.setVisible(False)
        self.run_btn.setVisible(True)
        self.save_result_check.setVisible(mode == 0)

        if mode == 0:  # 图片
            self.browse_input_btn.setText("浏览")
            self.input_path_edit.setPlaceholderText("选择图片文件...")
            self.input_path_edit.clear()
            self.browse_input_btn.setVisible(True)
            self.input_path_edit.setVisible(True)
            self.camera_combo.setVisible(False)
            self.refresh_cam_btn.setVisible(False)
            self.debug_cam_btn.setVisible(False)
            self.infer_batch_spin.setEnabled(True)
            self.run_btn.setText("▶ 开始推理")
        elif mode == 1:  # 视频
            self.browse_input_btn.setText("浏览")
            self.input_path_edit.setPlaceholderText("选择视频文件...")
            self.input_path_edit.clear()
            self.browse_input_btn.setVisible(True)
            self.input_path_edit.setVisible(True)
            self.camera_combo.setVisible(False)
            self.refresh_cam_btn.setVisible(False)
            self.debug_cam_btn.setVisible(False)
            self.infer_batch_spin.setEnabled(False)
            self.run_btn.setText("▶ 开始推理")
        elif mode == 2:  # 摄像头
            self.browse_input_btn.setVisible(False)
            self.input_path_edit.setVisible(False)
            self.camera_combo.setVisible(True)
            self.refresh_cam_btn.setVisible(True)
            self.debug_cam_btn.setVisible(True)
            self.infer_batch_spin.setEnabled(False)
            self.run_btn.setText("▶ 开始摄像头推理")
            # 只列设备名称，不打开摄像头（无延迟）
            self._populate_camera_list()

    def _browse_input(self):
        if self.image_radio.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "选择图片", "",
                "图片 (*.jpg *.jpeg *.png *.bmp);;所有 (*)")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "",
                "视频 (*.mp4 *.avi *.mov *.mkv);;所有 (*)")
        if path:
            self.input_path_edit.setText(path)

    def _run_inference(self):
        if self.manager.model is None:
            QMessageBox.warning(self, "提示", "请先加载模型")
            return
        import time
        t0 = time.time()

        if self.image_radio.isChecked():
            img_path = self.input_path_edit.text()
            if not img_path:
                QMessageBox.warning(self, "提示", "请先选择图片")
                return
            dev_map = {'自动选择': None, 'CPU': 'cpu', 'GPU 0': '0', 'GPU 1': '1'}
            result, error = self.manager.run_image(
                img_path,
                imgsz=self.infer_imgsz_spin.value(),
                half=self.half_prec_check.isChecked(),
                augment=self.augment_infer_check.isChecked(),
                max_det=self.max_det_spin.value(),
                device=dev_map.get(self.infer_device_combo.currentText()),
            )
            if error:
                QMessageBox.critical(self, "错误", error)
                return
            el = (time.time() - t0) * 1000

            # ── 记录推理参数（可回溯） ──
            param_log = (
                f"[推理] {Path(img_path).name} | "
                f"模型: {Path(self.manager.model_path).name if self.manager.model_path else '?'} | "
                f"conf={self.manager.conf_threshold:.2f} | "
                f"imgsz={self.infer_imgsz_spin.value()} | "
                f"耗时={el:.0f}ms"
            )
            print(param_log)  # 终端可回溯

            detections = result[1] if result else []
            self.inference_time_label.setText(f"⏱ {el:.0f}ms  |  {self.infer_imgsz_spin.value()}px")
            self._show_result(result[0] if result else None, detections)

            # ── 无检测框时提示 ──
            if len(detections) == 0:
                conf_val = self.manager.conf_threshold
                QMessageBox.information(
                    self, "未检测到目标",
                    f"当前图片未检测到任何目标。\n\n"
                    f"可尝试以下操作：\n"
                    f"  • 降低置信度阈值（当前 conf={conf_val:.2f}）\n"
                    f"  • 确认模型类别与图片内容匹配\n"
                    f"  • 检查图片是否正常加载\n\n"
                    f"{param_log}"
                )

            if self.save_result_check.isChecked():
                self._save_inference_image(result[0] if result else None)

        elif self.video_radio.isChecked():
            self.video_running = True
            self.run_btn.setVisible(False)
            self.stop_video_btn.setVisible(True)
            self.manager.run_video(self.input_path_edit.text(), self._on_video_frame)

        elif self.camera_radio.isChecked():
            cam_id = self.camera_combo.currentData()
            if cam_id is None or cam_id < 0:
                QMessageBox.warning(self, "提示", "未检测到可用摄像头")
                return
            self.video_running = True
            self.run_btn.setVisible(False)
            self.stop_video_btn.setVisible(True)
            self.manager.run_camera(cam_id, self._on_video_frame)

    def _stop_video(self):
        self.manager.stop()
        self.video_running = False
        self.run_btn.setVisible(True)
        self.stop_video_btn.setVisible(False)

    def _show_result(self, annotated_img, detections):
        if annotated_img is not None:
            h, w = annotated_img.shape[:2]
            cw = self.result_canvas.width()
            ch = self.result_canvas.height()
            scale = min(cw / max(w, 1), ch / max(h, 1)) * 0.92
            nw, nh = max(int(w * scale), 1), max(int(h * scale), 1)
            img = cv2.resize(annotated_img, (nw, nh))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            q_img = QImage(img.data, nw, nh, nw * 3, QImage.Format_RGB888)
            self.result_canvas.setPixmap(QPixmap.fromImage(q_img))

        self.detect_table.setRowCount(len(detections))
        classes = {}
        for i, det in enumerate(detections):
            bbox = det.get('bbox', [0, 0, 0, 0])
            cls_name = det.get('class_name', '')
            conf = det.get('confidence', 0)

            for c in range(7):
                item = QTableWidgetItem()
                item.setTextAlignment(Qt.AlignCenter)
                item.setFont(QFont('Microsoft YaHei', 10))
                self.detect_table.setItem(i, c, item)

            self.detect_table.item(i, 0).setText(str(i + 1))
            self.detect_table.item(i, 1).setText(cls_name)
            self.detect_table.item(i, 1).setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
            ci = QTableWidgetItem()
            ci.setText(f"{conf:.2f}")
            ci.setTextAlignment(Qt.AlignCenter)
            ci.setFont(QFont('Microsoft YaHei', 10, QFont.Bold))
            ci.setForeground(QColor('#222222'))
            self.detect_table.setItem(i, 2, ci)
            for j, v in enumerate(bbox[:4]):
                self.detect_table.item(i, 3 + j).setText(f"{v:.0f}")
            classes[cls_name] = classes.get(cls_name, 0) + 1

        self.detect_count_label.setText(f"检测到 {len(detections)} 个目标")
        cls_str = '  |  '.join(f"{k}×{v}" for k, v in list(classes.items())[:5])
        self.class_stats_label.setText(cls_str)
        self.result_summary.setText(f"共 {len(detections)} 个目标，{len(classes)} 个类别")

    def _on_video_frame(self, frame, detections):
        self.frame_signal.emit(frame, detections)

    def _save_inference_image(self, img):
        from utils.config import get_work_dir
        import datetime
        d = os.path.join(get_work_dir(), 'inference_results')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f'infer_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.jpg')
        cv2.imwrite(p, img)
        QMessageBox.information(self, "已保存", f"保存至:\n{p}")

    def _export_results(self):
        if self.detect_table.rowCount() == 0:
            return
        from utils.config import get_work_dir
        import csv, datetime
        d = os.path.join(get_work_dir(), 'inference_results')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f'results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        with open(p, 'w', newline='', encoding='utf-8-sig') as f:
            w = csv.writer(f)
            w.writerow(['序号', '类别', '置信度', 'X1', 'Y1', 'X2', 'Y2'])
            for r in range(self.detect_table.rowCount()):
                w.writerow([self.detect_table.item(r, c).text() for c in range(7)])
        QMessageBox.information(self, "导出完成", f"保存至:\n{p}")

    def _export_json(self):
        if self.detect_table.rowCount() == 0:
            return
        from utils.config import get_work_dir
        import json, datetime
        d = os.path.join(get_work_dir(), 'inference_results')
        os.makedirs(d, exist_ok=True)
        p = os.path.join(d, f'results_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        results = []
        for r in range(self.detect_table.rowCount()):
            results.append({
                'class': self.detect_table.item(r, 1).text(),
                'confidence': float(self.detect_table.item(r, 2).text()),
                'bbox': {
                    'x1': int(float(self.detect_table.item(r, 3).text())),
                    'y1': int(float(self.detect_table.item(r, 4).text())),
                    'x2': int(float(self.detect_table.item(r, 5).text())),
                    'y2': int(float(self.detect_table.item(r, 6).text())),
                }
            })
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        QMessageBox.information(self, "导出完成", f"保存至:\n{p}")

    def _get_friendly_camera_names(self):
        """通过PowerShell获取摄像头友好名称（不打开摄像头，无延迟）"""
        names = []
        try:
            import subprocess
            result = subprocess.run(
                ['powershell', '-Command',
                 'Get-PnpDevice -Class Camera -Status OK | Select-Object -ExpandProperty FriendlyName'],
                capture_output=True, text=True, timeout=8
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    name = line.strip()
                    if name:
                        names.append(name)
        except Exception:
            pass
        return names

    def _populate_camera_list(self):
        """填充摄像头下拉列表（仅用设备名，不打开摄像头）"""
        self.camera_combo.clear()
        names = self._get_friendly_camera_names()
        if names:
            for idx, name in enumerate(names):
                self.camera_combo.addItem(f"📹 {name}", idx)
        else:
            for idx in range(4):
                self.camera_combo.addItem(f"📹 摄像头 {idx}", idx)

    def _refresh_cameras(self):
        """刷新摄像头列表（打开设备检测可用性）"""
        self.camera_combo.clear()
        self.camera_combo.addItem("检测中...", -1)
        QApplication.processEvents()

        names = self._get_friendly_camera_names()
        if not names:
            names = [f"摄像头 {i}" for i in range(4)]

        found = False
        for idx in range(len(names)):
            try:
                cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
                if cap.isOpened():
                    name = names[idx] if idx < len(names) else f"摄像头 {idx}"
                    self.camera_combo.addItem(f"📹 {name}", idx)
                    found = True
                    cap.release()
            except Exception:
                pass

        if not found:
            self.camera_combo.addItem("未检测到可用摄像头", -1)

    def _debug_camera(self):
        """摄像头调试 - 打开预览窗口"""
        cam_id = self.camera_combo.currentData()
        if cam_id is None or cam_id < 0:
            QMessageBox.warning(self, "提示", "请先选择可用摄像头")
            return
        QMessageBox.information(self, "摄像头调试",
            f"即将打开摄像头 {cam_id} 预览窗口\n\n按 ESC 关闭预览窗口")

        # 在新线程中打开摄像头预览
        def preview():
            cap = cv2.VideoCapture(cam_id, cv2.CAP_DSHOW)
            if not cap.isOpened():
                return
            cv2.namedWindow('摄像头调试预览', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('摄像头调试预览', 640, 480)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                cv2.imshow('摄像头调试预览', frame)
                key = cv2.waitKey(30) & 0xFF
                if key == 27:  # ESC
                    break
                try:
                    if cv2.getWindowProperty('摄像头调试预览', cv2.WND_PROP_VISIBLE) < 1:
                        break
                except Exception:
                    break
            cap.release()
            try:
                cv2.destroyWindow('摄像头调试预览')
            except Exception:
                pass

        threading.Thread(target=preview, daemon=True).start()


class EvaluationWidget(QWidget):
    """模型评估页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = EvaluationManager()
        self.manager.set_callbacks(
            log_callback=self._on_log,
            progress_callback=self._on_progress
        )
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)

        # 左侧面板
        left_panel = QWidget()
        left_panel.setFixedWidth(320)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # 模型选择
        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)

        model_layout.addWidget(QLabel("模型文件:"))
        model_file_layout = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setReadOnly(True)
        self.model_path_edit.setPlaceholderText("选择.pt模型文件...")
        model_browse_btn = QPushButton("浏览")
        model_browse_btn.clicked.connect(self._browse_model)
        model_file_layout.addWidget(self.model_path_edit)
        model_file_layout.addWidget(model_browse_btn)
        model_layout.addLayout(model_file_layout)

        left_layout.addWidget(model_group)

        # 数据集选择
        data_group = QGroupBox("数据集选择")
        data_layout = QVBoxLayout(data_group)

        data_layout.addWidget(QLabel("数据集配置:"))
        data_file_layout = QHBoxLayout()
        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setReadOnly(True)
        self.data_yaml_edit.setPlaceholderText("选择data.yaml...")
        data_browse_btn = QPushButton("浏览")
        data_browse_btn.clicked.connect(self._browse_data)
        data_file_layout.addWidget(self.data_yaml_edit)
        data_file_layout.addWidget(data_browse_btn)
        data_layout.addLayout(data_file_layout)

        left_layout.addWidget(data_group)

        # 评估参数
        param_group = QGroupBox("评估参数")
        param_layout = QVBoxLayout(param_group)

        param_layout.addWidget(QLabel("置信度阈值:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.01, 1.0)
        self.conf_spin.setValue(0.5)
        self.conf_spin.setSingleStep(0.05)
        param_layout.addWidget(self.conf_spin)

        param_layout.addWidget(QLabel("IoU阈值:"))
        self.iou_spin = QDoubleSpinBox()
        self.iou_spin.setRange(0.01, 1.0)
        self.iou_spin.setValue(0.45)
        self.iou_spin.setSingleStep(0.05)
        param_layout.addWidget(self.iou_spin)

        left_layout.addWidget(param_group)

        # 评估按钮
        self.eval_btn = QPushButton("▶ 开始评估")
        self.eval_btn.setObjectName("btnTrain")
        self.eval_btn.clicked.connect(self._start_evaluation)
        left_layout.addWidget(self.eval_btn)

        # 导出模型
        export_group = QGroupBox("模型导出")
        export_layout = QVBoxLayout(export_group)

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("导出格式:"))
        self.export_format = QComboBox()
        self.export_format.addItems(['onnx', 'torchscript', 'tflite', 'openvino', 'engine'])
        format_layout.addWidget(self.export_format)
        export_layout.addLayout(format_layout)

        self.export_btn = QPushButton("导出模型")
        self.export_btn.clicked.connect(self._export_model)
        export_layout.addWidget(self.export_btn)

        left_layout.addWidget(export_group)
        left_layout.addStretch()

        # 右侧面板
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # 评估进度
        self.progress_bar = QProgressBar()
        right_layout.addWidget(self.progress_bar)

        # 结果展示
        result_group = QGroupBox("评估结果")
        result_layout = QVBoxLayout(result_group)

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "值"])
        self.metrics_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        result_layout.addWidget(self.metrics_table)

        right_layout.addWidget(result_group)

        # 日志
        log_group = QGroupBox("评估日志")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)

        # 组合
        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        layout.addWidget(splitter)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch (*.pt);;All (*)")
        if path:
            self.model_path_edit.setText(path)

    def _browse_data(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择数据集配置文件", "", "YAML (*.yaml *.yml);;All (*)")
        if path:
            self.data_yaml_edit.setText(path)

    def _start_evaluation(self):
        model_path = self.model_path_edit.text()
        data_yaml = self.data_yaml_edit.text()

        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "警告", "请先选择模型文件!")
            return
        if not data_yaml or not Path(data_yaml).exists():
            QMessageBox.warning(self, "警告", "请先选择数据集配置!")
            return

        self.eval_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.log_text.clear()
        self.manager.evaluate(
            model_path, data_yaml,
            conf=self.conf_spin.value(),
            iou=self.iou_spin.value()
        )

    def _export_model(self):
        model_path = self.model_path_edit.text()
        if not model_path or not Path(model_path).exists():
            QMessageBox.warning(self, "警告", "请先选择模型文件!")
            return

        fmt = self.export_format.currentText()
        success, result = self.manager.export_model(model_path, fmt)
        if success:
            QMessageBox.information(self, "导出成功", f"模型已导出到:\n{result}")
        else:
            QMessageBox.critical(self, "导出失败", str(result))

    def _on_log(self, message):
        self.log_text.append(message)
        QApplication.processEvents()

    def _on_progress(self, value):
        self.progress_bar.setValue(value)

        # 评估完成后更新结果表格
        if value >= 100 and self.manager.results:
            results = self.manager.results
            self.metrics_table.setRowCount(len(results))
            metrics_names = {
                'mAP50': 'mAP@0.5',
                'mAP50-95': 'mAP@0.5:0.95',
                'precision': 'Precision',
                'recall': 'Recall',
                'f1_score': 'F1 Score'
            }
            row = 0
            for key, display_name in metrics_names.items():
                if key in results:
                    self.metrics_table.setItem(row, 0, QTableWidgetItem(display_name))
                    self.metrics_table.setItem(row, 1, QTableWidgetItem(f"{results[key]:.4f}"))
                    row += 1

            self.eval_btn.setEnabled(True)


class DashboardWidget(QWidget):
    """仪表盘模块"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.card_labels = {}
        self._init_ui()

    def _init_ui(self):
        from utils.config import get_work_dir, auto_scan

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 15, 20, 15)
        layout.setSpacing(15)

        # 欢迎标题
        title_row = QHBoxLayout()
        title_col = QVBoxLayout()
        title = QLabel("YOLO CODE 模型训练平台")
        title.setStyleSheet("font-size: 24px; font-weight: bold; color: #333; padding: 5px 0;")
        subtitle = QLabel("一体化标注、训练、推理解决方案")
        subtitle.setStyleSheet("font-size: 15px; color: #999; padding-bottom: 5px;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        title_row.addLayout(title_col)
        title_row.addStretch()
        layout.addLayout(title_row)

        # 工作目录提示条 — 蓝色房屋风格
        wd_bar = QFrame()
        wd_bar.setStyleSheet("""
            QFrame#wdBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e3f2fd, stop:1 #bbdefb);
                border: 1px solid #90caf9;
                border-radius: 8px;
                padding: 8px 14px;
            }
        """)
        wd_bar.setObjectName("wdBar")
        wd_bar_layout = QHBoxLayout(wd_bar)
        wd_bar_layout.setContentsMargins(12, 6, 12, 6)
        wd_bar_layout.setSpacing(10)

        # 蓝色房子图标按钮
        home_btn = QPushButton("🏠")
        home_btn.setFixedSize(38, 38)
        home_btn.setCursor(Qt.PointingHandCursor)
        home_btn.setToolTip("点击设置工作目录")
        home_btn.setStyleSheet("""
            QPushButton {
                background: #1e88e5; border: 2px solid #1565c0;
                border-radius: 19px; font-size: 20px;
            }
            QPushButton:hover {
                background: #42a5f5; border-color: #1e88e5;
            }
        """)
        home_btn.clicked.connect(self._set_work_dir)
        wd_bar_layout.addWidget(home_btn)

        # 工作目录描述
        from utils.config import get_work_dir
        wd_path = get_work_dir()
        wd_name = os.path.basename(wd_path.rstrip('/\\')) or wd_path
        self.work_dir_label = QLabel(f"工作目录: <b>{wd_name}</b>")
        self.work_dir_label.setStyleSheet("font-size: 15px; color: #1565c0; border: none; background: transparent;")
        self.work_dir_label.setToolTip(wd_path)
        wd_bar_layout.addWidget(self.work_dir_label)

        wd_bar_layout.addStretch()

        # 子目录快捷信息
        scan = auto_scan(wd_path)
        info_text = f"📊 {len(scan['datasets'])} 数据集  |  🖼 {scan['total_images']} 图片  |  🤖 {len(scan['models'])} 模型"
        info_label = QLabel(info_text)
        info_label.setStyleSheet("font-size: 14px; color: #1976d2; border: none; background: transparent;")
        wd_bar_layout.addWidget(info_label)

        layout.addWidget(wd_bar)

        # 统计卡片行
        cards_layout = QHBoxLayout()
        cards_layout.setSpacing(15)

        stats = [
            ("📊", "数据集", "0", "#ff6b00"),
            ("🖼", "图片数量", "0", "#2ecc71"),
            ("🏷", "标注数量", "0", "#3498db"),
            ("🤖", "模型文件", "0", "#9b59b6"),
        ]
        for icon, name, value, color in stats:
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{ background: #fff; border: 1px solid #eee; border-radius: 8px;
                         border-left: 4px solid {color}; padding: 12px; }}
            """)
            card_layout_inner = QVBoxLayout(card)
            card_layout_inner.setSpacing(4)
            icon_lbl = QLabel(icon)
            icon_lbl.setStyleSheet("font-size: 24px; border: none;")
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-size: 13px; color: #999; border: none;")
            val_lbl = QLabel(value)
            val_lbl.setStyleSheet(f"font-size: 28px; font-weight: bold; color: {color}; border: none;")
            self.card_labels[name] = val_lbl
            card_layout_inner.addWidget(icon_lbl)
            card_layout_inner.addWidget(name_lbl)
            card_layout_inner.addWidget(val_lbl)
            cards_layout.addWidget(card)

        layout.addLayout(cards_layout)

        # 快捷操作
        quick_group = QGroupBox("快捷操作")
        quick_layout = QHBoxLayout(quick_group)
        quick_layout.setSpacing(10)

        quick_actions = [
            ("📁", "创建数据集", "创建新的训练数据集", 2),
            ("🖊", "开始标注", "打开标注工具", 1),
            ("🚀", "快速训练", "使用默认配置训练", 3),
            ("🔍", "模型推理", "对图片进行检测", 4),
        ]
        self.nav_callback = None  # 由MainWindow设置
        for icon, title_text, desc, target_index in quick_actions:
            btn = QPushButton(f"{icon}\n{title_text}")
            btn.setMinimumHeight(70)
            btn.setMaximumWidth(180)
            btn.setStyleSheet("""
                QPushButton { background: #fff; border: 1px solid #e0e0e0; border-radius: 6px;
                              font-size: 14px; color: #555; text-align: center; padding: 10px; }
                QPushButton:hover { background: #fff3e6; border-color: #ff6b00; color: #ff6b00; }
            """)
            btn.setToolTip(desc)
            btn.clicked.connect(lambda checked, idx=target_index: self._on_quick_action(idx))
            quick_layout.addWidget(btn)

        layout.addWidget(quick_group)

        # 系统状态 + 最近活动
        bottom_row = QHBoxLayout()

        sys_group = QGroupBox("系统状态")
        sys_layout = QVBoxLayout(sys_group)
        sys_layout.setSpacing(6)
        sys_items = [
            ("Python 环境", "检测中..."),
            ("PyTorch 状态", "检测中..."),
            ("CUDA 状态", "检测中..."),
            ("磁盘空间", "检测中..."),
            ("CPU 占用", "检测中..."),
            ("GPU 占用", "检测中..."),
        ]
        self.sys_labels = {}
        for name, value in sys_items:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name}:"))
            val_lbl = QLabel(value)
            val_lbl.setStyleSheet("color: #999;")
            self.sys_labels[name] = val_lbl
            row.addWidget(val_lbl)
            row.addStretch()
            sys_layout.addLayout(row)
        # CPU/GPU 进度条
        self.cpu_bar = QProgressBar(self)
        self.cpu_bar.setMaximum(100)
        self.cpu_bar.setMaximumHeight(8)
        self.cpu_bar.setTextVisible(False)
        self.cpu_bar.setStyleSheet("QProgressBar { background: #eee; border: none; border-radius: 4px; } "
                                    "QProgressBar::chunk { background: #3498db; border-radius: 4px; }")
        sys_layout.addWidget(self.cpu_bar)

        self.gpu_bar = QProgressBar(self)
        self.gpu_bar.setMaximum(100)
        self.gpu_bar.setMaximumHeight(8)
        self.gpu_bar.setTextVisible(False)
        self.gpu_bar.setStyleSheet("QProgressBar { background: #eee; border: none; border-radius: 4px; } "
                                    "QProgressBar::chunk { background: #2ecc71; border-radius: 4px; }")
        sys_layout.addWidget(self.gpu_bar)

        # 定时刷新 — 延迟启动避免与UI初始化冲突
        self._monitor_timer = QTimer(self)
        self._monitor_timer.timeout.connect(self._safe_refresh)
        self._monitor_timer.setInterval(3000)
        QTimer.singleShot(2000, self._monitor_timer.start)  # 2秒后才开始

        bottom_row.addWidget(sys_group)

        activity_group = QGroupBox("最近活动")
        activity_layout = QVBoxLayout(activity_group)
        self.activity_text = QTextEdit()
        self.activity_text.setReadOnly(True)
        self.activity_text.setMaximumHeight(120)
        self.activity_text.setStyleSheet("font-size: 13px; color: #555;")
        activity_layout.addWidget(self.activity_text)
        bottom_row.addWidget(activity_group)

        layout.addLayout(bottom_row)
        layout.addStretch()

    def _safe_refresh(self):
        """带保护的刷新调用"""
        try:
            self._refresh_usage()
        except Exception:
            pass

    def _on_quick_action(self, index):
        """处理快捷操作点击"""
        if self.nav_callback:
            self.nav_callback(index)

    def _set_work_dir(self):
        """设置工作目录"""
        from utils.config import set_work_dir, auto_scan
        path = QFileDialog.getExistingDirectory(self, "选择工作目录")
        if not path:
            return
        set_work_dir(path)
        wd_name = os.path.basename(path.rstrip('/\\')) or path
        self.work_dir_label.setText(f"工作目录: <b>{wd_name}</b>")
        self.work_dir_label.setToolTip(path)
        self.refresh_stats()
        self.add_activity(f"切换工作目录: {wd_name}")

    def refresh_stats(self):
        """刷新统计数据"""
        from utils.config import get_work_dir, auto_scan
        scan = auto_scan(get_work_dir())
        self.card_labels.get('数据集', QLabel()).setText(str(len(scan['datasets'])))
        self.card_labels.get('图片数量', QLabel()).setText(str(scan['total_images']))
        self.card_labels.get('标注数量', QLabel()).setText(str(scan['total_labels']))
        self.card_labels.get('模型文件', QLabel()).setText(str(len(scan['models'])))

    def update_stats(self, stats_dict):
        self.refresh_stats()

    def update_sys_status(self, info):
        """更新系统状态"""
        if 'python_version' in info:
            self.sys_labels.get('Python 环境', QLabel()).setText(f"Python {info['python_version']}")
        if 'pytorch_installed' in info:
            pt = info.get('pytorch_version', '未安装')
            self.sys_labels.get('PyTorch 状态', QLabel()).setText(
                f"✅ {pt}" if info['pytorch_installed'] else "❌ 未安装")
        if 'nvidia_driver' in info:
            drv = info.get('nvidia_driver', '')
            self.sys_labels.get('CUDA 状态', QLabel()).setText(
                f"✅ GPU驱动 {drv}" if drv and drv != 'N/A' else "⚠ 未检测到GPU")
        if 'disk_free' in info:
            free = info.get('disk_free', 'N/A')
            total = info.get('disk_total', 'N/A')
            self.sys_labels.get('磁盘空间', QLabel()).setText(f"{free} / {total}")

    def _refresh_usage(self):
        """定时刷新CPU/GPU占用"""
        self._fail_count = getattr(self, '_fail_count', 0)
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=0.1)
            self.cpu_bar.setValue(int(cpu))
            self.sys_labels.get('CPU 占用', QLabel()).setText(f"{cpu:.1f}%")
            self._fail_count = 0
        except Exception:
            self.sys_labels.get('CPU 占用', QLabel()).setText("---")
            self._fail_count += 1

        try:
            import subprocess
            r = subprocess.run(
                ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=3,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split(',')
                gpu_util = float(parts[0].strip()) if len(parts) > 0 else 0
                mem_used = parts[1].strip() if len(parts) > 1 else '?'
                mem_total = parts[2].strip() if len(parts) > 2 else '?'
                self.gpu_bar.setValue(int(gpu_util))
                self.sys_labels.get('GPU 占用', QLabel()).setText(
                    f"{gpu_util:.0f}% | 显存 {mem_used}/{mem_total} MiB")
            else:
                self.sys_labels.get('GPU 占用', QLabel()).setText("无GPU")
        except Exception:
            self.sys_labels.get('GPU 占用', QLabel()).setText("---")

        # 连续失败10次后停止定时器
        if self._fail_count > 10:
            self._monitor_timer.stop()

    def add_activity(self, text):
        """添加活动记录"""
        from datetime import datetime
        ts = datetime.now().strftime('%H:%M:%S')
        self.activity_text.append(f"[{ts}] {text}")


class DatasetWidget(QWidget):
    """数据集管理模块"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 左侧：数据集操作
        left = QWidget()
        left.setFixedWidth(320)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # 创建/导入
        create_group = QGroupBox("数据集操作")
        create_layout = QVBoxLayout(create_group)
        create_layout.setSpacing(8)

        create_layout.addWidget(QLabel("数据集名称:"))
        self.ds_name_edit = QLineEdit()
        self.ds_name_edit.setPlaceholderText("输入数据集名称...")
        create_layout.addWidget(self.ds_name_edit)

        btn_row = QHBoxLayout()
        self.create_ds_btn = QPushButton("📁 创建数据集")
        self.create_ds_btn.clicked.connect(self._create_dataset)
        self.import_ds_btn = QPushButton("📥 导入数据")
        self.import_ds_btn.clicked.connect(self._import_data)
        btn_row.addWidget(self.create_ds_btn)
        btn_row.addWidget(self.import_ds_btn)
        create_layout.addLayout(btn_row)

        self.ds_status = QLabel("尚未创建数据集")
        self.ds_status.setStyleSheet("color: #999; font-size: 14px;")
        create_layout.addWidget(self.ds_status)
        left_layout.addWidget(create_group)

        # 数据集分割
        split_group = QGroupBox("训练/验证/测试分割")
        split_layout = QVBoxLayout(split_group)
        split_layout.setSpacing(8)

        ratio_row = QHBoxLayout()
        ratio_row.addWidget(QLabel("训练:"))
        self.train_spin = QSpinBox()
        self.train_spin.setRange(50, 90)
        self.train_spin.setValue(70)
        self.train_spin.setSuffix("%")
        ratio_row.addWidget(self.train_spin)
        ratio_row.addWidget(QLabel("验证:"))
        self.val_spin = QSpinBox()
        self.val_spin.setRange(5, 30)
        self.val_spin.setValue(20)
        self.val_spin.setSuffix("%")
        ratio_row.addWidget(self.val_spin)
        split_layout.addLayout(ratio_row)

        self.split_btn = QPushButton("✂ 执行分割")
        self.split_btn.clicked.connect(self._split_dataset)
        split_layout.addWidget(self.split_btn)
        left_layout.addWidget(split_group)

        # 导出
        export_group = QGroupBox("导出数据集")
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(8)

        format_row = QHBoxLayout()
        self.export_format_combo = QComboBox()
        self.export_format_combo.addItems(['YOLO Format', 'COCO JSON', 'VOC XML', 'TFRecord'])
        format_row.addWidget(QLabel("格式:"))
        format_row.addWidget(self.export_format_combo)
        export_layout.addLayout(format_row)

        self.export_ds_btn = QPushButton("📤 导出数据集")
        self.export_ds_btn.clicked.connect(self._export_dataset)
        export_layout.addWidget(self.export_ds_btn)
        left_layout.addWidget(export_group)

        left_layout.addStretch()

        # 右侧：数据集列表
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        list_group = QGroupBox("已有数据集")
        list_layout = QVBoxLayout(list_group)

        self.ds_table = QTableWidget()
        self.ds_table.setColumnCount(5)
        self.ds_table.setHorizontalHeaderLabels(["名称", "图片数", "类别数", "格式", "状态"])
        self.ds_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ds_table.setAlternatingRowColors(True)
        list_layout.addWidget(self.ds_table)

        self.refresh_btn = QPushButton("🔄 刷新列表")
        self.refresh_btn.clicked.connect(self._refresh_dataset_list)
        list_layout.addWidget(self.refresh_btn)

        right_layout.addWidget(list_group)

        # 统计
        stat_group = QGroupBox("数据集统计")
        stat_layout = QHBoxLayout(stat_group)
        items = [("总数据集", "0"), ("总图片", "0"), ("总类别", "0"), ("总标注", "0")]
        self.ds_stat_labels = {}
        for name, val in items:
            vbox = QVBoxLayout()
            lbl = QLabel(val)
            lbl.setStyleSheet("font-size: 20px; font-weight: bold; color: #ff6b00;")
            lbl.setAlignment(Qt.AlignCenter)
            self.ds_stat_labels[name] = lbl
            name_lbl = QLabel(name)
            name_lbl.setStyleSheet("font-size: 13px; color: #999;")
            name_lbl.setAlignment(Qt.AlignCenter)
            vbox.addWidget(lbl)
            vbox.addWidget(name_lbl)
            stat_layout.addLayout(vbox)
        right_layout.addWidget(stat_group)

        layout.addWidget(left)
        layout.addWidget(right)

    def _create_dataset(self):
        name = self.ds_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "提示", "请输入数据集名称")
            return
        import os
        ds_path = os.path.join(os.getcwd(), 'datasets', name)
        os.makedirs(os.path.join(ds_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(ds_path, 'labels'), exist_ok=True)
        self.ds_status.setText(f"数据集已创建: {ds_path}")
        self._refresh_dataset_list()

    def _import_data(self):
        dir_path = QFileDialog.getExistingDirectory(self, "选择图片目录")
        if dir_path:
            self.ds_status.setText(f"已导入: {dir_path}")
            self._refresh_dataset_list()

    def _split_dataset(self):
        QMessageBox.information(self, "提示", f"将按 {self.train_spin.value()}%/{self.val_spin.value()}% 分割数据集")

    def _export_dataset(self):
        fmt = self.export_format_combo.currentText()
        QMessageBox.information(self, "提示", f"将以 {fmt} 格式导出数据集")

    def _refresh_dataset_list(self):
        import os
        self.ds_table.setRowCount(0)
        ds_root = os.path.join(os.getcwd(), 'datasets')
        if os.path.isdir(ds_root):
            for ds_name in os.listdir(ds_root):
                ds_path = os.path.join(ds_root, ds_name)
                if os.path.isdir(ds_path):
                    imgs = len([f for f in os.listdir(os.path.join(ds_path, 'images', ''))
                              if f.lower().endswith(('.jpg','.png','.jpeg'))]) if os.path.isdir(os.path.join(ds_path, 'images')) else 0
                    row = self.ds_table.rowCount()
                    self.ds_table.insertRow(row)
                    self.ds_table.setItem(row, 0, QTableWidgetItem(ds_name))
                    self.ds_table.setItem(row, 1, QTableWidgetItem(str(imgs)))
                    self.ds_table.setItem(row, 2, QTableWidgetItem("---"))
                    self.ds_table.setItem(row, 3, QTableWidgetItem("YOLO"))
                    self.ds_table.setItem(row, 4, QTableWidgetItem("就绪" if imgs > 0 else "空"))


class ExportWidget(QWidget):
    """模型导出模块"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._init_ui()

    def _init_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(15)

        # 左侧
        left = QWidget()
        left.setFixedWidth(340)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        model_group = QGroupBox("模型选择")
        model_layout = QVBoxLayout(model_group)
        model_layout.setSpacing(8)

        model_layout.addWidget(QLabel("模型文件:"))
        m_row = QHBoxLayout()
        self.export_model_edit = QLineEdit()
        self.export_model_edit.setReadOnly(True)
        self.export_model_edit.setPlaceholderText("选择 .pt 模型文件...")
        m_browse = QPushButton("浏览")
        m_browse.clicked.connect(self._browse_model)
        m_row.addWidget(self.export_model_edit)
        m_row.addWidget(m_browse)
        model_layout.addLayout(m_row)

        self.export_model_info = QLabel("未选择模型")
        self.export_model_info.setStyleSheet("color: #999; font-size: 14px;")
        model_layout.addWidget(self.export_model_info)
        left_layout.addWidget(model_group)

        fmt_group = QGroupBox("导出格式")
        fmt_layout = QVBoxLayout(fmt_group)
        fmt_layout.setSpacing(8)

        formats = [
            ("ONNX", "跨平台推理部署", "onnx"),
            ("TensorRT", "NVIDIA GPU加速", "engine"),
            ("OpenVINO", "Intel硬件优化", "openvino"),
            ("TFLite", "移动端/边缘设备", "tflite"),
            ("TorchScript", "PyTorch序列化", "torchscript"),
            ("CoreML", "Apple设备部署", "coreml"),
        ]
        self.format_radios = {}
        self.format_group = QButtonGroup()
        for i, (name, desc, key) in enumerate(formats):
            rb = QRadioButton(f"{name} - {desc}")
            if i == 0:
                rb.setChecked(True)
            self.format_group.addButton(rb, i)
            self.format_radios[key] = rb
            fmt_layout.addWidget(rb)

        left_layout.addWidget(fmt_group)

        opt_group = QGroupBox("导出选项")
        opt_layout = QVBoxLayout(opt_group)
        opt_layout.setSpacing(6)

        self.half_check = QCheckBox("FP16 半精度 (减小模型体积)")
        self.dynamic_check = QCheckBox("动态输入尺寸")
        self.simplify_check = QCheckBox("ONNX 模型简化")
        self.simplify_check.setChecked(True)
        self.opset_spin = QSpinBox()
        self.opset_spin.setRange(9, 20)
        self.opset_spin.setValue(17)
        opset_row = QHBoxLayout()
        opset_row.addWidget(QLabel("ONNX Opset:"))
        opset_row.addWidget(self.opset_spin)
        opset_row.addStretch()

        opt_layout.addWidget(self.half_check)
        opt_layout.addWidget(self.dynamic_check)
        opt_layout.addWidget(self.simplify_check)
        opt_layout.addLayout(opset_row)
        left_layout.addWidget(opt_group)

        self.export_btn = QPushButton("🚀 开始导出")
        self.export_btn.setObjectName("btnTrain")
        self.export_btn.clicked.connect(self._export_model)
        left_layout.addWidget(self.export_btn)

        left_layout.addStretch()

        # 右侧
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)

        result_group = QGroupBox("导出结果")
        result_layout = QVBoxLayout(result_group)
        self.export_log = QTextEdit()
        self.export_log.setReadOnly(True)
        result_layout.addWidget(self.export_log)
        right_layout.addWidget(result_group)

        export_history_group = QGroupBox("导出历史")
        history_layout = QVBoxLayout(export_history_group)
        self.history_table = QTableWidget()
        self.history_table.setColumnCount(4)
        self.history_table.setHorizontalHeaderLabels(["模型", "格式", "大小", "时间"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table)
        right_layout.addWidget(export_history_group)

        layout.addWidget(left)
        layout.addWidget(right)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "选择模型文件", "", "PyTorch (*.pt);;All (*)")
        if path:
            self.export_model_edit.setText(path)
            self.export_model_info.setText(f"模型: {Path(path).name} ({os.path.getsize(path)/1024/1024:.1f} MB)")

    def _export_model(self):
        model_path = self.export_model_edit.text()
        if not model_path:
            QMessageBox.warning(self, "提示", "请先选择模型文件")
            return
        self.export_log.append(f"开始导出: {Path(model_path).name}")
        self.export_log.append("导出功能开发中...")


class TerminalWidget(QWidget):
    """终端模块 — Linux风格命令行"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.command_history = []
        self.history_index = -1
        self._init_ui()
        self._print_banner()

    TERM_STYLE = """
        QTextEdit {
            background: #000000;
            color: #ffffff;
            font-family: 'Consolas', 'Courier New', 'Microsoft YaHei', monospace;
            font-size: 14px;
            border: none;
            padding: 8px 10px;
            selection-background-color: #ffffff;
            selection-color: #000000;
        }
        QScrollBar:vertical { background: #000; width: 10px; }
        QScrollBar::handle:vertical { background: #444; border-radius: 5px; min-height: 30px; }
        QScrollBar::handle:vertical:hover { background: #666; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
    """

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # 输出区
        self.terminal_output = QTextEdit()
        self.terminal_output.setReadOnly(True)
        self.terminal_output.setTextInteractionFlags(Qt.TextSelectableByMouse | Qt.TextSelectableByKeyboard)
        self.terminal_output.setStyleSheet(self.TERM_STYLE)
        layout.addWidget(self.terminal_output, 1)

        # 输入行
        input_bar = QWidget()
        input_bar.setStyleSheet("background: #0a0a0a; border-top: 1px solid #333;")
        input_layout = QHBoxLayout(input_bar)
        input_layout.setContentsMargins(10, 6, 10, 6)
        input_layout.setSpacing(6)

        prompt = QLabel("$")
        prompt.setStyleSheet("color: #00ff00; font-family: Consolas; font-size: 14px; font-weight: bold; "
                            "background: transparent; border: none;")
        input_layout.addWidget(prompt)

        self.cmd_input = QLineEdit()
        self.cmd_input.setPlaceholderText("输入命令...")
        self.cmd_input.setStyleSheet("""
            QLineEdit { background: transparent; color: #ffffff;
                        font-family: 'Consolas', 'Courier New', 'Microsoft YaHei', monospace;
                        font-size: 14px; border: none; padding: 4px 0; }
        """)
        self.cmd_input.returnPressed.connect(self._run)
        input_layout.addWidget(self.cmd_input, 1)

        layout.addWidget(input_bar)

    def _print_banner(self):
        import sys, platform
        w = f"{30 * '─'}"
        lines = [
            f"┌{w}┐",
            f"│  YOLO CODE Terminal v1.2{' ' * (30 - 26)}│",
            f"│  Python {sys.version.split()[0]:<25}│",
            f"│  {platform.platform()[:28]:<28}│",
            f"└{w}┘",
            f"",
            f"  输入 help 查看可用命令",
            f"",
        ]
        for line in lines:
            self.terminal_output.append(line)

    def _write(self, text, color='#ffffff'):
        """写入彩色文本"""
        self.terminal_output.setTextColor(QColor(color))
        self.terminal_output.insertPlainText(text)

    def _writeln(self, text='', color='#ffffff'):
        self._write(text + '\n', color)
        self._scroll_bottom()

    def _run(self):
        cmd = self.cmd_input.text().strip()
        self.cmd_input.clear()
        if not cmd:
            return

        self.command_history.append(cmd)
        self.history_index = len(self.command_history)

        # 绿提示符 + 白命令
        self._write('$ ', '#00ff00')
        self._writeln(cmd, '#ffffff')

        if cmd == 'clear':
            self.terminal_output.clear()
            return
        if cmd == 'help':
            self._writeln('  可用命令:')
            self._writeln('    python <脚本>  - 运行Python脚本', '#aaaaaa')
            self._writeln('    pip <命令>     - Python包管理', '#aaaaaa')
            self._writeln('    dir / ls       - 列出文件', '#aaaaaa')
            self._writeln('    cd <路径>      - 切换目录', '#aaaaaa')
            self._writeln('    clear          - 清屏', '#aaaaaa')
            self._writeln('    help           - 显示帮助', '#aaaaaa')
            self._writeln('    任意系统命令   - 直接执行', '#aaaaaa')
            return

        # 执行命令
        import subprocess
        try:
            # 处理 cd 命令
            if cmd.startswith('cd '):
                target = cmd[3:].strip().strip('"')
                try:
                    os.chdir(os.path.expanduser(target))
                    self._writeln(f'  → {os.getcwd()}', '#888888')
                except Exception as e:
                    self._writeln(f'  cd: {e}', '#ff4444')
                return

            result = subprocess.run(
                cmd, shell=True, capture_output=True, text=True,
                timeout=30, cwd=os.getcwd(), encoding='utf-8', errors='replace'
            )
            if result.stdout:
                for line in result.stdout.rstrip().split('\n'):
                    self._writeln(line, '#ffffff')
            if result.stderr:
                for line in result.stderr.rstrip().split('\n'):
                    self._writeln(line, '#ff6666')
        except subprocess.TimeoutExpired:
            self._writeln('  命令超时 (30s)', '#ff4444')
        except Exception as e:
            self._writeln(f'  错误: {e}', '#ff4444')

    def _scroll_bottom(self):
        sb = self.terminal_output.verticalScrollBar()
        sb.setValue(sb.maximum())

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Up:
            if self.history_index > 0:
                self.history_index -= 1
                self.cmd_input.setText(self.command_history[self.history_index])
        elif event.key() == Qt.Key_Down:
            if self.history_index < len(self.command_history) - 1:
                self.history_index += 1
                self.cmd_input.setText(self.command_history[self.history_index])
            else:
                self.history_index = len(self.command_history)
                self.cmd_input.clear()
        else:
            super().keyPressEvent(event)


class AboutWidget(QWidget):
    """关于页面"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)

        title = QLabel("YOLO CODE 模型训练平台")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("font-size: 28px; font-weight: bold; color: #ff6b00; padding: 20px;")
        layout.addWidget(title)

        version_label = QLabel("版本 1.2.0")
        version_label.setAlignment(Qt.AlignCenter)
        version_label.setStyleSheet("font-size: 14px; color: #999999;")
        layout.addWidget(version_label)

        layout.addSpacing(30)

        desc = QLabel(
            "基于YOLO模型的一体化训练平台\n\n"
            "功能模块:\n"
            "  • 标注 - 手动绘制边界框进行数据标注\n"
            "  • 模型训练 - 支持YOLOv5/v8/v11模型的训练\n"
            "  • 模型推理 - 支持图片/视频/摄像头实时推理\n"
            "  • 模型评估 - 模型性能评估和指标分析\n\n"
            "技术栈: PyQt5 + Ultralytics YOLO + OpenCV"
        )
        desc.setAlignment(Qt.AlignCenter)
        desc.setStyleSheet("font-size: 14px; color: #555555; line-height: 1.8;")
        layout.addWidget(desc)

        layout.addSpacing(30)

        copyright_label = QLabel("© 2025 YOLO CODE. All rights reserved.")
        copyright_label.setAlignment(Qt.AlignCenter)
        copyright_label.setStyleSheet("font-size: 14px; color: #aaaaaa;")
        layout.addWidget(copyright_label)

        layout.addSpacing(10)

        contact_label = QLabel("📧 欢迎联系我们: 2807087688@qq.com")
        contact_label.setAlignment(Qt.AlignCenter)
        contact_label.setStyleSheet("font-size: 14px; color: #ff6b00;")
        layout.addWidget(contact_label)


class NavButton(QPushButton):
    """侧边栏导航按钮"""

    def __init__(self, icon, text, parent=None):
        super().__init__(parent)
        self.setText(f"  {icon}  {text}")
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(48)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO CODE 模型训练平台 v1.2.0")
        self.resize(1600, 950)
        self.setMinimumSize(1200, 700)
        self._init_ui()

    def _init_ui(self):
        # 设置样式
        self.setStyleSheet(ORANGE_WHITE_STYLE + SIDEBAR_STYLE)

        # 菜单栏
        menubar = self.menuBar()
        file_menu = menubar.addMenu("文件(&F)")
        open_act = QAction("打开数据集", self)
        open_act.triggered.connect(self._switch_to_annotation)
        file_menu.addAction(open_act)

        save_act = QAction("保存标注", self)
        save_act.triggered.connect(self._save_annotation_menu)
        file_menu.addAction(save_act)

        file_menu.addSeparator()
        exit_act = QAction("退出(&X)", self)
        exit_act.triggered.connect(self.close)
        file_menu.addAction(exit_act)

        help_menu = menubar.addMenu("帮助(&H)")
        about_act = QAction("关于", self)
        about_act.triggered.connect(lambda: self._switch_page(7))  # 关于
        help_menu.addAction(about_act)

        # 主布局：侧边栏 + 内容区
        central = QWidget()
        central.setObjectName("centralWidget")
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---- 左侧导航栏 ----
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(220)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        # Logo区域
        logo_widget = QWidget()
        logo_widget.setObjectName("logoWidget")
        logo_layout = QVBoxLayout(logo_widget)
        logo_layout.setContentsMargins(15, 20, 15, 15)

        logo_title = QLabel("YOLO CODE")
        logo_title.setObjectName("logoTitle")
        logo_title.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        logo_sub = QLabel("模型训练平台")
        logo_sub.setObjectName("logoSub")
        logo_sub.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        logo_layout.addWidget(logo_title)
        logo_layout.addWidget(logo_sub)
        sidebar_layout.addWidget(logo_widget)

        # 分隔线
        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setObjectName("sidebarSep")
        sidebar_layout.addWidget(sep)

        # 导航按钮
        self.nav_buttons = []
        nav_items = [
            ("📊", "仪表盘"),
            ("📝", "标注"),
            ("📁", "数据集"),
            ("🚀", "模型训练"),
            ("🔍", "模型推理"),
            ("📈", "模型评估"),
            ("📤", "模型导出"),
            ("ℹ️", "关于"),
            ("💻", "终端"),
        ]

        for icon, text in nav_items:
            btn = NavButton(icon, text)
            btn.clicked.connect(self._make_nav_handler(len(self.nav_buttons)))
            self.nav_buttons.append(btn)
            sidebar_layout.addWidget(btn)

        sidebar_layout.addStretch()

        # 底部版本信息
        ver_label = QLabel("v1.2.0")
        ver_label.setObjectName("versionLabel")
        ver_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(ver_label)
        sidebar_layout.addSpacing(10)

        main_layout.addWidget(sidebar)

        # 分隔线
        vline = QFrame()
        vline.setFrameShape(QFrame.VLine)
        vline.setObjectName("sidebarBorder")
        main_layout.addWidget(vline)

        # ---- 右侧内容区 ----
        self.content_stack = QStackedWidget()
        self.content_stack.setObjectName("contentStack")

        self.dashboard_widget = DashboardWidget()
        self.dashboard_widget.nav_callback = self._switch_page
        self.annotation_widget = AnnotationWidget()
        self.dataset_widget = DatasetWidget()
        self.training_widget = TrainingWidget()
        self.training_widget.status_callback = self.set_training_status
        self.inference_widget = InferenceWidget()
        self.evaluation_widget = EvaluationWidget()
        self.export_widget = ExportWidget()
        self.about_widget = AboutWidget()
        self.terminal_widget = TerminalWidget()

        self.content_stack.addWidget(self.dashboard_widget)      # 0
        self.content_stack.addWidget(self.annotation_widget)     # 1
        self.content_stack.addWidget(self.dataset_widget)        # 2
        self.content_stack.addWidget(self.training_widget)       # 3
        self.content_stack.addWidget(self.inference_widget)      # 4
        self.content_stack.addWidget(self.evaluation_widget)     # 5
        self.content_stack.addWidget(self.export_widget)         # 6
        self.content_stack.addWidget(self.about_widget)          # 7
        self.content_stack.addWidget(self.terminal_widget)       # 8

        main_layout.addWidget(self.content_stack)

        self.setCentralWidget(central)

        # 默认选中仪表盘
        self._switch_page(0)

        # 状态栏
        self.status_bar = QStatusBar()
        # 状态栏 — 自定义布局，右侧显示
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # 状态指示灯 + 信息标签
        status_widget = QWidget()
        status_layout = QHBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 0, 8, 0)
        status_layout.setSpacing(8)

        # 圆形状态指示灯
        self.status_indicator = QLabel()
        self.status_indicator.setFixedSize(10, 10)
        self._set_status_color('green')
        status_layout.addWidget(self.status_indicator)

        # 状态文本
        self.status_label = QLabel("就绪")
        self.status_label.setStyleSheet("font-size: 14px; color: #666;")
        status_layout.addWidget(self.status_label)

        # 分隔
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #ddd; max-width: 1px;")
        status_layout.addWidget(sep1)

        # GPU 信息
        self.gpu_label = QLabel("GPU: 检测中...")
        self.gpu_label.setStyleSheet("font-size: 14px; color: #888;")
        status_layout.addWidget(self.gpu_label)

        # 版本信息
        self.version_label = QLabel()
        self.version_label.setStyleSheet("font-size: 12px; color: #aaa;")
        status_layout.addWidget(self.version_label)

        # 数据集信息
        self.dataset_info_label = QLabel()
        self.dataset_info_label.setStyleSheet("font-size: 12px; color: #888;")
        status_layout.addWidget(self.dataset_info_label)

        status_layout.addStretch()

        self.status_bar.addWidget(status_widget)

        # ── 启动时检测版本 ──
        QTimer.singleShot(100, self._show_version_info)

        # 自动检测环境并更新仪表盘（延迟执行避免阻塞UI）
        QTimer.singleShot(500, self._refresh_dashboard)

    def _show_version_info(self):
        """状态栏显示 Python / torch / CUDA 版本"""
        parts = [f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"]
        try:
            import torch
            parts.append(f"PyTorch {torch.__version__}")
            if torch.cuda.is_available():
                parts.append(f"CUDA {torch.version.cuda}")
        except ImportError:
            parts.append("PyTorch 未安装")
        self.version_label.setText(" | ".join(parts))

    def _make_nav_handler(self, index):
        return lambda: self._switch_page(index)

    def _switch_page(self, index):
        self.content_stack.setCurrentIndex(index)
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)

    def _set_status_color(self, color):
        """设置状态指示灯颜色: 'green' 就绪, 'red' 训练中"""
        c = '#4caf50' if color == 'green' else '#f44336' if color == 'red' else '#ff9800'
        self.status_indicator.setStyleSheet(
            f"background-color: {c}; border-radius: 5px; min-width: 10px; max-width: 10px; "
            f"min-height: 10px; max-height: 10px;"
        )

    def set_training_status(self, is_training):
        """更新训练状态指示灯"""
        if is_training:
            self._set_status_color('red')
            self.status_label.setText("训练中")
        else:
            self._set_status_color('green')
            self.status_label.setText("就绪")

    def _refresh_dashboard(self):
        """启动时自动检测环境并更新仪表盘和训练页"""
        try:
            from core.training import detect_environment
        except Exception:
            return
        try:
            info = detect_environment()

            # 更新仪表盘系统状态
            self.dashboard_widget.update_sys_status(info)
            self.dashboard_widget.refresh_stats()
            self.dashboard_widget.add_activity("系统启动完成")

            # 同时更新训练页的环境信息面板
            tw = self.training_widget
            def _set(k, v):
                if k in tw.env_labels:
                    tw.env_labels[k].setText(str(v))

            _set('python_version', info.get('python_version', '---'))
            _set('python_arch', info.get('python_arch', '---'))
            _set('pytorch_version', info.get('pytorch_version', '---'))
            pt_err = info.get('pytorch_error', '')
            if pt_err:
                _set('pytorch_error', pt_err)
                tw.env_labels.get('pytorch_error', QLabel()).setStyleSheet("color: #e74c3c; font-size: 14px;")
            else:
                _set('pytorch_error', '正常' if info.get('pytorch_installed') else '未安装')
            _set('cuda_version', info.get('cuda_version', '---'))
            _set('cuda_available', '是' if info.get('cuda_available') else '否')
            gpu_count = info.get('gpu_count', 0)
            _set('gpu_count', str(gpu_count))
            gpu_names = info.get('gpu_names', [])
            _set('gpu_names', ', '.join(gpu_names) if gpu_names else '---')
            _set('nvidia_driver', info.get('nvidia_driver', '---'))
            _set('nvidia_smi_cuda', info.get('nvidia_smi_cuda', '---'))
            _set('vc_redist', info.get('vc_redist', '---'))
            _set('ultralytics_version', info.get('ultralytics_version', '---'))
            _set('opencv_version', info.get('opencv_version', '---'))
            _set('platform', info.get('platform', '---'))

            # 自动设置训练模式
            if info.get('cuda_available'):
                tw.gpu_radio.setChecked(True)
                tw.device_combo.setCurrentText('自动选择')
            else:
                tw.cpu_radio.setChecked(True)
                tw.device_combo.setCurrentText('CPU')

            # 更新状态栏 GPU 信息
            gpu_names = info.get('gpu_names', [])
            if gpu_names:
                gpu_text = ', '.join(gpu_names)
            elif info.get('nvidia_driver', 'N/A') != 'N/A':
                gpu_text = f"驱动 {info.get('nvidia_driver', '')}"
            else:
                gpu_text = '无'
            self.gpu_label.setText(f"GPU: {gpu_text}")

        except Exception:
            pass

    def _switch_to_annotation(self):
        self._switch_page(1)

    def _save_annotation_menu(self):
        self.annotation_widget._save_annotations()
