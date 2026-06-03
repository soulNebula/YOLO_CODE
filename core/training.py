import os
import sys
import subprocess
import threading


def _preload_torch_dll():
    """预配置PyTorch DLL搜索路径，解决Windows上c10.dll加载失败问题"""
    if sys.platform != 'win32':
        return

    # 收集所有需要添加的DLL目录
    dll_dirs = []

    # 方法1: 遍历sys.path找已安装的torch
    for p in list(sys.path):
        for lib_dir in ['torch/lib', 'torch/Lib', 'ultralytics']:
            full = os.path.join(p, lib_dir)
            if os.path.isdir(full):
                dll_dirs.append(full)

    # 方法2: pip show 查找
    if not any('torch' in d.lower() for d in dll_dirs):
        try:
            result = subprocess.run(
                [sys.executable, '-m', 'pip', 'show', 'torch'],
                capture_output=True, text=True, timeout=15
            )
            for line in result.stdout.split('\n'):
                if line.startswith('Location:'):
                    loc = line.split(':', 1)[1].strip()
                    full = os.path.join(loc, 'torch', 'lib')
                    if os.path.isdir(full):
                        dll_dirs.append(full)
                    break
        except Exception:
            pass

    # 方法3: 查找intel-openmp DLL目录
    try:
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'show', 'intel-openmp'],
            capture_output=True, text=True, timeout=15
        )
        for line in result.stdout.split('\n'):
            if line.startswith('Location:'):
                loc = line.split(':', 1)[1].strip()
                dll_dirs.append(loc)
                # 也加入 site-packages/bin
                bin_dir = os.path.join(loc, '..', 'bin')
                if os.path.isdir(bin_dir):
                    dll_dirs.append(os.path.normpath(bin_dir))
                break
    except Exception:
        pass

    # 添加到PATH环境变量（影响子进程和ctypes）
    for d in dll_dirs:
        if d not in os.environ.get('PATH', ''):
            os.environ['PATH'] = d + os.pathsep + os.environ.get('PATH', '')

    # 使用Windows API添加DLL搜索目录（影响当前进程的LoadLibrary）
    if hasattr(os, 'add_dll_directory'):
        for d in dll_dirs:
            try:
                os.add_dll_directory(d)
            except OSError:
                pass


def _try_preload_torch_dlls():
    """在import torch前，确保DLL搜索路径正确（不使用ctypes预加载以免冲突）"""
    if sys.platform != 'win32':
        return

    # 只使用 os.add_dll_directory()，不使用 ctypes 预加载
    # ctypes预加载可能与Python import的DLL加载机制冲突导致错误1114
    _preload_torch_dll()

    # 额外：将torch/lib也添加到当前进程的PATH（已有_preload_torch_dll处理）
    # 如果还是失败，可能是intel-openmp与torch内置libiomp5md.dll冲突
    # 尝试移除冲突的intel-openmp
    try:
        import ctypes
        for p in sys.path:
            torch_lib = os.path.join(p, 'torch', 'lib')
            if os.path.isdir(torch_lib):
                # 确保路径在最前面（优先级最高）
                try:
                    os.add_dll_directory(torch_lib)
                except Exception:
                    pass
                break
    except Exception:
        pass


def _get_win32_error_details():
    """获取Windows错误详细信息"""
    if sys.platform != 'win32':
        return ''

    import ctypes
    details = []

    # 检查c10.dll是否能加载
    for p in sys.path:
        torch_lib = os.path.join(p, 'torch', 'lib')
        c10_path = os.path.join(torch_lib, 'c10.dll')
        if not os.path.exists(c10_path):
            continue

        # 尝试用LoadLibrary加载并获取错误码
        kernel32 = ctypes.windll.kernel32
        kernel32.SetLastError(0)
        handle = kernel32.LoadLibraryExW(c10_path, None, 0x00000008)
        if handle:
            kernel32.FreeLibrary(handle)
            details.append(f'c10.dll 加载: 成功')
        else:
            err = kernel32.GetLastError()
            buf = ctypes.create_unicode_buffer(2048)
            kernel32.FormatMessageW(0x1000, None, err, 0, buf, 2048, None)
            err_msg = buf.value.strip() if buf.value else '未知错误'
            details.append(f'c10.dll 加载失败 (WinErr {err}): {err_msg}')

            # 常见错误码解释
            err_hints = {
                126: '→ 缺少依赖DLL (可能是VC++ Redist或OpenMP)',
                127: '→ 找不到指定的程序 (DLL版本不匹配)',
                193: '→ 不是有效的Win32程序 (32/64位不匹配)',
                1114: '→ DLL初始化例程失败 (依赖DLL损坏或版本冲突)',
            }
            if err in err_hints:
                details.append(err_hints[err])
        break

    # 检查PATH中是否有冲突的DLL
    torch_lib_found = None
    for p in sys.path:
        tl = os.path.join(p, 'torch', 'lib')
        if os.path.isdir(tl):
            torch_lib_found = tl
            break

    if torch_lib_found:
        details.append(f'Torch lib路径: {torch_lib_found}')
        in_path = '是' if torch_lib_found in os.environ.get('PATH', '') else '否'
        details.append(f'是否在PATH中: {in_path}')

    return '\n'.join(details)


# 模块导入时立即执行DLL预加载
_preload_torch_dll()


def detect_environment():
    """检测运行环境信息"""
    env_info = {}

    # Windows：预加载DLL路径
    _preload_torch_dll()

    # Python版本
    py_ver = sys.version_info
    env_info['python_version'] = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro}"
    env_info['python_full'] = sys.version
    env_info['platform'] = sys.platform
    env_info['os_name'] = os.name
    env_info['python_path'] = sys.executable
    env_info['python_arch'] = '64-bit' if sys.maxsize > 2**32 else '32-bit'

    # PyTorch
    env_info['pytorch_version'] = '检测失败'
    env_info['pytorch_installed'] = False
    env_info['pytorch_error'] = ''

    # Windows: 在import torch前用ctypes预加载c10.dll及其依赖
    if sys.platform == 'win32':
        _try_preload_torch_dlls()

    try:
        import torch
        env_info['pytorch_version'] = torch.__version__
        env_info['pytorch_installed'] = True
        env_info['pytorch_error'] = ''

        # CUDA (only if torch loaded OK)
        env_info['cuda_available'] = torch.cuda.is_available()
        env_info['cuda_version'] = torch.version.cuda if torch.cuda.is_available() else 'N/A'
        env_info['gpu_count'] = torch.cuda.device_count() if torch.cuda.is_available() else 0
        env_info['gpu_names'] = []
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                try:
                    env_info['gpu_names'].append(torch.cuda.get_device_name(i))
                except Exception:
                    env_info['gpu_names'].append(f'GPU #{i}')
    except ImportError:
        env_info['pytorch_version'] = '未安装'
        env_info['pytorch_error'] = 'PyTorch 未安装，请点击"安装 PyTorch"按钮'
        env_info['cuda_available'] = False
        env_info['cuda_version'] = 'N/A'
        env_info['gpu_count'] = 0
        env_info['gpu_names'] = []
    except OSError as e:
        env_info['pytorch_version'] = 'DLL加载失败'
        # 尝试获取详细的Windows错误信息
        win_err = _get_win32_error_details()
        env_info['pytorch_error'] = _analyze_dll_error(str(e)) + '\n' + win_err
        env_info['cuda_available'] = False
        env_info['cuda_version'] = 'N/A'
        env_info['gpu_count'] = 0
        env_info['gpu_names'] = []
    except Exception as e:
        env_info['pytorch_version'] = f'加载异常'
        env_info['pytorch_error'] = str(e)[:200]
        env_info['cuda_available'] = False
        env_info['cuda_version'] = 'N/A'
        env_info['gpu_count'] = 0
        env_info['gpu_names'] = []

    # Ultralytics
    try:
        import ultralytics
        env_info['ultralytics_version'] = ultralytics.__version__
    except ImportError:
        env_info['ultralytics_version'] = '未安装'
    except OSError as e:
        env_info['ultralytics_version'] = f'DLL错误: {str(e)[:80]}'

    # OpenCV
    try:
        import cv2
        env_info['opencv_version'] = cv2.__version__
    except ImportError:
        env_info['opencv_version'] = '未安装'
    except OSError as e:
        env_info['opencv_version'] = f'DLL错误: {str(e)[:80]}'

    # VC++ Redist 检测
    env_info['vc_redist'] = _check_vc_redist()

    # NVIDIA驱动 (通过nvidia-smi)
    env_info['nvidia_driver'] = 'N/A'
    env_info['nvidia_smi_cuda'] = 'N/A'
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            env_info['nvidia_driver'] = result.stdout.strip()

        result2 = subprocess.run(
            ['nvidia-smi'],
            capture_output=True, text=True, timeout=10
        )
        if result2.returncode == 0:
            for line in result2.stdout.split('\n'):
                if 'CUDA Version' in line:
                    env_info['nvidia_smi_cuda'] = line.split('CUDA Version:')[-1].strip().split()[0]
                    break
    except Exception:
        pass

    # 磁盘空间
    env_info['disk_free'] = 'N/A'
    env_info['disk_total'] = 'N/A'
    try:
        import shutil
        usage = shutil.disk_usage(os.getcwd())
        free_gb = usage.free / (1024**3)
        total_gb = usage.total / (1024**3)
        env_info['disk_free'] = f'{free_gb:.1f} GB'
        env_info['disk_total'] = f'{total_gb:.1f} GB'
    except Exception:
        pass

    return env_info


def _analyze_dll_error(error_msg):
    """分析DLL加载错误并给出修复建议"""
    if 'c10.dll' in error_msg:
        return (
            "PyTorch DLL初始化失败 (c10.dll)。可能原因:\n"
            "1. Visual C++ Redistributable 未安装或版本过旧\n"
            "2. Python版本与PyTorch版本不兼容 (Python 3.13需PyTorch>=2.5)\n"
            "3. 系统中存在多个不兼容的DLL\n"
            "修复建议: 运行 'pip uninstall torch -y && pip install torch torchvision'"
        )
    elif 'torch' in error_msg.lower():
        return (
            f"PyTorch DLL加载失败。\n"
            f"建议: pip uninstall torch -y && pip install torch torchvision"
        )
    else:
        return f"DLL加载失败: {error_msg[:150]}"


def _check_vc_redist():
    """检测Visual C++ Redistributable是否安装"""
    vc_keys = [
        r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
        r"SOFTWARE\WOW6432Node\Microsoft\VisualStudio\14.0\VC\Runtimes\x64",
    ]
    try:
        import winreg
        for key_path in vc_keys:
            for root in [winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER]:
                try:
                    key = winreg.OpenKey(root, key_path)
                    installed = winreg.QueryValueEx(key, "Installed")
                    winreg.CloseKey(key)
                    if installed[0] == 1:
                        return "已安装 (VS2015-2022)"
                except OSError:
                    continue
        return "未检测到 (建议安装VC++ 2015-2022 Redist)"
    except Exception:
        return "无法检测"


def get_pip_mirrors():
    """返回常用pip镜像源列表"""
    return {
        '默认 (PyPI官方)': 'https://pypi.org/simple/',
        '清华大学 (TUNA)': 'https://pypi.tuna.tsinghua.edu.cn/simple/',
        '阿里云': 'https://mirrors.aliyun.com/pypi/simple/',
        '中科大 (USTC)': 'https://pypi.mirrors.ustc.edu.cn/simple/',
        '豆瓣 (Douban)': 'https://pypi.douban.com/simple/',
        '华为云': 'https://repo.huaweicloud.com/repository/pypi/simple/',
        '腾讯云': 'https://mirrors.cloud.tencent.com/pypi/simple/',
    }


class TrainingManager:
    """模型训练管理器"""

    def __init__(self):
        self.is_training = False
        self.stop_requested = False
        self.training_thread = None
        self.progress_callback = None
        self.log_callback = None
        self.metrics_callback = None
        self.current_model = None
        self.env_info = None
        self.epoch_metrics = []
        self.best_metrics = {}

    def set_callbacks(self, progress_callback=None, log_callback=None, metrics_callback=None):
        self.progress_callback = progress_callback
        self.log_callback = log_callback
        self.metrics_callback = metrics_callback

    def detect_env(self):
        """检测环境"""
        self._emit_log("正在检测运行环境...")
        _preload_torch_dll()
        self.env_info = detect_environment()
        self._emit_log("环境检测完成")
        return self.env_info

    def fix_pytorch(self, mirror_url=''):
        """强制重装修复PyTorch"""
        self._emit_log("=== 强制修复 PyTorch ===")

        # 1. 卸载
        self._emit_log("卸载现有 PyTorch...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', 'torch', 'torchvision',
                 'torchaudio', '-y'],
                capture_output=True, text=True, timeout=30
            )
        except Exception as e:
            self._emit_log(f"卸载警告: {e}")

        # 2. 清理pip缓存
        self._emit_log("清理 pip 缓存...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'cache', 'purge'],
                capture_output=True, text=True, timeout=30
            )
        except Exception:
            pass

        # 3. 卸载可能冲突的Intel OpenMP (PyTorch自带libiomp5md.dll)
        self._emit_log("移除可能冲突的 Intel OpenMP...")
        try:
            subprocess.run(
                [sys.executable, '-m', 'pip', 'uninstall', 'intel-openmp',
                 'intel-cmplr-lib-ur', 'umf', 'tcmlib', '-y'],
                capture_output=True, text=True, timeout=30
            )
        except Exception:
            pass

        # 4. 重装PyTorch
        self._emit_log("重新安装 PyTorch (--force-reinstall)...")
        try:
            pip_args = [sys.executable, '-m', 'pip', 'install',
                       'torch', 'torchvision', 'torchaudio',
                       '--force-reinstall', '--no-cache-dir']

            if mirror_url and 'pypi.org' not in mirror_url:
                pip_args += ['-i', mirror_url, '--trusted-host',
                           mirror_url.split('://')[1].split('/')[0]]

            pip_args += ['--index-url', 'https://download.pytorch.org/whl/cpu']

            self._emit_log(f"执行重装 (可能需要几分钟)...")
            result = subprocess.run(pip_args, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                self._emit_log("PyTorch 重装成功!")
                return True, "PyTorch 修复成功"
            else:
                err = result.stderr[-300:] if result.stderr else '未知错误'
                self._emit_log(f"重装失败: {err}")
                return False, f"修复失败"
        except Exception as e:
            return False, str(e)

    def install_pytorch(self, mirror_url='', cuda_version='cpu'):
        """安装PyTorch"""
        # 先检查是否已安装且能正常加载
        try:
            import torch
            self._emit_log(f"PyTorch 已安装: {torch.__version__}")
            try:
                torch.cuda.is_available()
            except Exception:
                pass
            return True, f"PyTorch 已安装 (版本 {torch.__version__})"
        except (ImportError, OSError):
            pass

        py_ver = sys.version_info
        self._emit_log(f"当前 Python: {py_ver.major}.{py_ver.minor}.{py_ver.micro}")

        # Python 3.13+ 可能需要预发布版本
        if py_ver.minor >= 13:
            self._emit_log("Python 3.13+ 检测到，将使用 --pre 安装预发布版 PyTorch")
            pre_flag = True
        else:
            pre_flag = False

        self._emit_log("正在安装 PyTorch (可能需要几分钟)...")
        try:
            pip_args = [sys.executable, '-m', 'pip', 'install', '--upgrade']
            if pre_flag:
                pip_args.append('--pre')

            if mirror_url and 'pypi.org' not in mirror_url:
                pip_args += ['-i', mirror_url, '--trusted-host',
                           mirror_url.split('://')[1].split('/')[0]]

            pip_args += ['torch', 'torchvision', 'torchaudio']

            # CUDA版本索引
            if cuda_version == 'cu121':
                pip_args += ['--index-url', 'https://download.pytorch.org/whl/cu121']
            elif cuda_version == 'cu118':
                pip_args += ['--index-url', 'https://download.pytorch.org/whl/cu118']

            self._emit_log(f"执行: {' '.join(pip_args)}")
            result = subprocess.run(pip_args, capture_output=True, text=True, timeout=600)

            if result.returncode == 0:
                self._emit_log("PyTorch 安装成功! 请重启应用以生效。")
                return True, "PyTorch 安装成功 (请重启应用)"
            else:
                err = result.stderr.split('\n')[-5:] if result.stderr else ['未知错误']
                err_str = '\n'.join(err)
                self._emit_log(f"安装失败:\n{err_str}")

                if pre_flag:
                    self._emit_log("提示: Python 3.13 兼容性仍在完善中，可考虑使用 Python 3.11/3.12")
                return False, f"安装失败: {err_str[:200]}"
        except Exception as e:
            return False, str(e)

    def install_ultralytics(self, mirror_url=''):
        """安装Ultralytics"""
        try:
            import ultralytics
            self._emit_log(f"Ultralytics 已安装: {ultralytics.__version__}")
            return True, f"Ultralytics 已安装"
        except ImportError:
            pass

        try:
            pip_args = [sys.executable, '-m', 'pip', 'install', 'ultralytics']
            if mirror_url:
                pip_args += ['-i', mirror_url, '--trusted-host', mirror_url.split('://')[1].split('/')[0]]

            self._emit_log("正在安装 Ultralytics...")
            result = subprocess.run(pip_args, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                self._emit_log("Ultralytics 安装成功!")
                return True, "Ultralytics 安装成功"
            else:
                return False, result.stderr[:200]
        except Exception as e:
            return False, str(e)

    def start_training(self, params):
        """开始训练"""
        if self.is_training:
            return False

        self.is_training = True
        self.stop_requested = False
        self.epoch_metrics = []  # 每轮指标记录
        self.best_metrics = {}   # 最佳值追踪

        self.training_thread = threading.Thread(
            target=self._run_training, args=(params,), daemon=True
        )
        self.training_thread.start()
        return True

    def stop_training(self):
        """停止训练"""
        self.stop_requested = True
        self.is_training = False

    def get_training_metrics(self):
        """返回训练指标数据"""
        return {
            'epoch_metrics': self.epoch_metrics,
            'best_metrics': self.best_metrics,
        }

    def _on_train_epoch_end(self, trainer):
        """训练每轮结束回调"""
        try:
            epoch = trainer.epoch + 1
            total = trainer.epochs

            # 获取损失值
            loss_items = getattr(trainer, 'loss_items', None)
            if loss_items is not None:
                box_loss = float(loss_items[0]) if len(loss_items) > 0 else 0
                cls_loss = float(loss_items[1]) if len(loss_items) > 1 else 0
                dfl_loss = float(loss_items[2]) if len(loss_items) > 2 else 0
            else:
                box_loss = cls_loss = dfl_loss = 0

            # 获取评估指标
            metrics = getattr(trainer, 'metrics', {}) or {}
            map50 = float(metrics.get('metrics/mAP50(B)', 0))
            map50_95 = float(metrics.get('metrics/mAP50-95(B)', 0))

            # 更新最佳值
            if map50 > self.best_metrics.get('mAP50', 0):
                self.best_metrics['mAP50'] = map50
            if map50_95 > self.best_metrics.get('mAP50-95', 0):
                self.best_metrics['mAP50-95'] = map50_95
            if box_loss > 0 and box_loss < self.best_metrics.get('box_loss', float('inf')):
                self.best_metrics['box_loss'] = box_loss
            if cls_loss > 0 and cls_loss < self.best_metrics.get('cls_loss', float('inf')):
                self.best_metrics['cls_loss'] = cls_loss
            if dfl_loss > 0 and dfl_loss < self.best_metrics.get('dfl_loss', float('inf')):
                self.best_metrics['dfl_loss'] = dfl_loss

            # 记录本轮指标
            epoch_data = {
                'epoch': epoch,
                'box_loss': box_loss,
                'cls_loss': cls_loss,
                'dfl_loss': dfl_loss,
                'mAP50': map50,
                'mAP50-95': map50_95,
            }
            self.epoch_metrics.append(epoch_data)

            # 发送到UI
            self._emit_progress(epoch, total, box_loss + cls_loss + dfl_loss)
            self._emit_log(
                f"Epoch {epoch}/{total} | "
                f"box={box_loss:.4f} cls={cls_loss:.4f} dfl={dfl_loss:.4f} | "
                f"mAP50={map50:.4f} mAP50-95={map50_95:.4f}"
            )
            self._emit_metrics(epoch_data)

            # 检查停止请求
            if self.stop_requested:
                trainer.stop_training = True

        except Exception as e:
            self._emit_log(f"回调错误: {e}")

    def _run_training(self, params):
        """执行训练过程"""
        try:
            from ultralytics import YOLO
        except ImportError:
            self._emit_log("错误: 未安装ultralytics库，请先安装: pip install ultralytics")
            self.is_training = False
            return

        try:
            model_name = params.get('model', 'yolov8n')
            data_yaml = params.get('data_yaml', '')
            epochs = params.get('epochs', 100)
            batch_size = params.get('batch_size', 16)
            img_size = params.get('img_size', 640)
            lr = params.get('learning_rate', 0.01)
            device = params.get('device', 'auto')
            workers = params.get('workers', 4)
            pretrained = params.get('pretrained', True)

            self._emit_log(f"开始训练: {model_name}")
            self._emit_log(f"数据集: {data_yaml}")
            self._emit_log(f"Epochs: {epochs} | Batch: {batch_size} | ImgSize: {img_size} | LR: {lr}")
            self._emit_log(f"设备: {device} | Workers: {workers}")

            if pretrained:
                model = YOLO(f"{model_name}.pt")
                self._emit_log("使用预训练权重")
            else:
                model = YOLO(f"{model_name}.yaml")
                self._emit_log("从头开始训练")

            self.current_model = model

            # 注册回调
            model.add_callback("on_train_epoch_end", self._on_train_epoch_end)

            model.train(
                data=data_yaml,
                epochs=epochs,
                batch=batch_size,
                imgsz=img_size,
                lr0=lr,
                device=device,
                workers=workers,
                verbose=False,
                exist_ok=True
            )

            save_dir = params.get('save_dir', 'runs/train')
            self._emit_log(f"训练完成! 模型保存至: {save_dir}")
            self._emit_progress(100, 100, 0)
            self._emit_log(f"最佳指标: mAP50={self.best_metrics.get('mAP50',0):.4f} "
                          f"mAP50-95={self.best_metrics.get('mAP50-95',0):.4f}")

        except Exception as e:
            self._emit_log(f"训练出错: {str(e)}")
        finally:
            self.is_training = False

    def load_model(self, model_path):
        """加载模型"""
        try:
            from ultralytics import YOLO
            self.current_model = YOLO(model_path)
            return True
        except Exception:
            return False

    def get_model_info(self):
        """获取模型信息"""
        if self.current_model is None:
            return None
        return {
            'task': getattr(self.current_model.model, 'task', 'detect'),
        }

    def _emit_progress(self, current, total, loss):
        if self.progress_callback:
            self.progress_callback(current, total, loss)

    def _emit_log(self, message):
        if self.log_callback:
            self.log_callback(message)

    def _emit_metrics(self, metrics):
        if self.metrics_callback:
            self.metrics_callback(metrics)
