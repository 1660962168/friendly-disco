import sys
import os
from cx_Freeze import setup, Executable

# 1. 定义要包含的文件
# 格式：("源文件路径", "打包后的路径")
# 这里的 "index.html" 会被复制到打包后的文件夹根目录
include_files = [
    ("index.html", "index.html") 
    # 如果有其他图片文件夹，继续加： ("assets/", "assets/")
]

# 2. 构建选项
build_exe_options = {
    "packages": ["os", "webview", "threading", "time"], # 需要强制包含的包
    "excludes": [],
    "include_files": include_files,
}

# 3. 设置 GUI 模式（去除黑色黑窗口）
base = None
if sys.platform == "win32":
    base = "Win32GUI"  # 如果你想看报错信息调试，可以先把这就改成 None

# 4. 配置主入口
setup(
    name = "车牌识别系统",
    version = "1.0",
    description = "基于 Pywebview 的车牌识别应用",
    options = {"build_exe": build_exe_options},
    executables = [Executable("app.py", base=base, target_name="车牌系统.exe")]
)