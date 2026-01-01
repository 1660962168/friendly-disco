import webview
import threading
import sys
import os
from flask import Flask, render_template, jsonify, request, url_for
import psutil
import GPUtil
from ultralytics import YOLO
import cv2 # 新增
import numpy as np # 新增
import uuid # 用于生成唯一文件名
import pathlib
import glob
import config
from exts import db
from models import *
from flask_migrate import Migrate
from paddleocr import PaddleOCR



ocr = PaddleOCR(use_textline_orientation=True, lang="ch")

def get_resource_path(relative_path):
    """ 获取资源绝对路径（兼容 PyInstaller 打包） """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

app = Flask(__name__)
app.config.from_object(config)
db.init_app(app)
migrate = Migrate(app, db)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'best.pt')

# 确保必要的文件夹存在
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
CROPS_FOLDER = os.path.join(BASE_DIR, 'static', 'crops') # 专门存放车牌截图

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, CROPS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def cleanup_old_files(folder_path, limit=30):
    try:
        # 获取文件夹内所有文件 (按完整路径)
        files = glob.glob(os.path.join(folder_path, "*"))
        # 过滤掉文件夹，只保留文件
        files = [f for f in files if os.path.isfile(f)]
        if len(files) > limit:
            # 按修改时间排序：旧在前，新在后
            files.sort(key=os.path.getmtime)
            # 需要删除的文件数量
            num_to_delete = len(files) - limit
            for i in range(num_to_delete):
                try:
                    os.remove(files[i])
                    print(f"已清理旧文件: {files[i]}")
                except Exception as e:
                    print(f"删除文件失败 {files[i]}: {e}")
    except Exception as e:
        print(f"清理文件夹出错 {folder_path}: {e}")

# 加载模型 (增加异常处理防止没模型时报错)
try:
    print(f"正在加载模型: {MODEL_PATH}")
    
    # <--- [新增 2]在此处加入跨平台修复代码 --->
    if os.name == 'nt':  # 如果是 Windows 系统
        pathlib.PosixPath = pathlib.WindowsPath
    # <---------------------------------------->

    model = YOLO(MODEL_PATH)
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败 (请确保 weights/best.pt 存在): {e}")
    model = None

@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    if not model:
        return jsonify({'error': 'Model not loaded'}), 500
        
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        cleanup_old_files(UPLOAD_FOLDER, limit=30)
        cleanup_old_files(RESULTS_FOLDER, limit=30)
        cleanup_old_files(CROPS_FOLDER, limit=30)
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".jpg"
        safe_filename = f"{uuid.uuid4().hex}{ext}" 
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)
        results = model(filepath)
        result = results[0]
        annotated_frame = result.plot()
        result_filename = f"result_{safe_filename}"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated_frame)
        boxes = result.boxes.xyxy.cpu().numpy()
        conf_scores = result.boxes.conf.cpu().numpy()
        
        has_plate = False
        crop_filename = ""
        confidence = 0.0
        if len(boxes) > 0:
            box = boxes[0]
            confidence = float(conf_scores[0])
            x1, y1, x2, y2 = map(int, box)
            original_img = cv2.imread(filepath)
            h, w, _ = original_img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            plate_crop = original_img[y1:y2, x1:x2]
            crop_filename = f"crop_{safe_filename}"
            crop_path = os.path.join(CROPS_FOLDER, crop_filename)
            cv2.imwrite(crop_path, plate_crop)
            temp_crop_path = os.path.join(CROPS_FOLDER, 'temp_plate.jpg')
            cv2.imwrite(temp_crop_path, plate_crop)
            has_plate = True
        plate_text = ""
        plate_confidence = 0.0
    
        if has_plate:
            try:
                # 使用 PaddleOCR 识别刚才裁剪下来的车牌图片 (temp_plate.jpg)
                # 这里的 temp_crop_path 是你之前代码里保存的 'static/crops/temp_plate.jpg'
                # 或者直接传 numpy 数组: ocr_result = ocr.ocr(plate_crop, cls=True)
            
                ocr_result = ocr.ocr(temp_crop_path, cls=True)
            
                # PaddleOCR 返回的结构比较复杂，通常是 [[[[x1,y1],...], ("识别文本", 置信度)]]
                # 我们需要解析它。对于车牌，通常只有一行文本。
                if ocr_result and ocr_result[0]:
                    # 取第一行识别结果
                    res = ocr_result[0][0] 
                    # res[1][0] 是文本，res[1][1] 是置信度
                    plate_text = res[1][0]
                    plate_confidence = float(res[1][1])
                    print(f"OCR 识别结果: {plate_text}, 置信度: {plate_confidence}")
            except Exception as e:
                print(f"OCR 识别出错: {e}")

        return jsonify({
            'success': True,
            'original_url': url_for('static', filename=f'uploads/{safe_filename}'),
            'result_url': url_for('static', filename=f'results/{result_filename}'),
            'has_plate': has_plate,
            'confidence': confidence, # 返回置信度 (0.0 - 1.0)
            'crop_url': url_for('static', filename=f'crops/{crop_filename}') if has_plate else None,
            'plate_text': plate_text,      # 新增：返回识别到的车牌号
            'confidence': plate_confidence
        })

    except Exception as e:
        print(f"Error during detection: {e}")
        return jsonify({'error': str(e)}), 500

# --- Flask 路由 ---
@app.route('/')
def index():
    # 这里不再直接读取文件，而是通过 render_template
    # 你的 index.html 应该放在当前目录下（因为上面设置了 template_folder='.'）
    return render_template('index.html')

@app.route('/api/performance')
def performance():
    # 1. 获取 CPU 占用率
    # interval=None 表示非阻塞模式，返回自上次调用以来的平均值。
    # 如果是第一次调用，可能会返回 0.0。
    # 为了更准确的瞬时值，可以使用 interval=0.1，但这会使 API 响应慢 0.1 秒
    cpu_usage = psutil.cpu_percent(interval=1)
    
    # 2. 获取 内存 占用率
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent

    # 3. 获取 GPU 占用率 (主要针对 NVIDIA)
    gpu_usage = 0
    gpu_name = "No GPU detected"
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            # 假设只获取第一张显卡的数据
            gpu = gpus[0]
            gpu_usage = gpu.load * 100  # GPUtil 返回的是 0.0-1.0，需要乘 100
            gpu_name = gpu.name
    except Exception as e:
        # 如果没有安装驱动或不是 NVIDIA 显卡，防止程序崩溃
        print(f"GPU Error: {e}")
        gpu_usage = 0

    data = {
        'cpu_usage': round(cpu_usage, 1),
        'memory_usage': round(memory_usage, 1),
        'gpu_usage': round(gpu_usage, 1),
        'gpu_name': gpu_name  #以此确认是否识别到了正确的显卡
    }
    
    return jsonify(data)

# --- 2. 启动 Flask 的函数 ---
def start_flask():
    # use_reloader=False 防止 Flask 在调试模式下启动两次导致报错
    app.run(host='127.0.0.1', port=5000, threaded=True, use_reloader=False,debug=True)

# --- 3. 主程序 ---
if __name__ == '__main__':
    # A. 在子线程启动 Flask
    t = threading.Thread(target=start_flask)
    t.daemon = True
    t.start()
    window = webview.create_window(
        title="车牌识别系统",
        url='http://127.0.0.1:5000',  # <--- 关键点：这里是 URL，不是文件路径
        width=1480, 
        height=900,
        maximized=False,
        vibrancy=True,
        shadow=True,
        resizable=False,
        confirm_close=True,
    )
    icon_path = get_resource_path('apple-touch-icon.png')
    # C. 启动 GUI
    webview.start(gui='edgechromium',icon=icon_path)