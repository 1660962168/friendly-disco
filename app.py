import webview
import threading
import sys
import os
import glob
from flask import Flask, render_template, jsonify, request, url_for
import psutil
import GPUtil
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import pathlib
import re
import config
from exts import db
from models import *
from flask_migrate import Migrate
from paddleocr import PaddleOCR
import traceback

# --- 1. 全局配置与初始化 ---

# 初始化 PaddleOCR
# use_textline_orientation=False: 关闭方向检测，防止车牌被误判旋转
# lang="ch": 支持中英文
ocr_engine = None 

def get_ocr_model():
    """ 懒加载 OCR 模型单例模式 """
    global ocr_engine
    if ocr_engine is None:
        print("正在初始化 OCR 模型 (第一次运行会稍慢)...")
        # 关掉 show_log 防止报错，关掉方向检测
        ocr_engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False)
    return ocr_engine

def get_resource_path(relative_path):
    """ 获取资源绝对路径（兼容 PyInstaller 打包） """
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

app = Flask(__name__)
app.config.from_object(config)

# 初始化数据库
db.init_app(app)
migrate = Migrate(app, db)

# 路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'weights', 'best.pt')
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
RESULTS_FOLDER = os.path.join(BASE_DIR, 'static', 'results')
CROPS_FOLDER = os.path.join(BASE_DIR, 'static', 'crops')

# 确保文件夹存在
for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, CROPS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- 2. 辅助函数 ---

def cleanup_old_files(folder_path, limit=30):
    """ 清理旧文件，防止磁盘占满 """
    try:
        files = glob.glob(os.path.join(folder_path, "*"))
        files = [f for f in files if os.path.isfile(f)]
        if len(files) > limit:
            files.sort(key=os.path.getmtime) # 按时间排序
            num_to_delete = len(files) - limit
            for i in range(num_to_delete):
                try:
                    os.remove(files[i])
                except Exception:
                    pass
    except Exception as e:
        print(f"清理文件夹出错 {folder_path}: {e}")

# 加载 YOLO 模型
try:
    print(f"正在加载模型: {MODEL_PATH}")
    # [Windows 路径兼容修复]
    if os.name == 'nt':
        pathlib.PosixPath = pathlib.WindowsPath
    model = YOLO(MODEL_PATH)
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败 (请确保 weights/best.pt 存在): {e}")
    model = None

import os
import uuid
import cv2
from flask import jsonify, request, url_for

# 1. 定义全局变量 (在所有路由之外)
# 这里的变量可以被其他路由访问
CURRENT_DETECTED_CLASS = None 

@app.route('/api/detect/yolo', methods=['POST'])
def detect_yolo():
    # 引入全局变量，以便在函数内修改它
    global CURRENT_DETECTED_CLASS
    
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

        # YOLO 推理
        results = model(filepath)
        result = results[0]
        
        annotated_frame = result.plot()
        result_filename = f"result_{safe_filename}"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated_frame)

        boxes = result.boxes.xyxy.cpu().numpy()
        conf_scores = result.boxes.conf.cpu().numpy()
        # 获取类别索引数组
        cls_ids = result.boxes.cls.cpu().numpy() 
        
        has_plate = False
        crop_filename = ""
        yolo_confidence = 0.0
        
        # 重置当前全局变量，防止如果没有检测到物体时保留了上一次的值
        CURRENT_DETECTED_CLASS = None 

        if len(boxes) > 0:
            box = boxes[0]
            yolo_confidence = float(conf_scores[0])
            
            # 2. 获取并设置类型名称
            class_id = int(cls_ids[0])       # 获取第一个框的类别ID (例如 0, 1)
            class_name = result.names[class_id] # 通过ID在names字典中查找名称 (例如 'blue_plate')
            # === 设置全局变量 ===
            CURRENT_DETECTED_CLASS = class_name
            print(f"全局变量已更新: {CURRENT_DETECTED_CLASS}")
            padding_x, padding_y = 10, 10
            x1 = int(box[0]) - padding_x
            y1 = int(box[1]) - padding_y
            x2 = int(box[2]) + padding_x
            y2 = int(box[3]) + padding_y
            
            original_img = cv2.imread(filepath)
            h, w, _ = original_img.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
            plate_crop = original_img[y1:y2, x1:x2]
            
            # 保存裁剪图 (OCR 将读取这个文件)
            crop_filename = f"crop_{safe_filename}"
            crop_path = os.path.join(CROPS_FOLDER, crop_filename)
            cv2.imwrite(crop_path, plate_crop)
            
            has_plate = True

        return jsonify({
            'success': True,
            'original_url': url_for('static', filename=f'uploads/{safe_filename}'),
            'result_url': url_for('static', filename=f'results/{result_filename}'),
            'has_plate': has_plate,
            'detected_type': CURRENT_DETECTED_CLASS, # 将识别到的类型也返回给前端
            'crop_filename': crop_filename if has_plate else None,
            'crop_url': url_for('static', filename=f'crops/{crop_filename}') if has_plate else None,
            'yolo_confidence': yolo_confidence
        })

    except Exception as e:
        print(f"YOLO Error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/detect/ocr', methods=['POST'])
def detect_ocr():
    # 获取前端传来的裁剪文件名
    data = request.json
    crop_filename = data.get('crop_filename')
    
    if not crop_filename:
        return jsonify({'error': 'No crop filename provided'}), 400
        
    crop_path = os.path.join(CROPS_FOLDER, crop_filename)
    if not os.path.exists(crop_path):
        return jsonify({'error': 'Crop file not found'}), 404

    plate_text = ""
    plate_confidence = 0.0

    try:
        # [优化] 获取懒加载的 OCR 模型
        

        # 读取图片
        plate_crop = cv2.imread(crop_path)
        
        # === 之前的 OCR 增强逻辑 (放大/灰度/反色) ===
        roi = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_inv = cv2.bitwise_not(gray)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_inv_bgr = cv2.cvtColor(gray_inv, cv2.COLOR_GRAY2BGR)
        
        PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
        candidates = []

        # 双向识别
        for img_input, label in [(gray_bgr, "Normal"), (gray_inv_bgr, "Inverted")]:
            res = ocr.predict(img_input)
            
            text_res = ""
            score_res = 0.0
            
            if res:
                if isinstance(res, list) and len(res) > 0:
                    item = res[0]
                    if isinstance(item, dict) and 'rec_texts' in item and len(item['rec_texts']) > 0:
                        text_res = item['rec_texts'][0]
                        score_res = float(item['rec_scores'][0])
                    elif isinstance(item, list) and len(item) >= 2:
                        text_res = item[1][0]
                        score_res = float(item[1][1])
            
            if text_res:
                clean_text = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text_res)
                is_valid = False
                current_score = score_res
                if len(clean_text) >= 2 and clean_text[0] in PROVINCES:
                    is_valid = True
                    current_score += 0.5
                if label == "Inverted" and is_valid:
                    current_score += 0.2
                
                if len(clean_text) > 1:
                    candidates.append({'text': clean_text, 'score': current_score, 'valid': is_valid, 'mode': label})

        if candidates:
            candidates.sort(key=lambda x: (x['valid'], x['score']), reverse=True)
            best = candidates[0]
            plate_text = best['text']
            plate_confidence = min(0.99, best['score'])
            print(f"OCR Success: {plate_text}")
            global CURRENT_DETECTED_CLASS
            plate_type = CURRENT_DETECTED_CLASS
            if len(plate_text) != 7 and len(plate_text) != 8:
                print("OCR Detected text length is invalid")
                record = LicensePlate(
                    plate_text=plate_text,
                    type=plate_type,
                    status=0,
                    )
            else:
                print("OCR Detected text length is valid")
                record = LicensePlate(
                    plate_text=plate_text,
                    type=plate_type,
                    status=1,
                    )
            db.session.add(record)
            db.session.commit()
        else:
            print("OCR Failed to find valid text")

        return jsonify({
            'success': True,
            'plate_text': plate_text,
            'confidence': plate_confidence
        })

    except Exception as e:
        print(f"OCR Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# --- 4. 辅助路由 ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/performance')
def performance():
    # CPU
    cpu_usage = psutil.cpu_percent(interval=None)
    # 内存
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    # GPU
    gpu_usage = 0
    gpu_name = "No GPU"
    try:
        gpus = GPUtil.getGPUs()
        if gpus:
            gpu = gpus[0]
            gpu_usage = gpu.load * 100
            gpu_name = gpu.name
    except:
        pass

    return jsonify({
        'cpu_usage': round(cpu_usage, 1),
        'memory_usage': round(memory_usage, 1),
        'gpu_usage': round(gpu_usage, 1),
        'gpu_name': gpu_name
    })

# --- 5. 启动程序 ---

def start_flask():
    app.run(host='127.0.0.1', port=5000, threaded=True, use_reloader=False)

if __name__ == '__main__':
    # 尝试自动创建数据库表 (如果 models.py 定义了表结构)
    with app.app_context():
        try:
            db.create_all()
            print("数据库表检查完成。")
        except Exception as e:
            print(f"数据库警告: {e}")

    # 启动 Flask 线程
    t = threading.Thread(target=start_flask)
    t.daemon = True
    t.start()
    ocr = get_ocr_model()
    # 启动 GUI 窗口
    icon_path = get_resource_path('apple-touch-icon.png')
    window = webview.create_window(
        title="车牌识别系统",
        url='http://127.0.0.1:5000',
        width=1480, 
        height=900,
        resizable=True,
        confirm_close=True,
    )
    webview.start(icon=icon_path)