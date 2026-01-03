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
import time  # [新增] 导入 time 模块
from exts import db
from models import *
from flask_migrate import Migrate
from paddleocr import PaddleOCR
import traceback
from sqlalchemy import func

# --- 1. 全局配置与初始化 ---

# 初始化 PaddleOCR
ocr_engine = None 

def get_ocr_model():
    """ 懒加载 OCR 模型单例模式 """
    global ocr_engine
    if ocr_engine is None:
        print("正在初始化 OCR 模型 (第一次运行会稍慢)...")
        ocr_engine = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False)
    return ocr_engine

def get_resource_path(relative_path):
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

for folder in [UPLOAD_FOLDER, RESULTS_FOLDER, CROPS_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# --- 2. 辅助函数 ---

def cleanup_old_files(folder_path, limit=30):
    try:
        files = glob.glob(os.path.join(folder_path, "*"))
        files = [f for f in files if os.path.isfile(f)]
        if len(files) > limit:
            files.sort(key=os.path.getmtime)
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
    if os.name == 'nt':
        pathlib.PosixPath = pathlib.WindowsPath
    model = YOLO(MODEL_PATH)
    print("模型加载成功!")
except Exception as e:
    print(f"模型加载失败 (请确保 weights/best.pt 存在): {e}")
    model = None

# 定义全局变量
CURRENT_DETECTED_CLASS = None 

@app.route('/api/detect/yolo', methods=['POST'])
def detect_yolo():
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

        results = model(filepath)
        result = results[0]
        
        annotated_frame = result.plot()
        result_filename = f"result_{safe_filename}"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated_frame)

        boxes = result.boxes.xyxy.cpu().numpy()
        conf_scores = result.boxes.conf.cpu().numpy()
        cls_ids = result.boxes.cls.cpu().numpy() 
        
        has_plate = False
        crop_filename = ""
        yolo_confidence = 0.0
        
        CURRENT_DETECTED_CLASS = None 

        if len(boxes) > 0:
            box = boxes[0]
            yolo_confidence = float(conf_scores[0])
            
            class_id = int(cls_ids[0])
            class_name = result.names[class_id]
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
            
            crop_filename = f"crop_{safe_filename}"
            crop_path = os.path.join(CROPS_FOLDER, crop_filename)
            cv2.imwrite(crop_path, plate_crop)
            
            has_plate = True

        return jsonify({
            'success': True,
            'original_url': url_for('static', filename=f'uploads/{safe_filename}'),
            'result_url': url_for('static', filename=f'results/{result_filename}'),
            'has_plate': has_plate,
            'detected_type': CURRENT_DETECTED_CLASS,
            'crop_filename': crop_filename if has_plate else None,
            'crop_url': url_for('static', filename=f'crops/{crop_filename}') if has_plate else None,
            'yolo_confidence': yolo_confidence
        })

    except Exception as e:
        print(f"YOLO Error: {e}")
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/detect/ocr', methods=['POST'])
def detect_ocr():
    # [新增] 开始计时
    start_time = time.time()

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
        # [修改] 修复 OCR 模型调用
        ocr = get_ocr_model()

        plate_crop = cv2.imread(crop_path)
        
        # 图像增强
        roi = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray_inv = cv2.bitwise_not(gray)
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_inv_bgr = cv2.cvtColor(gray_inv, cv2.COLOR_GRAY2BGR)
        
        PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
        candidates = []

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

        # [新增] 结束计时
        end_time = time.time()
        duration = round(end_time - start_time, 2)

        if candidates:
            candidates.sort(key=lambda x: (x['valid'], x['score']), reverse=True)
            best = candidates[0]
            plate_text = best['text']
            plate_confidence = min(0.99, best['score'])
            
            print(f"OCR Success: {plate_text}")
            global CURRENT_DETECTED_CLASS
            plate_type = CURRENT_DETECTED_CLASS if CURRENT_DETECTED_CLASS else "Unknown"
            
            status = 1
            if len(plate_text) != 7 and len(plate_text) != 8:
                 status = 0
            
            # [修改] 存入数据库，包含 duration 和 confidence
            try:
                # 先查询是否存在相同的车牌，如果存在则更新，不存在则插入
                # 为了演示简单，这里直接插入，如果报错(Unique constraint)则忽略或打印
                record = LicensePlate(
                    plate_text=plate_text,
                    type=plate_type,
                    status=status,
                    confidence=plate_confidence,
                    duration=duration
                )
                db.session.add(record)
                db.session.commit()
            except Exception as db_e:
                print(f"Database Save Error (Might be duplicate): {db_e}")
                db.session.rollback()

        else:
            print("OCR Failed to find valid text")

        return jsonify({
            'success': True,
            'plate_text': plate_text,
            'confidence': plate_confidence,
            'duration': duration  # [新增] 返回用时
        })

    except Exception as e:
        print(f"OCR Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- [新增] 统计与渲染专用路由 ---

@app.route('/api/stats/trend')
def stats_trend():
    """ 24小时识别趋势 """
    # 获取所有的记录的时间
    # 既然要做24小时趋势，这里简单起见统计所有历史记录的小时分布
    # 如果要仅限“今日”，可以在 query 中加 filter
    
    # 1. 查询所有记录的时间
    records = db.session.query(LicensePlate.time).all()
    
    # 2. Python 处理统计
    hours_count = {i: 0 for i in range(24)}
    
    for r in records:
        if r.time:
            h = r.time.hour
            hours_count[h] += 1
            
    return jsonify({
        'labels': [f"{i:02d}:00" for i in range(24)],
        'data': [hours_count[i] for i in range(24)]
    })

@app.route('/api/stats/distribution')
def stats_distribution():
    """ 车牌类型分布 """
    # 按类型分组计数
    results = db.session.query(
        LicensePlate.type, 
        func.count(LicensePlate.id)
    ).group_by(LicensePlate.type).all()
    
    # 映射表
    label_map = {
        'BlueCard': '蓝牌',
        'GreenCard': '绿牌',
        'YellowCard': '黄牌',
        'Unknown': '未知'
    }
    color_map = {
        'BlueCard': '#3b82f6',   # blue-500
        'GreenCard': '#10b981',  # emerald-500
        'YellowCard': '#eab308', # yellow-500
        'Unknown': '#9ca3af'     # gray-400
    }
    
    labels = []
    data = []
    colors = []
    
    for type_name, count in results:
        labels.append(label_map.get(type_name, type_name))
        data.append(count)
        colors.append(color_map.get(type_name, '#9ca3af'))
        
    return jsonify({
        'labels': labels,
        'data': data,
        'colors': colors
    })

@app.route('/api/stats/history')
def stats_history():
    """ 历史记录 (最新的20条) """
    # 按时间倒序查询
    logs = LicensePlate.query.order_by(LicensePlate.time.desc()).limit(20).all()
    
    history_data = []
    for log in logs:
        history_data.append({
            'id': log.id,
            'time': log.time.strftime('%H:%M:%S'),
            'plate': log.plate_text,
            'status': 'success' if log.status == 1 else 'failed',
            'confidence': round(log.confidence * 100, 1),
            'location': '默认入口', # 数据库未存地点，暂用默认
            'duration': log.duration
        })
        
    # 同时计算一下总计数据
    total_count = LicensePlate.query.count()
    success_count = LicensePlate.query.filter_by(status=1).count()
    failed_count = LicensePlate.query.filter_by(status=0).count()
    
    return jsonify({
        'logs': history_data,
        'stats': {
            'total': total_count,
            'success': success_count,
            'failed': failed_count
        }
    })

# --- 4. 辅助路由 ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/performance')
def performance():
    cpu_usage = psutil.cpu_percent(interval=None)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
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
    with app.app_context():
        try:
            db.create_all()
            print("数据库表检查完成。")
        except Exception as e:
            print(f"数据库警告: {e}")

    t = threading.Thread(target=start_flask)
    t.daemon = True
    t.start()
    
    # 预加载 OCR
    get_ocr_model()
    
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