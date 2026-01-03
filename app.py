import webview
import threading
import sys
import os

# --- [优化] 跳过 PaddleOCR 的联网检查，极大加快启动速度 ---
os.environ['DISABLE_MODEL_SOURCE_CHECK'] = 'True' 

import glob
from flask import Flask, render_template, jsonify, request, url_for, Response, stream_with_context, make_response
import psutil
import GPUtil
from ultralytics import YOLO
import cv2
import numpy as np
import uuid
import pathlib
import re
import config
import time
import csv
import io
from datetime import datetime
from exts import db
from models import *
from flask_migrate import Migrate
from paddleocr import PaddleOCR
import traceback
from sqlalchemy import func

# --- 1. 全局配置与初始化 ---

ocr_engine = None 

def get_ocr_model():
    """ 懒加载 OCR 模型单例模式 """
    global ocr_engine
    if ocr_engine is None:
        print("正在初始化 OCR 模型 (第一次运行会稍慢)...")
        # 显式指定模型路径或名称，确保 det 和 rec 都加载
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

# --- 3. 核心 API 路由 ---

@app.route('/api/init/ocr')
def init_ocr():
    try:
        print("正在后台预加载 OCR 模型...")
        get_ocr_model() 
        print("OCR 模型预加载完成！")
        return jsonify({'success': True})
    except Exception as e:
        print(f"预加载失败: {e}")
        return jsonify({'error': str(e)}), 500

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

        conf_threshold = 0.25
        try:
            sys_config = SystemConfig.query.first()
            if sys_config:
                conf_threshold = sys_config.confidence_threshold / 100.0
                print(f"使用置信度阈值: {conf_threshold}")
        except:
            pass

        results = model(filepath, conf=conf_threshold)
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

        # 图片上传模式，背景复杂，使用默认的 ocr.ocr(det=True) 或者 predict
        # 为了兼容性，这里使用 ocr.ocr(det=False) 因为我们已经有 crop 了
        # 但如果是用户手动上传的很乱的 crop，可能需要 det=True
        # 这里保持之前的逻辑（predict 或 ocr），但建议统一用 ocr
        
        for img_input, label in [(gray_bgr, "Normal"), (gray_inv_bgr, "Inverted")]:
            # [修改] 使用标准的 ocr 接口，关闭检测，只做识别
            res = ocr.ocr(img_input, det=False, rec=True, cls=False)
            
            text_res = ""
            score_res = 0.0
            
            # 解析 det=False 的结果: [[('文本', 0.99)]]
            if res and isinstance(res, list):
                for item in res:
                    if isinstance(item, list) and len(item) >= 1:
                        # item 是 [('text', score)]
                        if isinstance(item[0], tuple) or isinstance(item[0], list):
                            text_res = item[0][0]
                            score_res = float(item[0][1])
            
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
                    candidates.append({'text': clean_text, 'score': current_score, 'valid': is_valid})

        end_time = time.time()
        duration = round(end_time - start_time, 2)

        if candidates:
            candidates.sort(key=lambda x: (x['valid'], x['score']), reverse=True)
            best = candidates[0]
            plate_text = best['text']
            plate_confidence = min(0.99, best['score'])
            
            global CURRENT_DETECTED_CLASS
            plate_type = CURRENT_DETECTED_CLASS if CURRENT_DETECTED_CLASS else "Unknown"
            status = 1 if (len(plate_text) == 7 or len(plate_text) == 8) else 0
            
            # 数据库逻辑
            try:
                existing_record = LicensePlate.query.filter_by(plate_text=plate_text).first()
                if existing_record:
                    existing_record.time = datetime.now()
                    existing_record.type = plate_type
                    existing_record.status = status
                    existing_record.confidence = plate_confidence
                    existing_record.duration = duration
                    db.session.commit()
                else:
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
                print(f"DB Save Error: {db_e}")
                db.session.rollback()

        return jsonify({
            'success': True,
            'plate_text': plate_text,
            'confidence': plate_confidence,
            'duration': duration
        })

    except Exception as e:
        print(f"OCR Error: {e}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

# --- 视频识别模块 (深度修复版) ---

@app.route('/api/detect/video_upload', methods=['POST'])
def detect_video_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    try:
        cleanup_old_files(UPLOAD_FOLDER, limit=10)
        ext = os.path.splitext(file.filename)[1]
        safe_filename = f"video_{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)

        return jsonify({
            'success': True,
            'video_filename': safe_filename,
            'stream_url': url_for('video_feed', filename=safe_filename)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# [核心修复函数] 视频流生成
def generate_video_frames(video_path):
    conf_threshold = 0.25
    with app.app_context():
        try:
            sys_config = SystemConfig.query.first()
            if sys_config:
                conf_threshold = sys_config.confidence_threshold / 100.0
        except:
            pass

    cap = cv2.VideoCapture(video_path)
    ocr = get_ocr_model()
    PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"

    # OCR 辅助函数：处理单张小图
    def recognize_plate(img_crop):
        # 1. 预处理
        roi = cv2.resize(img_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        candidates = []
        
        # 2. 尝试正色 (gray_bgr) 和 反色 (gray_inv_bgr)
        # 视频流为了速度，可以只做一次，但为了准确率，建议做两次尝试
        gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        gray_inv = cv2.bitwise_not(gray)
        gray_inv_bgr = cv2.cvtColor(gray_inv, cv2.COLOR_GRAY2BGR)
        
        for input_img in [gray_bgr, gray_inv_bgr]:
            try:
                # [关键] det=False 强制关闭检测，只做识别
                res = ocr.ocr(input_img, det=False, rec=True, cls=False)
                
                # 解析结果: [[('苏E05EV9', 0.99)]]
                if res and isinstance(res, list):
                    for item in res:
                        if isinstance(item, list) and len(item) >= 1:
                            text_data = item[0] # ('text', score)
                            if isinstance(text_data, tuple) or isinstance(text_data, list):
                                text = text_data[0]
                                score = float(text_data[1])
                                
                                # 简单清洗
                                clean = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text)
                                if len(clean) >= 7 and clean[0] in PROVINCES:
                                    candidates.append((clean, score))
            except Exception:
                pass
        
        if candidates:
            # 返回置信度最高的
            candidates.sort(key=lambda x: x[1], reverse=True)
            return candidates[0] # (text, score)
        
        return None, 0.0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
            
        # YOLO 检测
        results = model(frame, conf=conf_threshold, verbose=False)
        annotated_frame = frame.copy()
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                # 获取坐标
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls[0])
                cls_name = result.names[cls_id]
                
                # 绘制绿色框
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # 裁剪车牌区域
                h, w, _ = frame.shape
                # 稍微扩大一点点，防止贴边
                pad = 5
                crop_x1, crop_y1 = max(0, x1 - pad), max(0, y1 - pad)
                crop_x2, crop_y2 = min(w, x2 + pad), min(h, y2 + pad)
                plate_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                
                # OCR 识别
                text, score = recognize_plate(plate_crop)
                
                if text:
                    # 数据库操作 (在 app context 中)
                    with app.app_context():
                        try:
                            existing = LicensePlate.query.filter_by(plate_text=text).first()
                            if existing:
                                existing.time = datetime.now()
                                existing.confidence = score
                                existing.status = 1
                                db.session.commit()
                            else:
                                new_plate = LicensePlate(
                                    plate_text=text,
                                    type=cls_name,
                                    status=1,
                                    confidence=score,
                                    duration=0.1 # 视频流模拟耗时
                                )
                                db.session.add(new_plate)
                                db.session.commit()
                        except:
                            db.session.rollback()
                    
                    # 绘制文字 (注意：cv2 不支持中文，这里用 ASCII 字符代替或简单绘制)
                    # 为了避免乱码，只绘制 "Plate Found" 或 英文部分，前端列表会显示中文
                    # 这里做一个简单的处理：尝试绘制，如果是乱码也不影响程序运行
                    display_text = f"Conf: {score:.2f}" 
                    cv2.putText(annotated_frame, display_text, (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # 编码图片推流
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed/<filename>')
def video_feed(filename):
    video_path = os.path.join(UPLOAD_FOLDER, filename)
    if not os.path.exists(video_path):
        return "Video not found", 404
    return Response(stream_with_context(generate_video_frames(video_path)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# --- 统计 API (保持不变) ---

@app.route('/api/stats/trend')
def stats_trend():
    records = db.session.query(LicensePlate.time).all()
    hours_count = {i: 0 for i in range(24)}
    for r in records:
        if r.time:
            hours_count[r.time.hour] += 1
    return jsonify({
        'labels': [f"{i:02d}:00" for i in range(24)],
        'data': [hours_count[i] for i in range(24)]
    })

@app.route('/api/stats/distribution')
def stats_distribution():
    results = db.session.query(LicensePlate.type, func.count(LicensePlate.id)).group_by(LicensePlate.type).all()
    label_map = {'BlueCard': '蓝牌', 'GreenCard': '绿牌', 'YellowCard': '黄牌', 'Unknown': '未知'}
    color_map = {'BlueCard': '#3b82f6', 'GreenCard': '#10b981', 'YellowCard': '#eab308', 'Unknown': '#9ca3af'}
    return jsonify({
        'labels': [label_map.get(r[0], r[0]) for r in results],
        'data': [r[1] for r in results],
        'colors': [color_map.get(r[0], '#9ca3af') for r in results]
    })

@app.route('/api/stats/history')
def stats_history():
    search_query = request.args.get('search', '').strip()
    log_query = LicensePlate.query
    if search_query:
        log_query = log_query.filter(LicensePlate.plate_text.contains(search_query))
    
    logs = log_query.order_by(LicensePlate.time.desc()).all()
    
    history_data = []
    for log in logs:
        history_data.append({
            'id': log.id,
            'time': log.time.strftime('%Y-%m-%d %H:%M:%S'),
            'plate': log.plate_text,
            'status': 'success' if log.status == 1 else 'failed',
            'confidence': round(log.confidence * 100, 1) if log.confidence else 0,
            'duration': log.duration
        })
        
    avg_duration = db.session.query(func.avg(LicensePlate.duration)).scalar()
    avg_duration = round(avg_duration, 2) if avg_duration else 0.00
    
    return jsonify({
        'logs': history_data,
        'stats': {
            'total': LicensePlate.query.count(),
            'success': LicensePlate.query.filter_by(status=1).count(),
            'failed': LicensePlate.query.filter_by(status=0).count(),
            'avg_time': avg_duration
        }
    })

@app.route('/api/download/history')
def download_history():
    records = LicensePlate.query.order_by(LicensePlate.time.desc()).all()
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(['ID', '时间', '车牌号', '车辆类型', '识别状态', '置信度', '识别耗时(秒)'])
    for r in records:
        cw.writerow([r.id, r.time.strftime('%Y-%m-%d %H:%M:%S'), r.plate_text, r.type, 
                     "成功" if r.status==1 else "失败", r.confidence, r.duration])
    output = make_response(si.getvalue())
    output.headers["Content-Disposition"] = "attachment; filename=history_records.csv"
    output.headers["Content-type"] = "text/csv; charset=utf-8-sig"
    return output

@app.route('/api/settings/update', methods=['POST'])
def update_settings():
    try:
        data = request.json
        config = SystemConfig.query.first()
        if config:
            config.confidence_threshold = int(data.get('confidence_threshold'))
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'error': 'Config not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def index():
    config = SystemConfig.query.first()
    if not config:
        config = SystemConfig(confidence_threshold=85)
        db.session.add(config)
        db.session.commit()
    return render_template('index.html', config=config)

@app.route('/api/performance')
def performance():
    cpu = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory().percent
    gpu = 0
    try:
        gpus = GPUtil.getGPUs()
        if gpus: gpu = gpus[0].load * 100
    except: pass
    return jsonify({'cpu_usage': round(cpu, 1), 'memory_usage': round(mem, 1), 'gpu_usage': round(gpu, 1)})

# --- 启动 ---
def start_flask():
    app.run(host='127.0.0.1', port=5000, threaded=True, use_reloader=False)

if __name__ == '__main__':
    with app.app_context():
        try:
            db.create_all()
        except: pass

    t = threading.Thread(target=start_flask)
    t.daemon = True
    t.start()
    
    icon_path = get_resource_path('apple-touch-icon.png')
    window = webview.create_window(
        title="车牌识别系统",
        url='http://127.0.0.1:5000',
        width=1480, height=900,
        resizable=True, confirm_close=True
    )
    webview.start(icon=icon_path)