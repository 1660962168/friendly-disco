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
ocr = PaddleOCR(use_textline_orientation=False, lang="ch")

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


# --- 3. 核心路由 ---

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
        # 0. 清理旧文件
        cleanup_old_files(UPLOAD_FOLDER, limit=30)
        cleanup_old_files(RESULTS_FOLDER, limit=30)
        cleanup_old_files(CROPS_FOLDER, limit=30)

        # 1. 保存文件
        ext = os.path.splitext(file.filename)[1]
        if not ext: ext = ".jpg"
        safe_filename = f"{uuid.uuid4().hex}{ext}" 
        filepath = os.path.join(UPLOAD_FOLDER, safe_filename)
        file.save(filepath)

        # 2. YOLO 推理
        results = model(filepath)
        result = results[0]
        
        # 保存检测结果图
        annotated_frame = result.plot()
        result_filename = f"result_{safe_filename}"
        result_path = os.path.join(RESULTS_FOLDER, result_filename)
        cv2.imwrite(result_path, annotated_frame)

        # 3. 提取数据
        boxes = result.boxes.xyxy.cpu().numpy()
        conf_scores = result.boxes.conf.cpu().numpy()
        
        has_plate = False
        crop_filename = ""
        yolo_confidence = 0.0
        
        # 初始化 OCR 结果变量
        plate_text = ""
        plate_confidence = 0.0

        if len(boxes) > 0:
            box = boxes[0]
            yolo_confidence = float(conf_scores[0])
            
            # --- [关键优化] 增加 Padding (内边距) ---
            # 防止 YOLO 框切到字符边缘，导致 OCR 识别错误
            padding_x = 10
            padding_y = 10
            
            x1 = int(box[0]) - padding_x
            y1 = int(box[1]) - padding_y
            x2 = int(box[2]) + padding_x
            y2 = int(box[3]) + padding_y
            
            # 读取原图
            original_img = cv2.imread(filepath)
            h, w, _ = original_img.shape
            
            # 边界检查
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            
            # 裁剪
            plate_crop = original_img[y1:y2, x1:x2]
            
            # 保存裁剪图
            crop_filename = f"crop_{safe_filename}"
            crop_path = os.path.join(CROPS_FOLDER, crop_filename)
            cv2.imwrite(crop_path, plate_crop)
            
            # 保存临时图 (虽然现在主要用内存里的 plate_crop 处理)
            temp_crop_path = os.path.join(CROPS_FOLDER, 'temp_plate.jpg')
            cv2.imwrite(temp_crop_path, plate_crop)
            
            has_plate = True

            # ================= [终极优化] OCR 双向识别 + 格式修正 =================
            try:
                # 1. 放大图片 (提升小图识别率)
                roi = cv2.resize(plate_crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
                
                # 2. 转灰度
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                
                # 3. 反色 (针对蓝牌白字)
                gray_inv = cv2.bitwise_not(gray)

                # --- [关键修复] 转回 3 通道 BGR ---
                # PaddleOCR v3+ 强制要求输入 (H, W, 3)，否则报 tuple index out of range
                gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                gray_inv_bgr = cv2.cvtColor(gray_inv, cv2.COLOR_GRAY2BGR)
                
                # 省份简称列表
                PROVINCES = "京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼使领"
                candidates = []

                # 执行两次识别：一次原图(灰度)，一次反色(灰度)
                for img_input, label in [(gray_bgr, "Normal"), (gray_inv_bgr, "Inverted")]:
                    
                    # 调试：保存送入 OCR 的图片
                    # debug_path = os.path.join(CROPS_FOLDER, f'debug_{label}.jpg')
                    # cv2.imwrite(debug_path, img_input)
                    
                    # 识别
                    res = ocr.predict(img_input)
                    
                    # 解析结果
                    text_res = ""
                    score_res = 0.0

                    if res:
                        # 兼容 res 是列表的情况
                        if isinstance(res, list) and len(res) > 0:
                            item = res[0]
                            # 适配 v3 字典格式
                            if isinstance(item, dict) and 'rec_texts' in item and len(item['rec_texts']) > 0:
                                text_res = item['rec_texts'][0]
                                score_res = float(item['rec_scores'][0])
                            # 适配 v2 列表格式
                            elif isinstance(item, list) and len(item) >= 2:
                                text_res = item[1][0]
                                score_res = float(item[1][1])
                    
                    # 结果清洗与评分
                    if text_res:
                        # 1. 正则清洗：只留 汉字、大写字母、数字
                        clean_text = re.sub(r'[^\u4e00-\u9fa5A-Z0-9]', '', text_res)
                        
                        is_valid = False
                        current_score = score_res
                        
                        # 2. 校验：首字符是否为省份
                        if len(clean_text) >= 2:
                            if clean_text[0] in PROVINCES:
                                is_valid = True
                                current_score += 0.5 # 奖励分
                        
                        # 3. 校验：蓝牌反色模式下的有效结果通常更准
                        if label == "Inverted" and is_valid:
                            current_score += 0.2
                        
                        if len(clean_text) > 1:
                            candidates.append({
                                'text': clean_text,
                                'score': current_score,
                                'valid': is_valid,
                                'mode': label
                            })

                # 挑选最佳结果
                if candidates:
                    # 排序：Valid优先 > 分数高优先
                    candidates.sort(key=lambda x: (x['valid'], x['score']), reverse=True)
                    best = candidates[0]
                    
                    plate_text = best['text']
                    # 限制最高置信度显示为 0.99
                    plate_confidence = min(0.99, best['score'])
                    
                    print(f"OCR 最终选择 ({best['mode']}): {plate_text} (原始分:{best['score']:.2f} 置信度:{plate_confidence:.2f})")
                else:
                    print("OCR 双向识别均未找到有效车牌")

            except Exception as e:
                print(f"OCR 识别流程出错: {e}")
                traceback.print_exc()

        # 4. 返回 JSON
        return jsonify({
            'success': True,
            'original_url': url_for('static', filename=f'uploads/{safe_filename}'),
            'result_url': url_for('static', filename=f'results/{result_filename}'),
            'has_plate': has_plate,
            'crop_url': url_for('static', filename=f'crops/{crop_filename}') if has_plate else None,
            'plate_text': plate_text,       # 识别到的车牌文本
            'confidence': plate_confidence  # OCR 置信度 (用于前端展示)
        })

    except Exception as e:
        print(f"Error during detection: {e}")
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

    # 启动 GUI 窗口
    window = webview.create_window(
        title="车牌识别系统",
        url='http://127.0.0.1:5000',
        width=1480, 
        height=900,
        resizable=True,
        confirm_close=True,
    )
    
    icon_path = get_resource_path('apple-touch-icon.png')
    webview.start(icon=icon_path)