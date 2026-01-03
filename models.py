from datetime import datetime
from exts import db

# 车牌信息记录
class LicensePlate(db.Model):
    __tablename__ = 'license_plate'
    id = db.Column(db.Integer, primary_key=True)
    plate_text = db.Column(db.String(10), unique=True, nullable=False)
    # 增加长度以容纳 BlueCard 等字符串
    type = db.Column(db.String(20), nullable=False)
    status = db.Column(db.Integer, default=0)
    # [新增] 识别置信度
    confidence = db.Column(db.Float, default=0.0)
    # [新增] 识别用时 (秒)
    duration = db.Column(db.Float, default=0.0)
    time = db.Column(db.DateTime, default=datetime.now)

class SystemConfig(db.Model):
    __tablename__ = 'system_config'
    id = db.Column(db.Integer, primary_key=True)
    # 置信度阈值 (0-100)
    confidence_threshold = db.Column(db.Integer, default=85)