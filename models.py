from datetime import datetime
from exts import db



# 车牌信息记录
class LicensePlate(db.Model):
    __tablename__ = 'license_plate'
    id = db.Column(db.Integer, primary_key=True)
    plate_text = db.Column(db.String(10), unique=True, nullable=False)
    type = db.Column(db.String(10), nullable=False)
    status = db.Column(db.Integer, default=0)
    time = db.Column(db.DateTime, default=datetime.now)
