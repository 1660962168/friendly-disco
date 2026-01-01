from datetime import datetime
from exts import db


# 设备

class IdentificationRecord(db.Model):
    __tablename__ = 'identification_record'
    id = db.Column(db.Integer, primary_key=True)
    total = db.Column(db.Integer, default=0)
    success = db.Column(db.Integer, default=0)
    fail = db.Column(db.Integer, default=0)
    time = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f"<IdentificationRecord {self.id} - Total: {self.total}, Success: {self.success}, Fail: {self.fail}>"