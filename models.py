from datetime import datetime
from exts import db

class IdentificationRecord(db.Model):
    __tablename__ = 'identification_record'
    id = db.Column(db.Integer, primary_key=True)
    total = db.Column(db.Integer, default=0)
    success = db.Column(db.Integer, default=0)
    fail = db.Column(db.Integer, default=0)
    TimeConsuming = db.Column(db.DateTime, default=datetime.now)

    def __repr__(self):
        return f"<IdentificationRecord {self.id} - Total: {self.total}, Success: {self.success}, Fail: {self.fail}>"

class LicensePlate(db.Model):
    __tablename__ = 'license_plate'
    id = db.Column(db.Integer, primary_key=True)
    number = db.Column(db.String(10), unique=True, nullable=False)
    type = db.Column(db.String(10), nullable=False)
    status = db.Column(db.Integer, default=0)
    time = db.Column(db.DateTime, default=datetime.now)
