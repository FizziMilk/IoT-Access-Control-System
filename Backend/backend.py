from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy
from flask_mqtt import Mqtt
from datetime import datetime
from twilio.rest import Client
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token
import ssl
import os
import json

# Load environment variables
load_dotenv("twilio.env")

TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
MQTT_BROKER_URL = os.getenv("MQTT_BROKER_URL")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
TWILIO_VERIFY_SID = "VA1c1c5d9906340e1c187f83dbc26057a0"

# Validate Twilio credentials
if not TWILIO_ACCOUNT_SID or not TWILIO_AUTH_TOKEN:
    raise ValueError("Twilio credentials are missing")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = Flask(__name__)
api = Api(app)
bcrypt = Bcrypt(app)

# Flask-MQTT Configuration
app.config['MQTT_BROKER_URL'] = MQTT_BROKER_URL
app.config['MQTT_BROKER_PORT'] = 8883
app.config['MQTT_TLS_VERSION'] = ssl.PROTOCOL_TLSv1_2
app.config['MQTT_TLS_ENABLED'] = True
app.config['MQTT_TLS_CA_CERTS'] = '/etc/mosquitto/ca_certificates/ca.crt'

mqtt = Mqtt(app)

# JWT Setup
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # Token expires in 1 hour
jwt = JWTManager(app)

# SQLite database configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///access_logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'pool_pre_ping': True}

db = SQLAlchemy(app)

class AccessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(50), nullable=False)
    method = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

def log_access(user, method, status):
    new_log = AccessLog(user=user, method=method, status=status)
    db.session.add(new_log)
    db.session.commit()

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    phone_number = db.Column(db.String(20), unique=True, nullable=False)

    def set_password(self, password):
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password):
        return bcrypt.check_password_hash(self.password_hash, password)

class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    day = db.Column(db.String(10), unique=True, nullable=False)
    open_time = db.Column(db.Time, nullable=True)
    close_time = db.Column(db.Time, nullable=True)
    force_locked = db.Column(db.Boolean, default=False)

    def to_dict(self):
        return {
            "day": self.day,
            "open_time": self.open_time.strftime("%H:%M") if self.open_time else None,
            "close_time": self.close_time.strftime("%H:%M") if self.close_time else None,
            "force_locked": self.force_locked,
        }

with app.app_context():
    db.create_all()

class LoginResource(Resource):
    def post(self):
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return {"error": "Username and password required"}, 400

        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return {"error": "Invalid credentials"}, 401

        try:
            verification = client.verify.v2.services(TWILIO_VERIFY_SID).verifications.create(to=user.phone_number, channel="sms")
            return {"status": "OK", "message": "OTP sent", "phone_number": user.phone_number, "verification_status": verification.status}, 200
        except Exception as e:
            return {"error": str(e)}, 500

class CheckVerification(Resource):
    def post(self):
        data = request.get_json()
        username = data.get("username")
        code = data.get("otp")

        user = User.query.filter_by(username=username).first()
        if not user:
            return {"error": "User not found"}, 404

        try:
            verification_check = client.verify.v2.services(TWILIO_VERIFY_SID).verification_checks.create(to=user.phone_number, code=code)
            if verification_check.status == "approved":
                access_token = create_access_token(identity=user.id)
                return {"status": "approved", "token": access_token}, 200
            return {"status": verification_check.status}, 200
        except Exception as e:
            return {"error": str(e)}, 500

class ScheduleAPI(Resource):
    def get(self):
        return jsonify([entry.to_dict() for entry in Schedule.query.all()])

    def put(self):
        data = request.get_json()
        if not isinstance(data, list):
            return {"error": "Expected a list"}, 400

        for entry in data:
            db_entry = Schedule.query.filter_by(day=entry["day"]).first()
            open_time = datetime.strptime(entry["open_time"], "%H:%M").time() if entry["open_time"] else None
            close_time = datetime.strptime(entry["close_time"], "%H:%M").time() if entry["close_time"] else None

            if db_entry:
                db_entry.open_time = open_time
                db_entry.close_time = close_time
                db_entry.force_locked = entry["force_locked"]
            else:
                db.session.add(Schedule(day=entry["day"], open_time=open_time, close_time=close_time, force_locked=entry["force_locked"]))

        db.session.commit()
        mqtt.publish("door/schedule", json.dumps(data))
        return {"message": "Schedule updated successfully"}, 200

api.add_resource(LoginResource, "/login")
api.add_resource(CheckVerification, "/verify-otp")
api.add_resource(ScheduleAPI, "/schedule")

if __name__ == '__main__':
    app.run(debug=True)