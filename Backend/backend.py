from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy
from flask_mqtt import Mqtt
from datetime import datetime
from twilio.rest import Client
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token
from apscheduler.schedulers.background import BackgroundScheduler
from apsecheduler.triggers.date import DateTrigger
import ssl
import os
import json

# Twilio Setup
load_dotenv("twilio.env")  # Load the Twilio environment file
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# Other secrets
MQTT_BROKER_URL = os.getenv("MQTT_BROKER_URL")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

TWILIO_VERIFY_SID = "VA1c1c5d9906340e1c187f83dbc26057a0"
####

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

## JWT Setup
app.config["JWT_SECRET_KEY"] = JWT_SECRET_KEY
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = 3600  # Token expires in 1 hour
jwt = JWTManager(app)

## Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///access_logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

## Define AccessLog model

class AccessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user = db.Column(db.String(50), nullable=True)
    method = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.timezone.utc)
    def __repr__(self):
        return f"<AccessLog {self.user} - {self.status}>"

## User database model

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    phone_number = db.Column(db.String(20), unique=True, nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password: str) -> bool:
        return bcrypt.check_password_hash(self.password_hash, password)

## Door schedule database model

class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    day = db.Column(db.String(10), unique=True, nullable=False)
    open_time = db.Column(db.Time, nullable=True)
    close_time = db.Column(db.Time, nullable=True)
    force_locked = db.Column(db.Boolean, default=False)  # True means door is forced locked
    #Updated after database creation, needs cleanup

    def to_dict(self):
        # Return forceUnlocked as the inverse of force_locked.
        return {
            "day": self.day,
            "open_time": self.open_time.strftime("%H:%M") if self.open_time else None,
            "close_time": self.close_time.strftime("%H:%M") if self.close_time else None,
            "forceUnlocked": not self.force_locked,  # Mobile app expects 'forceUnlocked'
        }

# Create the database tables if they don't exist
with app.app_context():
    db.create_all()

# Handles login request. Checks username and password
class LoginResource(Resource):
    def post(self):
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return {"error": "Username and password required"}, 400
        # Check credentials
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            return {"error": "Invalid credentials"}, 401
        # Credentials are valid, send OTP to user's phone
        try:
            verification = client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verifications \
                .create(to=user.phone_number, channel="sms")
            # Return OK to let client proceed 
            return {
                "status": "OK",
                "message": "OTP sent to user",
                "phone_number": user.phone_number,
                "verification_status": verification.status
            }, 200
        except Exception as e:
            return {"error": str(e)}, 500

## Check OTP for phone login
class CheckVerification(Resource):
    def post(self):
        data = request.get_json()
        username = data.get("username")
        code = data.get("otp")

        if not username or not code:
            return {"error": "username and code are required"}, 400

        user = User.query.filter_by(username=username).first()
        if not user:
            return {"error": "User not found"}, 404

        try:
            verification_check = client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verification_checks \
                .create(to=user.phone_number, code=code)

            if verification_check.status == "approved":
                access_token = create_access_token(identity=user.id)
                return {"status": "approved", "token": access_token}, 200
            else:
                return {"status": verification_check.status}, 200

        except Exception as e:
            return {"error": str(e)}, 500


# Function to send OTP code
def send_otp(phone_number):
    try:
        # Send the OTP code via Twilio Verify
        verification = client.verify.v2.services(TWILIO_VERIFY_SID) \
            .verifications \
            .create(to=phone_number, channel ="sms")

        new_log = AccessLog(
            user=phone_number,
            method="SMS OTP",
            status="Started"
        )
        db.session.add(new_log)
        db.session.commit()
        
        return {"status": "OTP sent"},200

    except Exception as e:
        # Log failure if Twilio call fails
        new_log = AccessLog(
            user=phone_number or "Unknown",
            method="SMS OTP",
            status="Failed"
        )
        db.session.add(new_log)
        db.session.commit()

        return {"error": str(e)}, 500
        
# Verification for users at the door
class StartVerificationRPI(Resource):
    def post(self):
        data = request.get_json()
        phone_number = data.get("phone_number")
        schedule_time = data.get("schedule_time")

        if not phone_number:
            return {"error": "phone_number is required"}, 400

        try:
            if schedule_time:
                # Schedule the OTP sending
                schedule_time = datetime.strptime(schedule_time, "%Y-%m-%dT%H:%M%S")
                scheduler.add_job(send_otp, DateTrigger(run_date=schedule_time), args=[phone_number])
                return{"status": "OTP scheduled"}, 200
            else:
                # Send the OTP immediately
                return send_otp(phone_number)
            
        except Exception as e:
            return {"error": str(e)}, 500
        
# Initialize the scheduler
scheduler = BackgroundScheduler()
scheduler.start()

# Twilio Verification Check for users at the door
class CheckVerificationRPI(Resource):
    def post(self):
        data = request.get_json()
        phone_number = data.get("phone_number")
        otp_code = data.get("otp_code")

        if not phone_number or not otp_code:
            return {"error": "phone_number and otp_code are required"}, 400
        
        try:
            verification_check = client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verification_checks \
                .create(to=phone_number, code=otp_code)
            
            if verification_check.status == "approved":
                return {"status": "approved"}, 200
            else:
                return{"status": verification_check.status}, 400
            
        except Exception as e:
            return {"error": str(e)},500
                
        

# Resource to retrieve all logs
class GetAccessLogs(Resource):
    def get(self):
        logs = AccessLog.query.all()
        results = []
        for log in logs:
            results.append({
                "id": log.id,
                "user": log.user,
                "method": log.method,
                "status": log.status,
                "timestamp": log.timestamp.isoformat()
            })
        return results, 200

## Handles the setting of door schedule
class ScheduleAPI(Resource):
    def get(self):
        schedule = Schedule.query.all()
        return jsonify([entry.to_dict() for entry in schedule])

    def put(self):
        data = request.get_json()
        if not isinstance(data, list):
            return {"error": "Invalid data format, expected a list"}, 400

        for entry in data:
            db_entry = Schedule.query.filter_by(day=entry["day"]).first()

            # Convert time strings to time objects if provided
            open_time = datetime.strptime(entry["open_time"], "%H:%M").time() if entry["open_time"] else None
            close_time = datetime.strptime(entry["close_time"], "%H:%M").time() if entry["close_time"] else None

            # Mobile app sends forceUnlocked (true means door forced unlocked),
            # so store the inverse in force_locked.
            force_unlocked = entry.get("forceUnlocked", False)
            force_locked = not force_unlocked

            if db_entry:
                db_entry.open_time = open_time
                db_entry.close_time = close_time
                db_entry.force_locked = force_locked
            else:
                db.session.add(Schedule(day=entry["day"], open_time=open_time, close_time=close_time, force_locked=force_locked))
        db.session.commit()

        # Send schedule update via MQTT
        mqtt_payload = json.dumps(data)
        mqtt.publish("door/schedule", mqtt_payload)

        return {"message": "Schedule updated successfully"}, 200

## MQTT Resources

## Event fires when app connects (debug purposes)
@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected to MQTT broker!:")
        mqtt.subscribe('door/responses')
    else:
        print("Failed to connect, return code %d\n", rc)

# MQTT Resource to unlock door with RPI
class UnlockDoor(Resource):
    def post(self):
        data = request.get_json()
        command = data.get("command", "unlock_door")
        mqtt.publish("door/commands", command, qos=1)
        return {"status": f"Door command '{command}' sent"}, 200

# MQTT Resource to lock door with RPI
class LockDoor(Resource):
    def post(self):
        data = request.get_json()
        command = data.get("command", "lock_door")
        # Fixing typo in topic name: 'door/commmands' -> 'door/commands'
        mqtt.publish("door/commands", command, qos=1)
        return {"status": f"Door command '{command}' sent"}, 200

## Exposing RESTful API endpoints
api.add_resource(LoginResource, "/login")
api.add_resource(CheckVerification, "/verify-otp")
api.add_resource(StartVerificationRPI, "/start-verification")
api.add_resource(CheckVerificationRPI, "/check-verification-RPI")
api.add_resource(UnlockDoor, '/unlock')
api.add_resource(ScheduleAPI, "/schedule")
api.add_resource(GetAccessLogs, "/access-logs")

if __name__ == '__main__':
    app.run(debug=True)
