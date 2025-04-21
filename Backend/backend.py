import os
import time
import json
import base64
from datetime import datetime, timezone
from flask import Flask, request, jsonify
from flask_restful import Api, Resource
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from flask_mqtt import Mqtt
from twilio.rest import Client
from flask_jwt_extended import JWTManager, jwt_required, create_access_token, get_jwt_identity
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import random
import threading
from flask_bcrypt import Bcrypt
from twilio.base.exceptions import TwilioRestException
from dotenv import load_dotenv
import ssl

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Load environment variables
load_dotenv("twilio.env")  # Load Twilio settings
load_dotenv()  # Load default .env for MQTT and other settings

# Twilio Setup
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
TWILIO_VERIFY_SID = os.getenv("TWILIO_VERIFY_SID")
# Other secrets
MQTT_BROKER_URL = os.getenv("MQTT_BROKER_URL")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

app = Flask(__name__)
api = Api(app)
bcrypt = Bcrypt(app)

# Flask-MQTT Configuration
app.config['MQTT_BROKER_URL'] = MQTT_BROKER_URL
app.config['MQTT_BROKER_PORT'] = int(os.getenv("MQTT_PORT", 8883))
app.config['MQTT_TLS_VERSION'] = ssl.PROTOCOL_TLSv1_2
app.config['MQTT_TLS_ENABLED'] = True
# Use env var for CA certs, default to internal_ca.pem
app.config['MQTT_TLS_CA_CERTS'] = os.getenv('MQTT_CA_CERTS', '/etc/mosquitto/certs/internal_ca.pem')
# Client certificate and key (for mTLS)
app.config['MQTT_TLS_CERTFILE'] = os.getenv('MQTT_CLIENT_CERT', None)
app.config['MQTT_TLS_KEYFILE']  = os.getenv('MQTT_CLIENT_KEY', None)

print(f"[DEBUG] Attempting to connect to MQTT broker at {MQTT_BROKER_URL}: {app.config['MQTT_BROKER_PORT']}")
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
    user_name = db.Column(db.String(100), nullable=True)  # Add user_name field
    method = db.Column(db.String(50), nullable=False)
    status = db.Column(db.String(20), nullable=False)
    timestamp = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    def __repr__(self):
        return f"<AccessLog {self.user} - {self.status}>"

## Admin database model

class Admin(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    phone_number = db.Column(db.String(20), unique=True, nullable=False)

    def set_password(self, password: str) -> None:
        self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

    def check_password(self, password: str) -> bool:
        return bcrypt.check_password_hash(self.password_hash, password)

## User database model

class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    name = db.Column(db.String(50), nullable = True)
    phone_number = db.Column(db.String(20), unique=True, nullable=False)
    is_allowed = db.Column(db.Boolean, default=False)
    face_data = db.Column(db.LargeBinary, nullable=True) # Legacy face data stored as binary
    face_encodings = db.Column(db.Text, nullable=True) # Multiple face encodings stored as JSON
    face_registered = db.Column(db.Boolean, default=False) # Whether face has been registered
    low_security = db.Column(db.Boolean, default=False) # Whether user can bypass OTP verification

## Door schedule database model

class Schedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    day = db.Column(db.String(10), unique=True, nullable=False)
    open_time = db.Column(db.Time, nullable=True)
    close_time = db.Column(db.Time, nullable=True)
    force_unlocked = db.Column(db.Boolean, default=False)  # True means door is forced unlocked

    def to_dict(self):
        return {
            "day": self.day,
            "open_time": self.open_time.strftime("%H:%M") if self.open_time else None,
            "close_time": self.close_time.strftime("%H:%M") if self.close_time else None,
            "forceUnlocked": self.force_unlocked, 
        }

## User schedule database model
class UserSchedule(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    start_date = db.Column(db.DateTime, nullable=False)
    end_date = db.Column(db.DateTime, nullable=False)
    created_at = db.Column(db.DateTime, default=lambda: datetime.now(timezone.utc))
    
    def __repr__(self):
        return f"<UserSchedule {self.user_id} {self.start_date} to {self.end_date}>"

# Create the database tables if they don't exist
with app.app_context():
    db.create_all()
    # Insert default schedule if none exists
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if Schedule.query.count() == 0:
        for day in days:
            open_time = datetime.strptime("09:00", "%H:%M").time()
            close_time = datetime.strptime("13:00", "%H:%M").time()
            new_schedule = Schedule(day=day, open_time=open_time, close_time=close_time, force_unlocked=False)
            db.session.add(new_schedule)
        db.session.commit()
        print("Default schedule added.")

# Handles login request for admin. Checks username and password
class LoginResource(Resource):
    def post(self):
        data = request.get_json()
        username = data.get("username")
        password = data.get("password")

        if not username or not password:
            return {"error": "Username and password required"}, 400
        # Check credentials
        admin = Admin.query.filter_by(username=username).first()
        if not admin or not admin.check_password(password):
            return {"error": "Invalid credentials"}, 401
        # Credentials are valid, send OTP to user's phone
        try:
            verification = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verifications \
                .create(to=admin.phone_number, channel="sms")
            # Return OK to let client proceed 
            return {
                "status": "OK",
                "message": "OTP sent to user",
                "phone_number": admin.phone_number,
                "verification_status": verification.status
            }, 200
        except Exception as e:
            print(f"Error sending OTP: {e}" )
            return {"error": str(e)}, 500

## Check OTP for phone login
class CheckVerification(Resource):
    def post(self):
        data = request.get_json()
        username = data.get("username")
        code = data.get("otp")

        if not username or not code:
            return {"error": "username and code are required"}, 400

        admin = Admin.query.filter_by(username=username).first()
        if not admin:
            return {"error": "User not found"}, 404

        try:
            verification_check = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verification_checks \
                .create(to=admin.phone_number, code=code)

            if verification_check.status == "approved":
                # Log successful admin login
                new_log = AccessLog(
                    user=admin.username,
                    user_name=admin.username,
                    method="Admin Login",
                    status="Successful"
                )
                db.session.add(new_log)
                db.session.commit()
                
                # Create JWT token with admin ID as identity
                print(f"[DEBUG] Creating access token for admin ID: {admin.id}")
                access_token = create_access_token(identity=str(admin.id))  # Convert ID to string
                print(f"[DEBUG] Generated token length: {len(access_token)}")
                return {"status": "approved", "token": access_token}, 200
            else:
                # Log failed admin login attempt
                new_log = AccessLog(
                    user=admin.username,
                    user_name=admin.username,
                    method="Admin Login",
                    status="Invalid OTP"
                )
                db.session.add(new_log)
                db.session.commit()
                
                return {"status": verification_check.status}, 200

        except Exception as e:
            return {"error": str(e)}, 500

# Function to send OTP code
def send_otp(phone_number):
    try:
        # Send the OTP code via Twilio Verify
        verification = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
            .verifications \
            .create(to=phone_number, channel ="sms")

        # Get user name if available
        user = User.query.filter_by(phone_number=phone_number).first()
        user_name = user.name if user else None

        new_log = AccessLog(
            user=phone_number,
            user_name=user_name,
            method="SMS OTP",
            status="Started"
        )
        db.session.add(new_log)
        db.session.commit()
        
        return {"status": "OTP sent"},200

    except TwilioRestException as e:
        print(f"[ERROR] Twilio error: {e}")
        return {"error": f"Twilio error: {e.msg}"}, 500

    except Exception as e:
        # Log failure if Twilio call fails
        new_log = AccessLog(
            user=phone_number or "Unknown",
            user_name=None,
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
            send_otp(phone_number)
            
        except Exception as e:
            return {"error": str(e)}, 500
        
# Twilio Verification Check for users at the door
class CheckVerificationRPI(Resource):
    def post(self):
        data = request.get_json()
        phone_number = data.get("phone_number")
        otp_code = data.get("otp_code")

        if not phone_number or not otp_code:
            return {"error": "phone_number and otp_code are required"}, 400
        
        try:
            verification_check = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verification_checks \
                .create(to=phone_number, code=otp_code)
            
            if verification_check.status == "approved":
                # Log successful OTP verification
                user = User.query.filter_by(phone_number=phone_number).first()
                user_name = user.name if user else None
                
                new_log = AccessLog(
                    user=phone_number,
                    user_name=user_name,
                    method="SMS OTP",
                    status="Verified"
                )
                db.session.add(new_log)
                db.session.commit()
                
                return {"status": "approved"}, 200
            else:
                # Log invalid OTP attempt
                user = User.query.filter_by(phone_number=phone_number).first()
                user_name = user.name if user else None
                
                new_log = AccessLog(
                    user=phone_number,
                    user_name=user_name,
                    method="SMS OTP",
                    status="Invalid Code"
                )
                db.session.add(new_log)
                db.session.commit()
                
                return {"status": verification_check.status}, 400
            
        except Exception as e:
            return {"error": str(e)}, 500
        
# Door entry API resource
class DoorEntryAPI(Resource):
    def post(self):
        data = request.get_json()
        phone_number = data.get("phone_number")

        user = User.query.filter_by(phone_number=phone_number).first()
        
        if not user:
            # Create a pending user
            new_user = User(phone_number=phone_number, is_allowed=False)
            db.session.add(new_user)
            db.session.commit()
            
            return {
                "status": "pending",
                "message": "No access rights. Please contact administrator."
            }, 200

        if not user.is_allowed:
            return {
                "status": "pending", 
                "message": "Waiting for admin approval"
            }, 200
            
        # User exists and is allowed access, send OTP
        try:
            verification = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verifications \
                .create(to=phone_number, channel="sms")
            
            # Log the OTP request
            new_log = AccessLog(
                user=phone_number,
                user_name=user.name,
                method="OTP Request",
                status="Sent"
            )
            db.session.add(new_log)
            db.session.commit()
            
            # Ensure we're clearly communicating OTP was sent
            return {
                "status": "OTP sent",
                "message": "OTP has been sent to your phone",
                "verification_status": verification.status,
                "has_face": user.face_registered, # Add info about face registration
                "is_allowed": user.is_allowed,    # Explicitly confirm user is allowed
                "user_name": user.name            # Include user name for display purposes
            }, 200
        except Exception as e:
            print(f"Error sending OTP: {e}")
            return {
                "error": "Failed to send OTP",
                "details": str(e)
            }, 500

# Resource to retrieve all logs
class GetAccessLogs(Resource):
    def get(self):
        try:
            logs = AccessLog.query.order_by(AccessLog.timestamp.desc()).all()
            log_list = [
                {
                    "user": log.user,
                    "user_name": log.user_name,  # Include user_name in response
                    "method": log.method,
                    "status": log.status,
                    "timestamp": log.timestamp.isoformat()
                }
                for log in logs
            ]
            return jsonify(log_list)
        except Exception as e:
            print(f"[ERROR] Failed to fetch access logs: {str(e)}")  # Add logging
            return {
                "error": "Failed to fetch access logs",
                "details": str(e)
            }, 500


## Handles the setting of door schedule
class ScheduleAPI(Resource):
    def get(self):
        schedule = Schedule.query.all()
        schedule_list = [entry.to_dict() for entry in schedule]
        print(f"Schedule data: {schedule_list}")  # Add logging
        return jsonify(schedule_list)

    def put(self):
        data = request.get_json()
        print(f"Received schedule update: {data}")  # Add logging
        if not isinstance(data, list):
            return {"error": "Invalid data format, expected a list"}, 400

        for entry in data:
            db_entry = Schedule.query.filter_by(day=entry["day"]).first()

            # Convert time strings to time objects if provided
            open_time = datetime.strptime(entry["open_time"], "%H:%M").time() if entry["open_time"] else None
            close_time = datetime.strptime(entry["close_time"], "%H:%M").time() if entry["close_time"] else None
            force_unlocked = entry.get("forceUnlocked", False)
            
            if db_entry:
                db_entry.open_time = open_time
                db_entry.close_time = close_time
                db_entry.force_unlocked = force_unlocked
            else:
                db.session.add(Schedule(day=entry["day"], open_time=open_time, close_time=close_time, force_unlocked=force_unlocked))
        db.session.commit()

        # Send schedule update via MQTT
        mqtt_payload = json.dumps(data)
        mqtt.publish("door/schedule", mqtt_payload)

        return {"message": "Schedule updated successfully"}, 200
    
# API resource to manage users
class UserManagementAPI(Resource):
    def get(self):
        users = User.query.all()
        user_list = []
        for user in users:
            user_list.append({
                "id": user.id,
                "name": user.name,
                "phone_number": user.phone_number,
                "is_allowed": user.is_allowed,
                "low_security": user.low_security
            })
        return user_list, 200
    
    def put(self):
        data = request.get_json()
        user_id = data.get("id")
        if not user_id:
            return {"error": "User id required"}, 400
        user = User.query.get(user_id)
        if not user:
            return {"error": "User not found"}, 404
        new_permission = data.get("is_allowed")
        if new_permission is None:
            return {"error": "is_allowed field is required"}, 400
        user.is_allowed = new_permission
        # Optionally update the user name if provided
        if "name" in data:
            user.name = data["name"]
        # Update low_security setting if provided
        if "low_security" in data:
            user.low_security = data["low_security"]
        db.session.commit()
        return {"message": "User updated successfully"}, 200    
    
    def post(self):
        data = request.get_json()
        name = data.get("name")
        phone_number = data.get("phone_number")
        if not phone_number:
            return {"error": "Phone number is required"}, 400
        if User.query.filter_by(phone_number = phone_number).first():
            return {"error": "User with this phone number already exists"}, 400
        # Get optional low_security setting or default to False
        low_security = data.get("low_security", False)
        new_user = User(
            name=name, 
            phone_number=phone_number, 
            is_allowed=False,
            low_security=low_security
        )
        db.session.add(new_user)
        db.session.commit()
        return {"id": new_user.id,
                "name": new_user.name,
                "phone_number": new_user.phone_number,
                "is_allowed": new_user.is_allowed,
                "low_security": new_user.low_security}, 201
    
    def delete(self):
        data = request.get_json()
        user_id = data.get("id")
        if not user_id:
            return {"error": "User id is required"}, 400
        user = User.query.get(user_id)
        if not user:
            return {"error": "User not found"}, 404
        db.session.delete(user)
        db.session.commit()
        return{"message": "User deleted successfully"}, 200

class UpdateUserNameAPI(Resource):
    def post(self):
        data = request.get_json()
        phone_number = data.get("phone_number")
        name = data.get("name")
        if not phone_number or not name:
            return {"error": "phone_number and name are required"}, 400
        user = User.query.filter_by(phone_number=phone_number).first()
        if not user:
            return {"error": "User not found"}, 404
        user.name = name
        db.session.commit()
        return {"status": "success", "message": "Name updated"}, 200

## MQTT Resources

@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    print(f"[DEBUG] handle_connect triggered with rc: {rc}")
    rc_codes = {
        0: "Connected successfully",
        1: "Incorrect protocol version",
        2: "Invalid client identifier",
        3: "Server unavailable",
        4: "Bad username or password",
        5: "Not authorized"
    }
    print(f"[DEBUG] MQTT Connect callback with result code: {rc} ({rc_codes.get(rc, 'Unknown error')})")
    if rc == 0:
        mqtt.subscribe([
            ("door/commands", 1),
            ("door/schedule", 1),
            ("door/otp/verify", 1),
            (f"door/otp/response/+", 1)
        ])
        print("[DEBUG] Successfully subscribed to MQTT topics")
    else:
        print(f"[ERROR] MQTT connection failed with rc: {rc}")

@mqtt.on_subscribe()
def handle_subscribe(client, userdata, mid, granted_qos):
    print(f"[DEBUG] Subscribed to topic with mid: {mid}, granted QoS: {granted_qos}")

@mqtt.on_disconnect()
def handle_disconnect(client, userdata, rc):
    print(f"[DEBUG] Disconnected from MQTT broker with code {rc}")
    try:
        mqtt._connect()  # Force connection attempt
    except Exception as e:
        print(f"[ERROR] Failed to connect to MQTT: {e}")

@mqtt.on_log()
def handle_logging(client, userdata, level, buf):
    print(f"[MQTT LOG] {buf}")

# MQTT Handler for OTP verification requests
@mqtt.on_message()
def handle_otp_verification(client, userdata, message):
    if message.topic == "door/otp/verify":
        try:
            payload = message.payload.decode()
            data = json.loads(payload)
            phone_number = data.get("phone_number")
            otp_code = data.get("otp_code")
            print(f"[DEBUG] Received OTP verification request for {phone_number}")

            # Get user name if available
            user = User.query.filter_by(phone_number=phone_number).first()
            if not user:
                res = {"phone_number": phone_number, "status": "denied", "message": "User not found"}
                mqtt.publish(f"door/otp/response/{phone_number}", json.dumps(res), qos=1)
                return

            user_name = user.name if user else None

            # 1. Check global schedule first
            current_time = datetime.now(timezone.utc)
            current_day = current_time.strftime("%A")  # Get current day name
            global_schedule = Schedule.query.filter_by(day=current_day).first()
            
            # If door is force unlocked globally, allow direct access
            if global_schedule and global_schedule.force_unlocked:
                # Log the access attempt
                new_log = AccessLog(
                    user=phone_number,
                    user_name=user_name,
                    method="Global Force Unlock",
                    status="Door Unlocked"
                )
                db.session.add(new_log)
                db.session.commit()
                res = {"phone_number": phone_number, "status": "approved", "message": "Door is globally unlocked"}
                mqtt.publish(f"door/otp/response/{phone_number}", json.dumps(res), qos=1)
                return

            # If within global schedule hours, allow direct access
            if global_schedule and global_schedule.open_time and global_schedule.close_time:
                current_time_only = current_time.time()
                if global_schedule.open_time <= current_time_only <= global_schedule.close_time:
                    # Log the access attempt
                    new_log = AccessLog(
                        user=phone_number,
                        user_name=user_name,
                        method="Global Schedule Hours",
                        status="Door Unlocked"
                    )
                    db.session.add(new_log)
                    db.session.commit()
                    res = {"phone_number": phone_number, "status": "approved", "message": "Within global schedule hours"}
                    mqtt.publish(f"door/otp/response/{phone_number}", json.dumps(res), qos=1)
                    return

            # 2. If not globally accessible, check if user is allowed
            if user.is_allowed:
                # User is allowed, require OTP verification
                verification_check = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                    .verification_checks \
                    .create(to=phone_number, code=otp_code)
                
                if verification_check.status == "approved":
                    new_log = AccessLog(
                        user=phone_number,
                        user_name=user_name,
                        method="SMS OTP",
                        status="Door Unlocked"
                    )
                    db.session.add(new_log)
                    db.session.commit()
                    res = {"phone_number": phone_number, "status": "approved", "message": "OTP verified"}
                else:
                    # Log invalid OTP attempt for users with schedules
                    new_log = AccessLog(
                        user=phone_number,
                        user_name=user_name,
                        method="SMS OTP",
                        status="Invalid Code"
                    )
                    db.session.add(new_log)
                    db.session.commit()
                    res = {"phone_number": phone_number, "status": "denied", "message": "Invalid OTP"}
                mqtt.publish(f"door/otp/response/{phone_number}", json.dumps(res), qos=1)
                return

            # 3. If user is not allowed, check their schedule
            user_schedule = UserSchedule.query.filter(
                UserSchedule.user_id == user.id,
                UserSchedule.start_date <= current_time,
                UserSchedule.end_date >= current_time
            ).first()

            if user_schedule:
                # User has valid schedule, require OTP verification
                verification_check = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                    .verification_checks \
                    .create(to=phone_number, code=otp_code)
                
                if verification_check.status == "approved":
                    new_log = AccessLog(
                        user=phone_number,
                        user_name=user_name,
                        method="SMS OTP",
                        status="Door Unlocked"
                    )
                    db.session.add(new_log)
                    db.session.commit()
                    res = {"phone_number": phone_number, "status": "approved", "message": "OTP verified"}
                else:
                    res = {"phone_number": phone_number, "status": "denied", "message": "Invalid OTP"}
            else:
                res = {"phone_number": phone_number, "status": "denied", "message": "No valid schedule found"}

        except Exception as e:
            res = {"phone_number": phone_number, "status": "error", "message": str(e)}
            print(f"[DEBUG] Error during OTP verification for {phone_number}: {e}")

        # Publish the verification response
        response_topic = f"door/otp/response/{phone_number}"
        mqtt.publish(response_topic, json.dumps(res), qos=1)
        print(f"[DEBUG] Published OTP verification result to {response_topic}")

# Resource to manage user schedules
class UserScheduleAPI(Resource):
    def get(self):
        try:
            schedules = UserSchedule.query.all()
            schedule_list = [
                {
                    "id": schedule.id,
                    "user_id": schedule.user_id,
                    "start_date": schedule.start_date.isoformat(),
                    "end_date": schedule.end_date.isoformat(),
                    "created_at": schedule.created_at.isoformat()
                }
                for schedule in schedules
            ]
            return jsonify(schedule_list)
        except Exception as e:
            return {"error": str(e)}, 500

    def post(self):
        try:
            data = request.get_json()
            user_id = data.get("user_id")
            start_date = datetime.fromisoformat(data.get("start_date"))
            end_date = datetime.fromisoformat(data.get("end_date"))

            if not all([user_id, start_date, end_date]):
                return {"error": "Missing required fields"}, 400

            # Check if user exists
            user = User.query.get(user_id)
            if not user:
                return {"error": "User not found"}, 404

            # Check for overlapping schedules
            existing_schedule = UserSchedule.query.filter(
                UserSchedule.user_id == user_id,
                UserSchedule.start_date <= end_date,
                UserSchedule.end_date >= start_date
            ).first()

            if existing_schedule:
                return {"error": "Schedule overlaps with existing schedule"}, 400

            new_schedule = UserSchedule(
                user_id=user_id,
                start_date=start_date,
                end_date=end_date
            )
            db.session.add(new_schedule)
            db.session.commit()

            return {
                "id": new_schedule.id,
                "user_id": new_schedule.user_id,
                "start_date": new_schedule.start_date.isoformat(),
                "end_date": new_schedule.end_date.isoformat(),
                "created_at": new_schedule.created_at.isoformat()
            }, 201

        except Exception as e:
            return {"error": str(e)}, 500

    def delete(self):
        try:
            data = request.get_json()
            schedule_id = data.get("id")

            if not schedule_id:
                return {"error": "Schedule ID is required"}, 400

            schedule = UserSchedule.query.get(schedule_id)
            if not schedule:
                return {"error": "Schedule not found"}, 404

            db.session.delete(schedule)
            db.session.commit()

            return {"message": "Schedule deleted successfully"}, 200

        except Exception as e:
            return {"error": str(e)}, 500
        
# MQTT Resource to unlock door with RPI
class UnlockDoor(Resource):
    @jwt_required()  # Require JWT authentication
    def post(self):
        data = request.get_json()
        command = data.get("command", "unlock_door")
        
        print(f"[DEBUG] Received unlock door request with command: {command}")
        
        # Log admin unlock action
        try:
            admin_id = get_jwt_identity()
            print(f"[DEBUG] JWT identity extracted: {admin_id}")
            
            if admin_id is None:
                print("[ERROR] Failed to extract admin ID from JWT token")
                raise ValueError("Invalid JWT token - could not extract admin ID")
                
            admin = Admin.query.get(admin_id)
            if admin is None:
                print(f"[ERROR] No admin found with ID: {admin_id}")
                raise ValueError(f"Admin with ID {admin_id} not found in database")
                
            admin_name = admin.username
            
            print(f"[DEBUG] Admin authenticated: {admin_name} (ID: {admin_id})")
            
            new_log = AccessLog(
                user=admin_name,
                user_name=admin_name,
                method="Mobile App",
                status="Door Unlocked"
            )
            db.session.add(new_log)
            db.session.commit()
            print(f"[DEBUG] Admin door unlock logged to database")
        except Exception as e:
            print(f"[ERROR] Failed to log admin door unlock: {str(e)}")
            # Continue with door unlock even if logging fails
        
        # Send the command to unlock the door
        print(f"[DEBUG] Publishing MQTT message to topic 'door/commands': {command}")
        try:
            mqtt.publish("door/commands", command, qos=1)
            print(f"[DEBUG] MQTT message published successfully")
        except Exception as e:
            print(f"[ERROR] Failed to publish MQTT message: {str(e)}")
            return {"status": "error", "message": f"Failed to send door command: {str(e)}"}, 500
            
        return {"status": f"Door command '{command}' sent"}, 200

# MQTT Resource to lock door with RPI
class LockDoor(Resource):
    def post(self):
        # Send the lock door command via MQTT
        mqtt.publish("door/commands", "lock_door")
        return {"message": "Lock door command sent"}, 200

# New resource for logging door access events
class LogDoorAccess(Resource):
    def post(self):
        data = request.get_json()
        
        user = data.get("user", "Schedule System")
        method = data.get("method", "Unknown")
        status = data.get("status", "Unknown")
        details = data.get("details", "")
        
        # Get user name if available and user is a phone number
        user_name = None
        if user != "Schedule System":
            user_obj = User.query.filter_by(phone_number=user).first()
            if user_obj:
                user_name = user_obj.name
        
        # Create the log entry
        new_log = AccessLog(
            user=user,
            user_name=user_name,
            method=method,
            status=status
        )
        
        try:
            db.session.add(new_log)
            db.session.commit()
            return {"status": "success", "message": "Door access logged"}, 200
        except Exception as e:
            db.session.rollback()
            print(f"[ERROR] Failed to log door access: {str(e)}")
            return {"status": "error", "message": str(e)}, 500

class RegisterFaceAPI(Resource):
    def post(self):
        data = request.get_json()
        phone_number = data.get("phone_number")
        face_encoding_base64 = data.get("face_encoding")
        name = data.get("name", "")  # Get name with empty string as default
        is_additional = data.get("is_additional", False) # Flag to indicate if this is an additional encoding
        
        if not phone_number or not face_encoding_base64:
            return {"error": "Phone number and face encoding required"}, 400
            
        try:
            # Find or create user
            user = User.query.filter_by(phone_number=phone_number).first()
            
            if user:
                # Update existing user
                
                # If this is legacy data, migrate it
                if is_additional or user.face_encodings:
                    # Handle multiple encodings case - store as JSON
                    
                    # Load existing encodings if any
                    existing_encodings = []
                    if user.face_encodings:
                        try:
                            existing_encodings = json.loads(user.face_encodings)
                        except:
                            # If data is corrupt, start fresh
                            existing_encodings = []
                    
                    # Add the new encoding
                    existing_encodings.append(face_encoding_base64)
                    
                    # Store as JSON
                    user.face_encodings = json.dumps(existing_encodings)
                else:
                    # Always use JSON encodings, never legacy binary storage
                    user.face_encodings = json.dumps([face_encoding_base64])
                
                # Update name if provided and currently not set
                if name and (user.name is None or user.name == ""):
                    user.name = name
                    
                user.face_registered = True
                db.session.commit()
                
                # Log the update
                status = "Updated (Added Face)" if is_additional else "Updated"
                log = AccessLog(
                    user=phone_number,
                    user_name=user.name,
                    method="Face Registration",
                    status=status
                )
                db.session.add(log)
                db.session.commit()
                
                return {"status": "success", "message": "Face data updated for existing user"}, 200
            else:
                # Create new user with multiple encodings support
                new_user = User(
                    name=name,  # Set the name from the request
                    phone_number=phone_number,
                    is_allowed=False,  # Require admin approval
                    face_encodings=json.dumps([face_encoding_base64]),  # Store as JSON array
                    face_registered=True
                )
                db.session.add(new_user)
                db.session.commit()
                
                # Log the registration
                log = AccessLog(
                    user=phone_number,
                    user_name=name,  # Include the name in the log
                    method="Face Registration",
                    status="Pending Approval"
                )
                db.session.add(log)
                db.session.commit()
                
                return {"status": "success", "message": "New user registered with face"}, 201
        except Exception as e:
            print(f"Error registering face: {e}")
            return {"error": str(e)}, 500

class GetFaceDataAPI(Resource):
    def get(self):
        """Retrieve face data for all authorized users"""
        try:
            # Query users with registered faces who are allowed access
            users = User.query.filter(User.face_registered == True, User.is_allowed == True).all()
            
            face_data = []
            for user in users:
                # Process multiple encodings if available
                if user.face_encodings:
                    try:
                        # Parse JSON array of encodings
                        encodings_list = json.loads(user.face_encodings)
                        for encoding in encodings_list:
                            face_data.append({
                                "phone_number": user.phone_number,
                                "face_encoding": encoding
                            })
                    except Exception as e:
                        print(f"Error parsing face encodings for user {user.phone_number}: {e}")
                
                # Skip legacy face_data - we don't want to use numpy on the backend
            
            return {
                "status": "success",
                "face_data": face_data
            }, 200
        except Exception as e:
            print(f"Error retrieving face data: {e}")
            return {"error": str(e)}, 500

class GetUserEncodingsAPI(Resource):
    def get(self):
        """Retrieve face encodings for all registered users in a format optimized for face recognition"""
        try:
            # Query all users with registered faces, regardless of approval status
            users = User.query.filter(User.face_registered == True).all()
            
            if not users:
                return {"error": "No registered users found"}, 404
                
            result = {"users": []}
            
            for user in users:
                # Process multiple encodings if available
                if user.face_encodings:
                    try:
                        # Parse JSON array of encodings
                        encodings_list = json.loads(user.face_encodings)
                        if encodings_list:
                            # For each encoding, we need to decode it from base64, 
                            # then parse the JSON string to get the actual encoding values
                            processed_encodings = []
                            for encoding_base64 in encodings_list:
                                try:
                                    # First try to decode as JSON
                                    encoding_json = base64.b64decode(encoding_base64).decode('utf-8')
                                    encoding_data = json.loads(encoding_json)
                                    processed_encodings.append(encoding_data)
                                except Exception as e:
                                    print(f"Error decoding encoding for user {user.phone_number}: {e}")
                                    continue
                            
                            if processed_encodings:
                                result["users"].append({
                                    "id": user.id,
                                    "name": user.name or "Unknown User",
                                    "phone_number": user.phone_number,
                                    "is_allowed": user.is_allowed,  # Include approval status
                                    "low_security": getattr(user, "low_security", False),  # Handle existing users safely
                                    "face_encoding": processed_encodings[0]  # Just use the first encoding for now
                                })
                    except Exception as e:
                        print(f"Error processing encodings for user {user.phone_number}: {e}")
                
                # Skip legacy face_data handling - we're not going to use numpy for face comparisons
                # Legacy data won't be accessible but that's OK as per requirements
            
            return result, 200
        except Exception as e:
            print(f"Error retrieving user encodings: {e}")
            import traceback
            traceback.print_exc()
            return {"error": str(e)}, 500

## Exposing RESTful API endpoints
api.add_resource(LoginResource, "/login")
api.add_resource(CheckVerification, "/verify-otp")
api.add_resource(StartVerificationRPI, "/start-verification")
api.add_resource(CheckVerificationRPI, "/check-verification-RPI")
api.add_resource(UnlockDoor, '/unlock')
api.add_resource(ScheduleAPI, "/schedule")
api.add_resource(GetAccessLogs, "/access-logs")
api.add_resource(DoorEntryAPI, "/door-entry")
api.add_resource(UserManagementAPI, "/users")
api.add_resource(UpdateUserNameAPI, "/update-user-name")
api.add_resource(UserScheduleAPI, '/user-schedule')
api.add_resource(LockDoor, '/lock')
api.add_resource(LogDoorAccess, '/log-door-access')
api.add_resource(RegisterFaceAPI, '/register-face')
api.add_resource(GetFaceDataAPI, '/get-face-data')
api.add_resource(GetUserEncodingsAPI, '/get-user-encodings')

# Simple health check endpoint
@app.route('/health')
def health_check():
    return {"status": "ok"}, 200

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")



