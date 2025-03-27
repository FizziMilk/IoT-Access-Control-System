from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy
from flask_mqtt import Mqtt
from datetime import datetime, timezone
from twilio.rest import Client
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager, create_access_token
import ssl
import os
import json
from twilio.base.exceptions import TwilioRestException
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s [%(levelname)s] %(message)s")

# Twilio Setup
load_dotenv("twilio.env")  # Load the Twilio environment file
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")
# Other secrets
MQTT_BROKER_URL = os.getenv("MQTT_BROKER_URL")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY")
twilio_client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

TWILIO_VERIFY_SID = "VA1c1c5d9906340e1c187f83dbc26057a0"

app = Flask(__name__)
api = Api(app)
bcrypt = Bcrypt(app)

# Flask-MQTT Configuration
app.config['MQTT_BROKER_URL'] = MQTT_BROKER_URL
app.config['MQTT_BROKER_PORT'] = 8883
app.config['MQTT_TLS_VERSION'] = ssl.PROTOCOL_TLSv1_2
app.config['MQTT_TLS_ENABLED'] = True
app.config['MQTT_TLS_CA_CERTS'] = '/etc/mosquitto/ca_certificates/ca.crt'

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
    is_allowed = db.Column(db.Boolean,default=False)
    face_data = db.Column(db.LargeBinary, nullable=True) # Assuming face data is stored as binary

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
                access_token = create_access_token(identity=admin.id)
                return {"status": "approved", "token": access_token}, 200
            else:
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

        new_log = AccessLog(
            user=phone_number,
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
                return {"status": "approved"}, 200
            else:
                return {"status": verification_check.status}, 400
            
        except Exception as e:
            return {"error": str(e)}, 500
        
# Door entry API resource
class DoorEntryAPI(Resource):
    def post(self):
        data = request.get_json()
        phone_number = data.get("phone_number")
        if not phone_number:
            return {"error": "phone_number is required"}, 400
        
        user = User.query.filter_by(phone_number=phone_number).first()
        if user:
            if user.is_allowed:
                # Allowed on file - send OTP now and return status.
                otp_result, status = send_otp(phone_number) 
                if status == 200:
                    return {"status" : "OTP sent", "phone_number": phone_number}, 200
                else:
                    return otp_result, status
            else:
                #User exists but not allowed yet.
                return {"status": "pending", "message": "Your number is pending review."}, 200
        else:
            #Not found - add to user database pending review.
            new_user = User(phone_number=phone_number, is_allowed=False)
            db.session.add(new_user)
            db.session.commit()
            return {"status": "pending", "message": "Number added. Your access is now pending review."}, 200                

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
                "is_allowed": user.is_allowed
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
        new_user = User(name=name, phone_number=phone_number, is_allowed=False)
        db.session.add(new_user)
        db.session.commit()
        return {"message": "User added successfully"}, 201
    
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

            # Use Twilio Verify to check OTP
            verification_check = twilio_client.verify.v2.services(TWILIO_VERIFY_SID) \
                .verification_checks \
                .create(to=phone_number, code=otp_code)
             
            if verification_check.status == "approved":
                res = {"phone_number": phone_number, "status": "approved"}
            else:
                res = {"phone_number": phone_number, "status": "denied"}

        except Exception as e:
            res = {"phone_number": phone_number, "status": "error", "message": str(e)}
            print(f"[DEBUG] Error during OTP verification for {phone_number}: {e}")

        # Publish the verification response
        response_topic = f"door/otp/response/{phone_number}"
        mqtt.publish(response_topic, json.dumps(res), qos=1)
        print(f"[DEBUG] Published OTP verification result to {response_topic}")
        
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
api.add_resource(DoorEntryAPI, "/door-entry")
api.add_resource(UserManagementAPI, "/users")
api.add_resource(UpdateUserNameAPI, "/update-user-name")

if __name__ == '__main__':
    try:
        app.run(debug=True, use_reloader=False)
    except Exception as e:
        print(f"[ERROR] Failed to start application: {e}")



