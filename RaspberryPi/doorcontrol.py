from flask import Flask, render_template, request, redirect, url_for, flash
import RPi.GPIO as GPIO
import time
import requests
from dotenv import load_dotenv
from datetime import datetime
import json
from flask_mqtt import Mqtt
import ssl
import os
import atexit
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context
from threading import Timer


# Load environment variables
load_dotenv()

# GPIO setup
DOOR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(DOOR_PIN, GPIO.OUT)

BACKEND_URL = os.getenv("BACKEND_URL")
BACKEND_CA_CERT = os.getenv("CA_CERT")

# Create a custom SSL context that trusts the backend cert
ctx = create_urllib3_context()
ctx.load_verify_locations(BACKEND_CA_CERT)

# Create a persistent session
session = requests.Session()
session.mount("http://", HTTPAdapter(max_retries=3, pool_connections=10, pool_maxsize=100))
session.verify = BACKEND_CA_CERT

#Get MQTT broker IP from .env
MQTT_COMMAND_TOPIC = "door/commands"
MQTT_SCHEDULE_TOPIC = "door/schedule"

# Schedule storage
schedule = {}
# Global dictionary to track pending OTP verifications.
pending_verifications = {}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Flask-MQTT Configuration
app.config['MQTT_BROKER'] = os.getenv("MQTT_BROKER_URL")
app.config['MQTT_BROKER_PORT'] = int(os.getenv("MQTT_PORT"))
app.config['MQTT_TLS_VERSION'] = ssl.PROTOCOL_TLSv1_2
app.config['MQTT_TLS_ENABLED'] = True
app.config['MQTT_TLS_CA_CERTS'] = os.getenv("CA_CERT")

required_env_vars = ["MQTT_BROKER_URL", "MQTT_PORT", "CA_CERT", "BACKEND_URL"]
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

mqtt = Mqtt(app)

def unlock_door(duration=10):
    print("[DEBUG] Unlocking door...")
    GPIO.output(DOOR_PIN, GPIO.HIGH)  # Activate door relay
    print("[DEBUG] Door unlocked")
    
    # Send confirmation to backend
    try:
        response = session.post(f"{BACKEND_URL}/door-unlock-confirmation", json={
            "status": "unlocked",
            "method": "schedule",  # This will be overridden by the actual method
            "timestamp": datetime.now().isoformat()
        })
        print(f"[DEBUG] Door unlock confirmation sent: {response.status_code}")
    except Exception as e:
        print(f"[ERROR] Failed to send door unlock confirmation: {e}")

    # Schedule the door to lock after the specified duration
    Timer(duration, lock_door).start()

def lock_door():
    GPIO.output(DOOR_PIN, GPIO.LOW)  # Deactivate door relay
    print("[DEBUG] Door locked")

def verify_otp_rest(phone_number, otp_code):
    try:
        payload = {"phone_number": phone_number, "otp_code": otp_code}
        print(f"[DEBUG] Sending OTP verification request to backend: {payload}")
        
        # Send the request to the backend
        response = session.post(f"{BACKEND_URL}/check-verification-RPI", json=payload)
        print(f"[DEBUG] Backend response: {response.status_code} - {response.text}")
        
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "message": response.json().get("error", "Unknown error")}
    except Exception as e:
        print(f"[DEBUG] Error in verify_otp_rest: {str(e)}")
        return {"status": "error", "message": str(e)}

## Flask Routes

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    phone_number = request.form['phone_number']
    otp_code = request.form['otp_code']
    response = verify_otp_rest(phone_number, otp_code)
    if response.get("status") == "approved":
        unlock_door()
        # Send OTP-specific confirmation
        try:
            session.post(f"{BACKEND_URL}/door-unlock-confirmation", json={
                "status": "unlocked",
                "method": "SMS OTP",
                "phone_number": phone_number,
                "timestamp": datetime.now().isoformat()
            })
        except Exception as e:
            print(f"[ERROR] Failed to send OTP unlock confirmation: {e}")
        flash("OTP verified, door unlocked", "success")
        time.sleep(10)
        return redirect(url_for('index'))
    elif response.get("status") == "error":
        flash(response.get("message", "An error occurred during verification"), "danger")
        return render_template("otp.html", phone_number=phone_number)
    else:
        flash("Invalid OTP or unexpected response", "danger")
        return render_template("otp.html", phone_number=phone_number)

@app.route('/update_schedule', methods=['POST'])
def update_schedule(data = None):
    global schedule
    try:
        if data is None:
            data = request.get_json()

        # Convert array to a dict keyed by day
        new_schedule = {}
        for entry in data:
            day = entry.get("day")
            if day:
                new_schedule[day] = entry
        schedule = new_schedule
        return {"status": "success"}, 200
    except Exception as e:
        print(f"Error updating schedule: {e}")
        return {"status": "error", "message": str(e)}, 500
    
@app.route('/door-entry', methods=['GET', 'POST'])
def door_entry():
    if request.method == "POST":
        now = datetime.now()
        weekday = now.strftime("%A")

        # Check if the current time is within the schedule
        if weekday in schedule:
            entry = schedule[weekday]
            print(f"[DEBUG] Schedule entry for {weekday}: {entry}")
            open_time_str = entry.get("open_time")
            close_time_str = entry.get("close_time")
            force_unlocked = entry.get("forceUnlocked", False)

            if force_unlocked:
                print("[DEBUG] Force unlock is enabled. Unlocking door.")
                unlock_door()
                flash("Door unlocked based on schedule.", "success")
                return render_template("door_unlocked.html")  # Render the new page

            if open_time_str and close_time_str:
                try:
                    open_time = datetime.strptime(open_time_str, "%H:%M").time()
                    close_time = datetime.strptime(close_time_str, "%H:%M").time()
                except ValueError as ve:
                    flash("Schedule time format error.", "danger")
                    print(f"[DEBUG] Schedule time format error: {ve}")
                    return redirect(url_for("door_entry"))

                current_time = now.time().replace(second=0, microsecond=0)

                # If forced unlocked or within schedule, unlock the door
                if force_unlocked or (open_time <= current_time <= close_time):
                    unlock_door()
                    flash("Door unlocked based on schedule.", "success")
                    return render_template("door_unlocked.html")  # Render the new page

        # Otherwise proceed with OTP verification
        phone_number = request.form.get('phone_number')
        if not phone_number:
            flash("Phone number is required for verification.", "danger")
            return redirect(url_for("door_entry"))

        try:
            resp = session.post(f"{BACKEND_URL}/door-entry", json={"phone_number": phone_number})
            print(f"[DEBUG] Backend response: {resp.status_code} - {resp.text}")
            data = resp.json()
        except Exception as e:
            flash("Error connecting to backend.", "danger")
            print(f"[DEBUG] Error connecting to backend: {e}")
            return redirect(url_for("door_entry"))

        if data.get("status") == "OTP sent":
            flash("OTP sent to your phone. Please enter the OTP.", "success")
            return render_template("otp.html", phone_number=phone_number)
        elif data.get("status") == "pending":
            flash(data.get("message", "Access pending."), "warning")
            return render_template("pending.html", phone_number=phone_number)
        else:
            flash(data.get("error", "An error occurred."), "danger")
            return redirect(url_for("door_entry"))

    return render_template("door_entry.html")

@app.route('/update-name', methods=['POST'])
def update_name():
    phone_number = request.form.get('phone_number')
    name = request.form.get('name')
    if not phone_number or not name:
        flash("Name and phone number are required.", "danger")
        return redirect(url_for("door_entry"))
    try:
        resp = session.post(f"{BACKEND_URL}/update-user-name", json={"phone_number": phone_number, "name": name})
        data = resp.json()
        if data.get("status") == "success":
            flash("Name updated succesffuly. Please wait for admin approval.", "info")
        else:
            flash(data.get("error", "Error updating name"), "danger")
    except Exception as e:
        flash("Error connecting to backend.", "danger")
    return redirect(url_for("door_entry"))
    
## MQTT Handling
    
@mqtt.on_connect()
def handle_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[DEBUG] Connected to MQTT broker with result code {rc}")
        mqtt.subscribe([
            ("door/commands", 1),
            ("door/schedule", 1),
            (f"door/otp/response/+", 1)  # Use + wildcard for phone numbers
        ])
        print("[DEBUG] Subscribed to all necessary topics")

@mqtt.on_message()
def handle_mqtt_message(client,userdata,message):
    print(f"[DEBUG] Received message on topic: {message.topic}")
    try:
        if message.topic.startswith("door/otp/response/"):
            # Extract phone number from topic
            phone_number = message.topic.split('/')[-1]
            if phone_number in pending_verifications:
                payload = json.loads(message.payload.decode())
                print(f"[DEBUG] OTP response payload: {payload}")
                pending_verifications[phone_number]["result"] = payload
                pending_verifications[phone_number]["event"].set()

        elif message.topic == "door/commands":
            command = message.payload.decode()
            if command == "unlock_door":
                unlock_door()
            elif command == "lock_door":
                GPIO.output(DOOR_PIN,GPIO.LOW)

        elif message.topic == "door/schedule":
            schedule_data = json.loads(message.payload.decode())
            update_schedule(schedule_data)

    except Exception as e:
        print(f"[DEBUG] Error handling MQTT message: {str(e)}")

@mqtt.on_subscribe()
def handle_subscribe(client, userdata, mid, granted_qos):
    print(f"[DEBUG] Subscribed to topic with mid: {mid}, granted QoS: {granted_qos}")

def cleanup_gpio():
    GPIO.cleanup()
    print("[DEBUG] GPIO cleanup completed")

atexit.register(cleanup_gpio)
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
