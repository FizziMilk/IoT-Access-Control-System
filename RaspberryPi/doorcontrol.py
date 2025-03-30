from flask import Flask, render_template, request, redirect, url_for, flash
import RPi.GPIO as GPIO
import time
import requests
from dotenv import load_dotenv
from datetime import datetime
import threading
import json
from flask_mqtt import Mqtt
import ssl
import os
import atexit
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context


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
session.mount("https://", HTTPAdapter(max_retries=3, pool_connections=10, pool_maxsize=100))
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
app.config['MQTT_BROKER_URL'] = os.getenv("MQTT_BROKER")
app.config['MQTT_BROKER_PORT'] = int(os.getenv("MQTT_PORT"))
app.config['MQTT_TLS_VERSION'] = ssl.PROTOCOL_TLSv1_2
app.config['MQTT_TLS_ENABLED'] = True
app.config['MQTT_TLS_CA_CERTS'] = os.getenv("CA_CERT")

required_env_vars = ["MQTT_BROKER", "MQTT_PORT", "CA_CERT", "BACKEND_URL"]
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

mqtt = Mqtt(app)

def unlock_door(duration=10):
    GPIO.output(DOOR_PIN, GPIO.HIGH)  # Activate door relay
    print("Door unlocked")
    time.sleep(duration)  # Keep the door unlocked for the specified duration
    GPIO.output(DOOR_PIN, GPIO.LOW)  # Deactivate door relay
    print("Door locked")

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
    
def check_schedule():
    while True:
        now = datetime.now()
        weekday = now.strftime("%A")  # e.g., "Monday"

        if weekday in schedule:
            entry = schedule[weekday]
            open_time_str = entry.get("open_time")
            close_time_str = entry.get("close_time")
            force_unlocked = entry.get("forceUnlocked", False)

            if open_time_str and close_time_str:
                try:
                    # Parse times from "HH:MM" strings
                    open_time = datetime.strptime(open_time_str, "%H:%M").time()
                    close_time = datetime.strptime(close_time_str, "%H:%M").time()
                except ValueError as ve:
                    print(f"Time format error: {ve}")
                    time.sleep(60)
                    continue
                current_time = now.time().replace(second=0, microsecond=0)

                # If forced unlocked, ensure door is unlocked
                if force_unlocked:
                    GPIO.output(DOOR_PIN, GPIO.HIGH)
                    print(f"Force Unlocked: {weekday} - Door unlocked")
                # Else, if current time falls within open and close times, unlock
                elif open_time <= current_time <= close_time:
                    GPIO.output(DOOR_PIN, GPIO.HIGH)
                    print(f"{weekday}: Door unlocked at {current_time.strftime('%H:%M')}")
                else:
                    GPIO.output(DOOR_PIN, GPIO.LOW)
                    print(f"{weekday}: Door locked at {current_time.strftime('%H:%M')}")
        else:
            print(f"No schedule entry for {weekday}")

        time.sleep(60)  # Check every minute

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
def update_schedule():
    global schedule
    try:
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
    
@app.route('/door-entry', methods=['GET','POST'])
def door_entry():
    if request.method == "POST":
        phone_number = request.form['phone_number']
        try:
            ## This is semi-okay, will need to be hidden behind https or implemented with MQTT
            # May reveal the backend IP address
            resp = session.post(f"{BACKEND_URL}/door-entry",json={"phone_number":phone_number})
            print("Status code:", resp.status_code)
            print("Response text:",resp.text)
            data = resp.json()
        except Exception as e:
            flash("Error connecting to backend.", "danger")
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
    # Start the schedule checking in a separate thread
    schedule_thread = threading.Thread(target=check_schedule)
    schedule_thread.daemon = True
    schedule_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)
