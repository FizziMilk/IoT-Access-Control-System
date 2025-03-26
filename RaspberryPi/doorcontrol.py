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

# Load environment variables
load_dotenv()

# GPIO setup
DOOR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(DOOR_PIN, GPIO.OUT)

BACKEND_URL = os.getenv("BACKEND_URL")

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

mqtt = Mqtt(app)

def unlock_door(duration=10):
    GPIO.output(DOOR_PIN, GPIO.HIGH)  # Activate door relay
    print("Door unlocked")
    time.sleep(duration)  # Keep the door unlocked for the specified duration
    GPIO.output(DOOR_PIN, GPIO.LOW)  # Deactivate door relay
    print("Door locked")

def verify_otp_mqtt(phone_number, otp_code):
    # Create an event to wait for the response
    event = threading.Event()
    pending_verifications[phone_number] = {"event": event, "result": None}

    # Create payload and print it for debugging
    payload = json.dumps({"phone_number": phone_number, "otp_code": otp_code})
    print(f"[DEBUG] Publishing OTP verification request: {payload}")

    # Publish the verification request to MQTT topic
    mqtt.publish("door/otp/verify", payload, qos=1)
    
    # Wait for a response (timeout after 30 seconds)
    print(f"[DEBUG] Waiting for OTP verification response for {phone_number}")
    if not event.wait(timeout=30):
        print(f"[DEBUG] Timeout waiting for OTP verification response for {phone_number}")
        pending_verifications.pop(phone_number, None)
        return {"status": "error", "message": "Verification timeout"}
    
    # Retrieve and print the result before returning it
    result = pending_verifications.pop(phone_number)["result"]
    print(f"[DEBUG] Received OTP verification response for {phone_number}: {result}")
    return result

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
    response = verify_otp_mqtt(phone_number, otp_code)
    if response.get("status") == "approved":
        unlock_door()
        flash("OTP verified, door unlocked", "success")
        time.sleep(10)
        return redirect(url_for('index'))
    else:
        flash(response.get("message", "Invalid OTP"), "danger")
        return render_template("otp.html", phone_number = phone_number)

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
            resp = requests.post(f"{BACKEND_URL}/door-entry",json={"phone_number":phone_number})
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
        resp = requests.post(f"{BACKEND_URL}/update-user-name", json={"phone_number": phone_number, "name": name})
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
        print("Connected successfully")
        mqtt.subscribe("door/commands")
        mqtt.subscribe("door/schedule")
        mqtt.subscribe("door/otp")
    else:
        print(f"Connection failed with code {rc}")

@mqtt.on_message()
def handle_mqtt_message(client, userdata, message):
    global schedule
    topic = message.topic
    payload = message.payload.decode()
    print(f"Received message on {topic}: {payload}")
    # Handle commands for the door
    if topic == "door/commands":
        if payload == "unlock_door":
            unlock_door()  # Use the unlock_door function
        elif payload == "lock_door":
            GPIO.output(DOOR_PIN, GPIO.LOW)  # Deactivate door relay
            print("Door locked")
    # Update schedule
    elif topic == "door/schedule":
        try:
            # Parse the raw JSON received (expected as an array)
            raw = json.loads(payload)
            # Convert array to a dict keyed by day
            new_schedule = {}
            for entry in raw:
                day = entry.get("day")
                if day:
                    new_schedule[day] = entry
            schedule = new_schedule
            print(f"Updated schedule: {schedule}")
        except json.JSONDecodeError:
            print("Invalid schedule format received")
    # OTP verification responses
    elif topic.startswith("door/otp/response/"):
        try:
            response_payload = json.loads(payload)
            phone_number = response_payload.get("phone_number")
            print(f"OTP response for {phone_number}: {response_payload}")
            if phone_number in pending_verifications:
                pending_verifications[phone_number]["result"] = response_payload
                pending_verifications[phone_number]["event"].set()
        except Exception as e:
            print(f"Error processing OTP response: {e}")

if __name__ == '__main__':
    # Start the schedule checking in a separate thread
    schedule_thread = threading.Thread(target=check_schedule)
    schedule_thread.daemon = True
    schedule_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)
