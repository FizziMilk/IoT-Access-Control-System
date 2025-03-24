from flask import Flask, render_template, request, redirect, url_for, flash
import RPi.GPIO as GPIO
import time
import requests
from dotenv import load_dotenv
import os
from datetime import datetime
import threading
import json
from flask_mqtt import Mqtt

# Load environment variables
load_dotenv()

# GPIO setup
DOOR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(DOOR_PIN, GPIO.OUT)

#Get MQTT broker IP from .env
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT")) #Default to 8883 if not set
MQTT_COMMAND_TOPIC = "door/commands"
MQTT_SCHEDULE_TOPIC = "door/schedule"
# Backend URL
BACKEND_URL = os.getenv("BACKEND_URL")


# Schedule storage (we'll store it as a dict keyed by day)
schedule = {}

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Flask-MQTT Configuration
app.config['MQTT_BROKER_URL'] = os.getenv("MQTT_BROKER_URL")
app.config['MQTT_BROKER_PORT'] = 8883
app.config['MQTT_TLS_VERSION'] = ssl.PROTOCOL_TLSv1_2
app.config['MQTT_TLS_ENABLED'] = True
app.config['MQTT_TLS_CA_CERTS'] = '/etc/mosquitto/ca_certificates/ca.crt'

mqtt = Mqtt(app)

def unlock_door(duration=10):
    GPIO.output(DOOR_PIN, GPIO.HIGH)  # Activate door relay
    print("Door unlocked")
    time.sleep(duration)  # Keep the door unlocked for the specified duration
    GPIO.output(DOOR_PIN, GPIO.LOW)  # Deactivate door relay
    print("Door locked")

def verify_otp(phone_number, otp_code):
    try:
        response = requests.post(f"{BACKEND_URL}/check-verification-RPI", json={
            "phone_number": phone_number,
            "otp_code": otp_code
        })
        return response.json()
    except Exception as e:
        print(f"Error verifying OTP: {e}")
        return {"status": "error", "message": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/verify', methods=['POST'])
def verify():
    phone_number = request.form['phone_number']
    otp_code = request.form['otp_code']
    response = verify_otp(phone_number, otp_code)
    if response.get("status") == "approved":
        unlock_door()
        flash("OTP verified, door unlocked", "success")
    else:
        flash("Invalid OTP", "danger")
    return redirect(url_for('index'))

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
                    open_time = datetime.strptime(open_time_str, "%H:%M")
                    close_time = datetime.strptime(close_time_str, "%H:%M")
                except ValueError as ve:
                    print(f"Time format error: {ve}")
                    time.sleep(60)
                    continue
                current_time = now.replace(second=0, microsecond=0)

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
    command = message.payload.decode()
    print(f"Received message on {topic}: {command}")
    # Handle commands for the door
    if topic == "door/commands":
        if command == "unlock_door":
            unlock_door()  # Use the unlock_door function
        elif command == "lock_door":
            GPIO.output(DOOR_PIN, GPIO.LOW)  # Deactivate door relay
            print("Door locked")
    # Update schedule
    elif topic == "door/schedule":
        try:
            # Parse the raw JSON received (expected as an array)
            raw = json.loads(command)
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

if __name__ == '__main__':
    # Start the schedule checking in a separate thread
    schedule_thread = threading.Thread(target=check_schedule)
    schedule_thread.daemon = True
    schedule_thread.start()

    app.run(host='0.0.0.0', port=5000, debug=True)
