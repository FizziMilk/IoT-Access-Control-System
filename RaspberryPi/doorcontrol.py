import os
import time
import json
import paho.mqtt.client as mqtt
import RPi.GPIO as GPIO
from dotenv import load_dotenv
import ssl
from datetime import datetime

# Load environment variables
load_dotenv()

# GPIO setup
DOOR_PIN = 17
GPIO.setmode(GPIO.BCM)
GPIO.setup(DOOR_PIN, GPIO.OUT)

# Get MQTT broker IP from .env
MQTT_BROKER = os.getenv("MQTT_BROKER")
MQTT_PORT = int(os.getenv("MQTT_PORT", 8883))  # Default to 8883 if not set
MQTT_COMMAND_TOPIC = "door/commands"
MQTT_SCHEDULE_TOPIC = "door/schedule"

# Schedule storage (we'll store it as a dict keyed by day)
schedule = {}

# Path to certificate
CA_CERT = os.getenv("CA_CERT")

def on_connect(client, userdata, flags, rc, properties):
    if rc == 0:
        print("Connected successfully")
        client.subscribe(MQTT_COMMAND_TOPIC)
        client.subscribe(MQTT_SCHEDULE_TOPIC)
    else:
        print(f"Connection failed with code {rc}")

def on_message(client, userdata, msg):
    global schedule
    topic = msg.topic
    command = msg.payload.decode()
    print(f"Received message on {topic}: {command}")
    # Handle commands for the door
    if topic == MQTT_COMMAND_TOPIC:
        if command == "unlock_door":
            GPIO.output(DOOR_PIN, GPIO.HIGH)  # Activate door relay
            print("Door unlocked")
        elif command == "lock_door":
            GPIO.output(DOOR_PIN, GPIO.LOW)   # Deactivate door relay
            print("Door locked")
    # Update schedule
    elif topic == MQTT_SCHEDULE_TOPIC:
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

def check_schedule():
    # Continuously check schedule and control door
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

# Initialize MQTT client using API version 2
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.on_connect = on_connect
client.on_message = on_message

# Configure TLS
client.tls_set(ca_certs=CA_CERT, tls_version=ssl.PROTOCOL_TLSv1_2)
client.tls_insecure_set(False)

# Connect to the MQTT broker and start loop
client.connect(MQTT_BROKER, MQTT_PORT, 60)

# Start the MQTT loop in a separate thread
import threading
mqtt_thread = threading.Thread(target=client.loop_forever)
mqtt_thread.daemon = True
mqtt_thread.start()

# Start checking the schedule in the main thread
check_schedule()
