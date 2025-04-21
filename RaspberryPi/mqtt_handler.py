import json
from flask_mqtt import Mqtt
import ssl
import os
import requests
import socket
import paho.mqtt.client as mqtt
import logging
from threading import Thread, Event

logger = logging.getLogger(__name__)

# Define callback functions for MQTT events
def on_connect(client, userdata, flags, rc, properties=None):
    """Callback for when the client connects to the MQTT broker"""
    if rc == 0:
        logger.info("Connected to MQTT broker successfully")
        # Subscribe to relevant topics
        client.subscribe("door/commands", qos=1)
        client.subscribe("door/status", qos=1)
    else:
        logger.error(f"Failed to connect to MQTT broker with code {rc}")

def on_disconnect(client, userdata, rc, properties=None):
    """Callback for when the client disconnects from the MQTT broker"""
    if rc != 0:
        logger.warning(f"Unexpected disconnection from MQTT broker: {rc}")
    else:
        logger.info("Disconnected from MQTT broker")

def on_message(client, userdata, msg, properties=None):
    """Callback for when a message is received from the MQTT broker"""
    logger.info(f"Received message on topic {msg.topic}")
    try:
        payload = json.loads(msg.payload.decode())
        logger.info(f"Message payload: {payload}")
        
        # Process message based on topic
        if msg.topic == "door/commands":
            process_door_command(payload)
        elif msg.topic == "door/status":
            logger.info(f"Door status update: {payload}")
    except json.JSONDecodeError:
        logger.error(f"Failed to decode message payload: {msg.payload}")
    except Exception as e:
        logger.error(f"Error processing message: {e}")

def process_door_command(payload):
    """Process door command messages"""
    if 'command' in payload:
        command = payload['command']
        logger.info(f"Processing door command: {command}")
        # Implement door command handling logic here
        # This would integrate with the door control system

class MQTTHandler:
    def __init__(self, app, door_controller):
        self.app = app
        self.door_controller = door_controller
        self.schedule = {}
        self.pending_verifications = {}
        
        # MQTT Configuration
        self.app.config['MQTT_BROKER_URL'] = os.getenv("MQTT_BROKER_URL")
        self.app.config['MQTT_BROKER_PORT'] = int(os.getenv("MQTT_PORT"))
        self.app.config['MQTT_TLS_VERSION'] = ssl.PROTOCOL_TLSv1_2
        self.app.config['MQTT_TLS_ENABLED'] = True
        
        # Let's Encrypt certificates for MQTT (trim whitespace)
        ca_cert_env = os.getenv("CA_CERT", "").strip()
        self.app.config['MQTT_TLS_CA_CERTS'] = ca_cert_env if ca_cert_env else None
        
        # Client certificate authentication for MQTT, only if files are readable (trim whitespace)
        certfile_path = os.getenv("CLIENT_CERT_PATH", "").strip()
        keyfile_path = os.getenv("CLIENT_KEY_PATH", "").strip()
        if certfile_path and keyfile_path and os.path.isfile(certfile_path) and os.access(certfile_path, os.R_OK) and os.path.isfile(keyfile_path) and os.access(keyfile_path, os.R_OK):
            self.app.config['MQTT_TLS_CERTFILE'] = certfile_path
            self.app.config['MQTT_TLS_KEYFILE'] = keyfile_path
        else:
            logger.warning("MQTT client cert or key not accessible; skipping mutual TLS")

        # If using Let's Encrypt certificates, we don't need to skip verification
        # Only enable this if your MQTT broker is using self-signed certificates
        # self.app.config['MQTT_TLS_INSECURE'] = True

        # Initialize Flask-MQTT client, retry without client cert if permission denied
        try:
            self.mqtt = Mqtt(self.app)
        except PermissionError as e:
            logger.error(f"Permission denied loading MQTT client cert: {e}. Disabling mutual TLS and retrying.")
            # Remove client cert config and retry
            self.app.config.pop('MQTT_TLS_CERTFILE', None)
            self.app.config.pop('MQTT_TLS_KEYFILE', None)
            self.mqtt = Mqtt(self.app)
        self.setup_mqtt_handlers()

    def setup_mqtt_handlers(self):
        @self.mqtt.on_connect()
        def handle_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"[DEBUG] Connected to MQTT broker with result code {rc}")
                self.mqtt.subscribe([
                    ("door/commands", 1),
                    ("door/schedule", 1),
                    (f"door/otp/response/+", 1),
                    ("door/otp/verify", 1)
                ])
                print("[DEBUG] Subscribed to all necessary topics")

        @self.mqtt.on_message()
        def handle_mqtt_message(client, userdata, message):
            print(f"[DEBUG] Received message on topic: {message.topic}")
            try:
                if message.topic.startswith("door/otp/response/"):
                    phone_number = message.topic.split('/')[-1]
                    if phone_number in self.pending_verifications:
                        payload = json.loads(message.payload.decode())
                        print(f"[DEBUG] OTP response payload: {payload}")
                        self.pending_verifications[phone_number]["result"] = payload
                        self.pending_verifications[phone_number]["event"].set()

                elif message.topic == "door/commands":
                    command = message.payload.decode()
                    print(f"[DEBUG] Received door command: {command}")
                    if command == "unlock_door":
                        print(f"[DEBUG] Executing unlock_door command")
                        self.door_controller.unlock_door()
                        print(f"[DEBUG] Door unlock command executed successfully")
                    elif command == "lock_door":
                        print(f"[DEBUG] Executing lock_door command")
                        self.door_controller.lock_door()
                        print(f"[DEBUG] Door lock command executed successfully")
                    else:
                        print(f"[WARNING] Unknown door command received: {command}")

                elif message.topic == "door/schedule":
                    schedule_data = json.loads(message.payload.decode())
                    self.update_schedule(schedule_data)

                elif message.topic == "door/otp/verify":
                    try:
                        payload = json.loads(message.payload.decode())
                        phone_number = payload.get("phone_number")
                        otp_code = payload.get("otp_code")
                        print(f"[DEBUG] Received OTP verification request for {phone_number}")

                        # Send verification request to backend
                        response = requests.post(
                            f"{os.getenv('BACKEND_URL')}/check-verification-RPI",
                            json={"phone_number": phone_number, "otp_code": otp_code}
                        )
                        response_data = response.json()
                        print(f"[DEBUG] Backend verification response: {response_data}")

                        # Handle different access scenarios
                        if response_data.get("status") == "approved":
                            # Door is unlocked (either globally or through verification)
                            self.door_controller.unlock_door()
                            # Publish success response
                            client.publish(f"door/otp/response/{phone_number}", json.dumps({
                                "phone_number": phone_number,
                                "status": "approved",
                                "message": response_data.get("message", "Door unlocked")
                            }))
                        else:
                            # Access denied
                            client.publish(f"door/otp/response/{phone_number}", json.dumps({
                                "phone_number": phone_number,
                                "status": "denied",
                                "message": response_data.get("message", "Access denied")
                            }))

                    except Exception as e:
                        print(f"[ERROR] Error handling OTP verification: {e}")
                        client.publish(f"door/otp/response/{phone_number}", json.dumps({
                            "phone_number": phone_number,
                            "status": "error",
                            "message": str(e)
                        }))

            except Exception as e:
                print(f"[DEBUG] Error handling MQTT message: {str(e)}")

        @self.mqtt.on_subscribe()
        def handle_subscribe(client, userdata, mid, granted_qos):
            print(f"[DEBUG] Subscribed to topic with mid: {mid}, granted QoS: {granted_qos}")

    def update_schedule(self, data):
        try:
            new_schedule = {}
            for entry in data:
                day = entry.get("day")
                if day:
                    new_schedule[day] = entry
            self.schedule = new_schedule
            return {"status": "success"}, 200
        except Exception as e:
            print(f"Error updating schedule: {e}")
            return {"status": "error", "message": str(e)}, 500 