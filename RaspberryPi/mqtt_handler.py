import json
from flask_mqtt import Mqtt
import ssl
import os
import requests

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
        self.app.config['MQTT_TLS_CA_CERTS'] = os.getenv("CA_CERT")

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
