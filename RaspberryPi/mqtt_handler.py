import json
from flask_mqtt import Mqtt
import ssl
import os

class MQTTHandler:
    def __init__(self, app, door_controller):
        self.app = app
        self.door_controller = door_controller
        self.schedule = {}
        self.pending_verifications = {}
        
        # MQTT Configuration
        self.app.config['MQTT_BROKER_URL'] = os.getenv("MQTT_BROKER")
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
                    (f"door/otp/response/+", 1)
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
                    if command == "unlock_door":
                        self.door_controller.unlock_door()
                    elif command == "lock_door":
                        self.door_controller.lock_door()

                elif message.topic == "door/schedule":
                    schedule_data = json.loads(message.payload.decode())
                    self.update_schedule(schedule_data)

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