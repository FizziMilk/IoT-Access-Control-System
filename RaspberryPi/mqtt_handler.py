import json
from utils import update_schedule
from gpio import unlock_door, lock_door

def setup_mqtt(mqtt):
    @mqtt.on_connect()
    def handle_connect(client, userdata, flags, rc):
        if rc == 0:
            print("[DEBUG] Connected to MQTT broker")
            mqtt.subscribe("door/commands")
            mqtt.subscribe("door/schedule")

    @mqtt.on_message()
    def handle_message(client, userdata, message):
        print(f"[DEBUG] Received message on topic: {message.topic}")
        if message.topic == "door/commands":
            command = message.payload.decode()
            if command == "unlock_door":
                unlock_door()
            elif command == "lock_door":
                lock_door()
        elif message.topic == "door/schedule":
            schedule_data = json.loads(message.payload.decode())
            update_schedule(schedule_data)