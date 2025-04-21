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
        
        # Let's Encrypt certificates for MQTT
        self.app.config['MQTT_TLS_CA_CERTS'] = os.getenv("CA_CERT")
        
        # Client certificate authentication for MQTT
        self.app.config['MQTT_TLS_CERTFILE'] = os.getenv("CLIENT_CERT_PATH")
        self.app.config['MQTT_TLS_KEYFILE'] = os.getenv("CLIENT_KEY_PATH")
        
        # If using Let's Encrypt certificates, we don't need to skip verification
        # Only enable this if your MQTT broker is using self-signed certificates
        # self.app.config['MQTT_TLS_INSECURE'] = True

        self.mqtt = Mqtt(self.app)
        self.setup_mqtt_handlers()

    def setup_mqtt_handlers(self):
        @self.mqtt.on_connect()
        def handle_connect(client, userdata, flags, rc):
            if rc == 0:
                print(f"[DEBUG] Connected to MQTT broker with result code {rc}")
                print(f"[DEBUG] Attempting to subscribe to door/commands and other topics")
                self.mqtt.subscribe([
                    ("door/commands", 1),
                    ("door/schedule", 1),
                    (f"door/otp/response/+", 1),
                    ("door/otp/verify", 1)
                ])
                print("[DEBUG] Subscribed to all necessary topics")
            else:
                print(f"[ERROR] Failed to connect to MQTT broker with code {rc}")

        @self.mqtt.on_message()
        def handle_mqtt_message(client, userdata, message):
            print(f"[DEBUG] Received message on topic: {message.topic}")
            print(f"[DEBUG] Message payload: {message.payload}")
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
                    print(f"[DEBUG] Received schedule data: {schedule_data}")
                    self.update_schedule(schedule_data)
                    print(f"[DEBUG] Updated schedule: {self.schedule}")

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
            print(f"[DEBUG] MQTT client active subscriptions: {self.mqtt.topics}")

        # Register explicit callback for door commands to ensure they're captured
        @self.mqtt.on_topic('door/commands')
        def handle_door_command(client, userdata, message):
            print(f"[DEBUG] Direct door command received: {message.payload.decode()}")
            command = message.payload.decode()
            
            try:
                if command == "unlock_door":
                    print(f"[DEBUG] Explicitly unlocking door from dedicated handler")
                    self.door_controller.unlock_door()
                    print(f"[DEBUG] Door unlock command executed successfully (dedicated handler)")
                elif command == "lock_door":
                    print(f"[DEBUG] Explicitly locking door from dedicated handler")
                    self.door_controller.lock_door()
                    print(f"[DEBUG] Door lock command executed successfully (dedicated handler)")
                else:
                    print(f"[WARNING] Unknown door command received in dedicated handler: {command}")
            except Exception as e:
                print(f"[ERROR] Error executing door command in dedicated handler: {e}")

    def update_schedule(self, data):
        try:
            if not isinstance(data, list):
                return {"status": "error", "message": "Schedule data must be a list"}, 400
                
            new_schedule = {}
            processed_days = []
            
            for entry in data:
                # Skip non-dictionary entries
                if not isinstance(entry, dict):
                    logger.warning(f"Skipping invalid schedule entry: {entry}")
                    continue
                    
                day = entry.get("day")
                if not day:
                    logger.warning(f"Skipping schedule entry without day: {entry}")
                    continue
                    
                # Check for duplicate days
                if day in processed_days:
                    logger.warning(f"Duplicate day entry for {day}. Using last entry.")
                
                # Process the entry
                processed_days.append(day)
                new_schedule[day] = entry
                
                # Log the entry details
                open_time = entry.get("open_time", "None")
                close_time = entry.get("close_time", "None")
                force_unlocked = entry.get("forceUnlocked", False)
                logger.info(f"Schedule entry for {day}: Open={open_time}, Close={close_time}, Force={force_unlocked}")
                
            # Update the schedule
            self.schedule = new_schedule
            logger.info(f"Schedule updated with {len(new_schedule)} days")
            return {"status": "success", "message": f"Schedule updated with {len(new_schedule)} days"}, 200
        except Exception as e:
            logger.error(f"Error updating schedule: {e}")
            return {"status": "error", "message": str(e)}, 500

def setup_mqtt_client():
    """
    Set up and configure the MQTT client with TLS certificate support
    """
    # Get config from environment or use defaults
    mqtt_broker = os.environ.get("MQTT_BROKER_URL", "mqtt.example.com")
    mqtt_port = int(os.environ.get("MQTT_PORT", 8883))
    client_id = os.environ.get("MQTT_CLIENT_ID", f"raspberry-pi-{socket.gethostname()}")
    
    # Certificate paths
    ca_cert = os.environ.get("CA_CERT", "/etc/ssl/certs/letsencrypt-fullchain.pem")
    client_cert = os.environ.get("CLIENT_CERT_PATH", None)
    client_key = os.environ.get("CLIENT_KEY_PATH", None)
    
    logger.info(f"Setting up MQTT client with ID: {client_id}")
    logger.info(f"MQTT broker: {mqtt_broker}:{mqtt_port}")
    logger.info(f"CA cert path: {ca_cert}")
    
    # Create MQTT client
    client = mqtt.Client(client_id=client_id, protocol=mqtt.MQTTv5)
    
    # Set up callbacks
    client.on_connect = on_connect
    client.on_disconnect = on_disconnect
    client.on_message = on_message
    
    # Set up TLS
    try:
        tls_context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH, cafile=ca_cert)
        
        # If client cert and key are provided, set up mutual TLS
        if client_cert and client_key:
            logger.info(f"Using client cert: {client_cert} and key: {client_key}")
            try:
                tls_context.load_cert_chain(client_cert, client_key)
            except PermissionError as pe:
                logger.warning(f"Permission denied loading client cert: {pe}. Skipping client cert")
        
        # Enforce hostname verification
        tls_context.check_hostname = True
        
        client.tls_set_context(tls_context)
        logger.info("TLS configuration completed successfully")
    except Exception as e:
        logger.error(f"Failed to set up TLS: {e}. Continuing without TLS")
        # Proceed without TLS
    
    # Set up auth if provided
    username = os.environ.get("MQTT_USERNAME")
    password = os.environ.get("MQTT_PASSWORD")
    if username and password:
        logger.info(f"Using MQTT authentication with username: {username}")
        client.username_pw_set(username, password)
    
    return client 

class MQTTPahoHandler:
    def __init__(self, mqtt_broker=None, mqtt_port=None):
        """Initialize the MQTT handler"""
        self.client = None
        self.connected = False
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.stop_event = Event()
        self.mqtt_thread = None
        
    def start(self):
        """Start the MQTT client and connect to the broker"""
        try:
            self.client = setup_mqtt_client()
            
            # Start the client in a separate thread
            self.mqtt_thread = Thread(target=self._mqtt_loop, daemon=True)
            self.mqtt_thread.start()
            
            # Connect to the broker
            if self.mqtt_broker and self.mqtt_port:
                self.client.connect(self.mqtt_broker, self.mqtt_port, keepalive=60)
            else:
                # Use values from environment variables
                broker = os.environ.get("MQTT_BROKER", "mqtt.example.com")
                port = int(os.environ.get("MQTT_PORT", 8883))
                self.client.connect(broker, port, keepalive=60)
                
            logger.info("MQTT handler started")
            return True
        except Exception as e:
            logger.error(f"Failed to start MQTT handler: {e}")
            return False
    
    def _mqtt_loop(self):
        """MQTT client loop running in a separate thread"""
        while not self.stop_event.is_set():
            try:
                self.client.loop(timeout=1.0)
            except Exception as e:
                logger.error(f"Error in MQTT loop: {e}")
                # Try to reconnect after a delay
                self.stop_event.wait(5.0)
                if not self.stop_event.is_set():
                    try:
                        self.client.reconnect()
                    except Exception as e:
                        logger.error(f"Failed to reconnect: {e}")
    
    def stop(self):
        """Stop the MQTT client and clean up resources"""
        if self.client:
            self.stop_event.set()
            self.client.disconnect()
            if self.mqtt_thread:
                self.mqtt_thread.join(timeout=2.0)
            logger.info("MQTT handler stopped")
    
    def publish(self, topic, payload, qos=1, retain=False):
        """Publish a message to the MQTT broker"""
        if not self.client:
            logger.error("MQTT client not initialized")
            return False
            
        try:
            if isinstance(payload, dict):
                payload = json.dumps(payload)
            
            result = self.client.publish(topic, payload, qos=qos, retain=retain)
            if result.rc == mqtt.MQTT_ERR_SUCCESS:
                logger.info(f"Published message to topic {topic}")
                return True
            else:
                logger.error(f"Failed to publish message: {result.rc}")
                return False
        except Exception as e:
            logger.error(f"Error publishing message: {e}")
            return False 