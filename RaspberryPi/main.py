from flask import Flask
from dotenv import load_dotenv
import os
import atexit
import logging

from door_controller import DoorController
from mqtt_handler import MQTTHandler
from routes import setup_routes
from utils import create_backend_session

# Configure logging once at the application level
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Check required environment variables
required_env_vars = ["MQTT_BROKER_URL", "MQTT_PORT", "CA_CERT", "BACKEND_URL"]
for var in required_env_vars:
    if not os.getenv(var):
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Initialize components
door_controller = DoorController()
session, backend_url = create_backend_session()
mqtt_handler = MQTTHandler(app, door_controller)

# Setup routes
setup_routes(app, door_controller, mqtt_handler, session, backend_url)

# Register cleanup
atexit.register(door_controller.cleanup)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False) 
