from flask import Flask
from dotenv import load_dotenv
import os
import atexit
import logging
import json

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

# Set up upload folder for face recognition debug frames
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

# Ensure proper permissions on the uploads directory
try:
    os.chmod(uploads_dir, 0o755)  # rwxr-xr-x
    # Create initial empty files to avoid 404 errors
    debug_frame_path = os.path.join(uploads_dir, 'debug_frame.jpg')
    debug_data_path = os.path.join(uploads_dir, 'debug_data.json')
    
    # Create an empty debug frame (black image) if it doesn't exist
    if not os.path.exists(debug_frame_path):
        import cv2
        import numpy as np
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Initializing camera...", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imwrite(debug_frame_path, img)
    
    # Create empty debug data if it doesn't exist
    if not os.path.exists(debug_data_path):
        with open(debug_data_path, 'w') as f:
            json.dump({
                "face_locations": [],
                "ear_values": {
                    "left_ear": 0,
                    "right_ear": 0,
                    "avg_ear": 0,
                    "blink_detected": False
                }
            }, f)
    
    logging.info(f"Initialized upload directory: {uploads_dir}")
except Exception as e:
    logging.warning(f"Could not set permissions on uploads directory: {e}")

app.config['UPLOAD_FOLDER'] = uploads_dir

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
