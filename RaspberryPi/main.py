"""
Main entry point for the Raspberry Pi facial recognition access control system.
This file initializes the Flask app, sets up routes, and configures the components.
"""
from flask import Flask
from dotenv import load_dotenv
import os
import atexit
import logging
import cv2
import time
import subprocess
import platform

# Import local modules
from door_controller import DoorController
from mqtt_handler import MQTTHandler
from routes import setup_routes
from utils import create_backend_session
from camera_config import CAMERA_INDEX

# Configure logging once at the application level
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])

logger = logging.getLogger("MainApp")

# Load environment variables
load_dotenv()

def cleanup_camera_resources():
    """
    Ensure camera resources are clean at startup and create necessary directories.
    This function attempts to reset any stale camera resources and ensures debug frames
    directory exists for storing face recognition visualization.
    """
    logger.info("Initializing and cleaning up camera resources")
    
    # Try to release any existing OpenCV camera resources
    try:
        # Create and immediately release a camera to reset any stale resources
        camera = cv2.VideoCapture(CAMERA_INDEX)
        time.sleep(1.0)
        camera.release()
        logger.info(f"Successfully reset camera at index {CAMERA_INDEX}")
    except Exception as e:
        logger.error(f"Error resetting camera: {e}")
    
    # Destroy any OpenCV windows
    try:
        cv2.destroyAllWindows()
        cv2.waitKey(1)
    except Exception as e:
        logger.error(f"Error destroying windows: {e}")
    
    # Ensure debug frame directory exists
    debug_frames_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static', 'debug_frames')
    os.makedirs(debug_frames_dir, exist_ok=True)
    logger.info(f"Created debug frames directory: {debug_frames_dir}")


# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "supersecretkey")

# Run camera cleanup at startup
cleanup_camera_resources()

# Set up upload folder for face recognition debug frames
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)
app.config['UPLOAD_FOLDER'] = uploads_dir

# Check required environment variables
required_env_vars = ["MQTT_BROKER_URL", "MQTT_PORT", "CA_CERT", "BACKEND_URL"]
for var in required_env_vars:
    if not os.getenv(var):
        logger.error(f"Missing required environment variable: {var}")
        raise EnvironmentError(f"Missing required environment variable: {var}")

# Initialize components
door_controller = DoorController()
session, backend_url = create_backend_session()
mqtt_handler = MQTTHandler(app, door_controller)
mqtt_handler.start()  # Start MQTT handler to subscribe to topics including door/schedule

# Setup routes
logger.info("Setting up application routes")
app_with_routes = setup_routes(app, door_controller, mqtt_handler, session, backend_url)

# Register cleanup on exit
atexit.register(door_controller.cleanup)
logger.info("Door controller cleanup registered with atexit")

if __name__ == '__main__':
    logger.info("Starting Flask application on port 5000")
    app.run(host='0.0.0.0', port=5000, debug=False) 
