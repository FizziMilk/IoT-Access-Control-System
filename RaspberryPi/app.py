from flask import Flask
from flask_mqtt import Mqtt
from dotenv import load_dotenv
from config import Config
from routes import register_routes
from mqtt_handler import setup_mqtt
from gpio import cleanup_gpio
import atexit

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config.from_object(Config)

# Initialize MQTT
mqtt = Mqtt(app)

# Register routes
register_routes(app, mqtt)

# Setup MQTT handlers
setup_mqtt(mqtt)

# Cleanup GPIO on exit
atexit.register(cleanup_gpio)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)