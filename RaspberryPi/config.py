import os
import ssl

class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "supersecretkey")
    BACKEND_URL = os.getenv("BACKEND_URL")
    BACKEND_CA_CERT = os.getenv("CA_CERT")
    MQTT_BROKER_URL = os.getenv("MQTT_BROKER")
    MQTT_BROKER_PORT = int(os.getenv("MQTT_PORT", 1883))
    MQTT_TLS_VERSION = ssl.PROTOCOL_TLSv1_2
    MQTT_TLS_ENABLED = True
    MQTT_TLS_CA_CERTS = os.getenv("CA_CERT")

    # Ensure required environment variables are set
    REQUIRED_ENV_VARS = ["MQTT_BROKER", "MQTT_PORT", "CA_CERT", "BACKEND_URL"]
    for var in REQUIRED_ENV_VARS:
        if not os.getenv(var):
            raise EnvironmentError(f"Missing required environment variable: {var}")