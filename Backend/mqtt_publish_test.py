import paho.mqtt.client as mqtt

# Broker configuration

MQTT_BROKER = "ec2-51-20-233-128.eu-north-1.compute.amazonaws.com"
MQTT_PORT = 8883
TOPIC = "door/commands"
MESSAGE = "open_door"
CA_CERT = "/etc/mosquitto/ca_certificates/ca.crt"

def on_connect(client, userdata, flags, rc):
	if rc == 0:
		print("Connected successfully")
		client.publish(TOPIC, MESSAGE)
		print(f"Published message: {MESSAGE} to topic: {TOPIC}")
	else:
		print("Connection failed with code:",rc)
	client.disconnect()

client = mqtt.Client()

# Set TLS parameters
client.tls_set(ca_certs = CA_CERT)

client.on_connect = on_connect

client.connect(MQTT_BROKER,MQTT_PORT,keepalive = 60)
client.loop_forever()
