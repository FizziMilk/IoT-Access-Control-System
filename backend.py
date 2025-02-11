
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from twilio.rest import Client
import os

app = Flask(__name__)
api = Api(app)

guest_pins = {}

#Twilio setup
class AuthenticatePin(Resource):
	def post(self):
		data = request.json
		pin = data.get('pin')
		#pin validation logic
		return{"status":"success"}
	
class UnlockDoor(Resource):
	def post(self):
		data = request.json
		#door unlock logic
		return {"status": "unlocked"}
	
class GeneratePin(Resource):
	def post(self):
		data = request.json
		phone_number = data.get('phone_number')
		pin = str(random.randint(1000, 9999)) # Generate a 4-digit PIN
		guest_pins[phone_number] = pin
		# SMS logic here
		return {"status": "PIN sent", "pin": pin}
	
class AccessLogs(Resource):
	def get(self):
		#Add logic to retrieve access logs from the database
		logs = {"timestamp": "2025-11-02 14:36", "user": "John", "method":
		  "facial", "status": "success"}
		return logs

api.add_resource(AuthenticatePin, '/authenticate-pin')
api.add_resource(UnlockDoor, '/unlock')
api.add_resource(GeneratePin, '/generate-pin')
api.add_resource(AccessLogs, '/access-logs')

if __name__ == '__main__':
	app.run(debug=True)
