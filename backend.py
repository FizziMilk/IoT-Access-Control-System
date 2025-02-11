
from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from twilio.rest import Client
from dotenv import load_dotenv
import os

#Twilio Setup
load_dotenv("twilio.env") # Load the Twilio environment file
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN")

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

TWILIO_VERIFY_SID="VA1c1c5d9906340e1c187f83dbc26057a0"
####

app = Flask(__name__)
api = Api(app)

guest_pins = {}

class StartVerification(Resource):
	def post (self):
		data = request.get_json()
		phone_number = data.get("phone_number")

		if not phone_number:
			return {"error": "phone_number is required"}, 400

		try:
			verification = client.verify.v2.services(TWILIO_VERIFY_SID) \
				.verifications \
				.create(to=phone_number, channel = "sms")

			return{"status": verification.status}, 200
		except Exception as e:
			return {"error": str(e)}, 500

class CheckVerification(Resource):
	def post(self):
		data = request.get_json()
		phone_number = data.get("phone_number")
		code = data.get("code")

		if not phone_number or not code:
			return {"error": "phone_number and code are required"}, 400

		try:
			verification_check = client.verify.v2.services(TWILIO_VERIFY_SID) \
				.verification_checks \
				.create(to=phone_number, code=code)
			
			return {"status": verification_check.status}, 200
		except Exception as e:
			return {"error": str(e)}, 500

class UnlockDoor(Resource):
	def post(self):
		data = request.json
		#door unlock logic
		return {"status": "unlocked"}


class AccessLogs(Resource):
	def get(self):
		#Add logic to retrieve access logs from the database
		logs = {"timestamp": "2025-11-02 14:36", "user": "John", "method":
		  "facial", "status": "success"}
		return logs


api.add_resource(StartVerification, "/start-verification")
api.add_resource(CheckVerification, "/check-verification")
api.add_resource(UnlockDoor, '/unlock')
api.add_resource(AccessLogs, '/access-logs')

if __name__ == '__main__':
	app.run(debug=True)
