from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
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

## Configure SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///access_logs.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

## Define AccessLog model

class AccessLog(db.Model):
	id = db.Column(db.Integer, primary_key = True)
	user = db.Column(db.String(50), nullable = False)
	method = db.Column(db.String(50), nullable = False)
	status = db.Column(db.String(20), nullable = False)
	timestamp = db.Column(db.DateTime, default = datetime.utcnow)

	def __repr__(self):
		return f"<AccessLog {self.user} - {self.status}>"
# Create the database tables if they don't exist
with app.app_context():
	db.create_all()

# Twilio Verification Start
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

			# Log the initiation of verification
			new_log = AccessLog(
				user = phone_number,
				method = "SMS OTP",
				status = "Started"
			)
			db.session.add(new_log)
			db.session.commit()

			return{"status": verification.status}, 200

		except Exception as e:
			#Log failure if Twilio call fails
			new_log = AccessLog(
				user = phone_number or "Unknown",
				method = "SMS OTP",
				status = "Failed"
			)
			db.session.add(new_log)
			db.session.commit()

			return {"error": str(e)}, 500
# Twilio Verification Check
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

			# Log the verification result
			new_log = AccessLog(
				user = phone_number,
				method = "SMS OTP",
				status = verification_check.status
			)
			db.session.add(new_log)
			db.session.commit()

			return {"status": verification_check.status}, 200

		except Exception as e:
			#Log failure
			new_log = AccessLog(
				user = phone_number or "Unknown",
				method = "SMS OTP",
				status = "Failed"
			)
			db.session.add(new_log)
			db.session.commit()

			return {"error": str(e)}, 500

#Resource to retrieve all logs
class GetAccessLogs(Resource):
	def get(self):
		logs = AccessLog.query.all()
		results = []
		for log in logs:
			results.append({
				"id":log.id,
				"user": log.user,
				"method": log.method,
				"status": log.status,
				"timestamp": log.timestamp.isoformat()
			})
		return results, 200

class UnlockDoor(Resource):
	def post(self):
		data = request.json
		#door unlock logic
		return {"status": "unlocked"}

api.add_resource(StartVerification, "/start-verification")
api.add_resource(CheckVerification, "/check-verification")
api.add_resource(UnlockDoor, '/unlock')
api.add_resource(GetAccessLogs, "/access-logs")
if __name__ == '__main__':
	app.run(debug=True)
