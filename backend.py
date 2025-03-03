from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from twilio.rest import Client
from dotenv import load_dotenv
from flask_bcrypt import Bcrypt
from flask_jwt_extended import JWTManager
from flask_jwt_extended import create_access_token
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
bcrypt = Bcrypt(app)

## JWT Setup
app.config["JWT_SECRET_KEY"] = "super-secret-key" # change to more secure later
jwt = JWTManager(app)


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
## User database model
class User(db.Model):
	id = db.Column(db.Integer, primary_key = True)
	username = db.Column(db.String(50), unique=True, nullable= False)
	password_hash = db.Column(db.String(128),nullable= False)
	phone_number = db.Column(db.String(20), unique=True, nullable=False)

	def set_password(self, password: str) -> None:
		self.password_hash = bcrypt.generate_password_hash(password).decode("utf-8")

	def check_password(self, password: str) -> bool:
		return bcrypt.check_password_hash(self.password_hash, password)
# Create the database tables if they don't exist
with app.app_context():
	db.create_all()

# Handles login request. Checks username and password
class LoginResource(Resource):
	def post(self):
		data = request.get_json()
		username = data.get("username")
		password = data.get("password")

		if not username or not password:
			return {"error": "Username and password required"}, 400
	#Check credentials
		user= User.query.filter_by(username=username).first()
		if not user or not user.check_password(password):
			return {"error": "Invalid credentials"}, 401
	#Credentials are valid, send OTP to user's phone
		try:
			verification = client.verify.v2.services(TWILIO_VERIFY_SID) \
				.verifications \
				.create(to=user.phone_number, channel="sms") 	

		#Return ok to let client proceed 
			return {
				"status": "OK",
				"message": "OTP sent to user",
				"phone_number": user.phone_number,
				"verification_status": verification.status
			}, 200

		except Exception as e:
			return {"error": str(e)}, 500

## Check OTP for phone login
class CheckVerification(Resource):
	def post(self):
		data = request.get_json()
		username = data.get("username")
		code = data.get("otp")

		if not username or not code:
			return{"error": "username and code are required"}, 400

		user = User.query.filter_by(username = username).first()
		if not user:
			return{"error": "User not found"}, 404

		try:
			verification_check = client.verify.v2.services(TWILIO_VERIFY_SID)\
				.verification_checks\
				.create(to=user.phone_number, code=code)

			if verification_check.status == "approved":

				access_token = create_access_token(identity=user.id)
				return{
					"status": "approved",
					"token": access_token
				}, 200
			else:
				return{"status":verification_check.status},200

		except Exception as e:
			return {"error": str(e)},500


# Verification for RPI?
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
class CheckVerificationRPI(Resource):
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
api.add_resource(LoginResource,"/login")
api.add_resource(CheckVerification,"/verify-otp")
api.add_resource(StartVerification, "/start-verification")
api.add_resource(CheckVerificationRPI, "/check-verification-RPI")
api.add_resource(UnlockDoor, '/unlock')
api.add_resource(GetAccessLogs, "/access-logs")
if __name__ == '__main__':
	app.run(debug=True)
