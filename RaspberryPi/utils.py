import requests
from flask import current_app

def verify_otp_rest(phone_number, otp_code):
    payload = {"phone_number": phone_number, "otp_code": otp_code}
    response = requests.post(f"{current_app.config['BACKEND_URL']}/check-verification-RPI", json=payload)
    return response.json()

def update_schedule(data):
    new_schedule = {}
    for entry in data:
        day = entry.get("day")
        if day:
            new_schedule[day] = entry
    return new_schedule