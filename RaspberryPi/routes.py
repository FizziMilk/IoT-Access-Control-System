from flask import render_template, request, redirect, url_for, flash, jsonify
from datetime import datetime
import time
from utils import verify_otp_rest
import requests
import os
import json

def check_schedule(door_controller, mqtt_handler):
    """Check if door should be unlocked based on current schedule"""
    now = datetime.now()
    weekday = now.strftime("%A")

    if weekday in mqtt_handler.schedule:
        entry = mqtt_handler.schedule[weekday]
        print(f"[DEBUG] Schedule entry for {weekday}: {entry}")
        open_time_str = entry.get("open_time")
        close_time_str = entry.get("close_time")
        force_unlocked = entry.get("forceUnlocked", False)

        if force_unlocked:
            print("[DEBUG] Force unlock is enabled. Unlocking door.")
            door_controller.unlock_door()
            flash("Door unlocked based on schedule.", "success")
            return True

        if open_time_str and close_time_str:
            try:
                open_time = datetime.strptime(open_time_str, "%H:%M").time()
                close_time = datetime.strptime(close_time_str, "%H:%M").time()
            except ValueError as ve:
                flash("Schedule time format error.", "danger")
                print(f"[DEBUG] Schedule time format error: {ve}")
                return False

            current_time = now.time().replace(second=0, microsecond=0)

            if open_time <= current_time <= close_time:
                door_controller.unlock_door()
                flash("Door unlocked based on schedule.", "success")
                return True
    return False

def setup_routes(app, door_controller, mqtt_handler, session, backend_url):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/verify', methods=['POST'])
    def verify():
        try:
            data = request.get_json()
            phone_number = data.get('phone_number')
            otp_code = data.get('otp_code')

            if not phone_number or not otp_code:
                return jsonify({"status": "error", "message": "Missing phone number or OTP code"}), 400

            # Send verification request to backend
            response = requests.post(
                f"{os.getenv('BACKEND_URL')}/check-verification-RPI",
                json={"phone_number": phone_number, "otp_code": otp_code}
            )
            response_data = response.json()

            if response_data.get("status") == "approved":
                # Door is unlocked (either globally or through verification)
                door_controller.unlock_door(method="SMS OTP", phone_number=phone_number)
                return jsonify({"status": "success", "message": "Door unlocked"}), 200
            else:
                return jsonify({"status": "error", "message": response_data.get("message", "Access denied")}), 403

        except Exception as e:
            print(f"[ERROR] Error in verify route: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/update_schedule', methods=['POST'])
    def update_schedule():
        try:
            data = request.get_json()
            mqtt_handler.update_schedule(data)
            return jsonify({"status": "success"}), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/door-entry', methods=['POST'])
    def door_entry():
        try:
            data = request.get_json()
            phone_number = data.get('phone_number')
            
            if not phone_number:
                return jsonify({"status": "error", "message": "Missing phone number"}), 400

            # Send entry request to backend
            response = requests.post(
                f"{os.getenv('BACKEND_URL')}/door-entry",
                json={"phone_number": phone_number}
            )
            response_data = response.json()

            if response_data.get("status") == "approved":
                # Door is unlocked through entry request
                door_controller.unlock_door(method="entry", phone_number=phone_number)
                return jsonify({"status": "success", "message": "Door unlocked"}), 200
            else:
                return jsonify({"status": "error", "message": response_data.get("message", "Access denied")}), 403

        except Exception as e:
            print(f"[ERROR] Error in door_entry route: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route('/update-name', methods=['POST'])
    def update_name():
        phone_number = request.form.get('phone_number')
        name = request.form.get('name')
        if not phone_number or not name:
            flash("Name and phone number are required.", "danger")
            return redirect(url_for("door_entry"))
        try:
            resp = session.post(f"{backend_url}/update-user-name", json={"phone_number": phone_number, "name": name})
            data = resp.json()
            if data.get("status") == "success":
                flash("Name updated succesffuly. Please wait for admin approval.", "info")
            else:
                flash(data.get("error", "Error updating name"), "danger")
        except Exception as e:
            flash("Error connecting to backend.", "danger")
        return redirect(url_for("door_entry"))

def verify(door_controller):
    """Handle verification requests from the backend"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    user_id = data.get('user_id')
    method = data.get('method')
    phone_number = data.get('phone_number')
    
    if not all([user_id, method, phone_number]):
        return jsonify({"error": "Missing required fields"}), 400
        
    # Unlock the door
    door_controller.unlock_door(method, phone_number)
    
    return jsonify({"status": "success", "message": "Door unlocked"}), 200

def door_entry(door_controller):
    """Handle door entry events"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    user_id = data.get('user_id')
    method = data.get('method')
    phone_number = data.get('phone_number')
    
    if not all([user_id, method, phone_number]):
        return jsonify({"error": "Missing required fields"}), 400
        
    # Log the entry
    entry_log = {
        "user_id": user_id,
        "method": method,
        "phone_number": phone_number,
        "timestamp": datetime.now().isoformat()
    }
    
    # TODO: Save entry_log to database
    
    return jsonify({"status": "success", "message": "Entry logged"}), 200

def update_schedule(door_controller):
    """Handle schedule updates"""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400
        
    schedule = data.get('schedule')
    if not schedule:
        return jsonify({"error": "No schedule provided"}), 400
        
    # TODO: Update schedule in database
    
    return jsonify({"status": "success", "message": "Schedule updated"}), 200 