from flask import render_template, request, redirect, url_for, flash
from datetime import datetime
import time
from utils import verify_otp_rest

def setup_routes(app, door_controller, mqtt_handler, session, backend_url):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/verify', methods=['POST'])
    def verify():
        phone_number = request.form['phone_number']
        otp_code = request.form['otp_code']
        response = verify_otp_rest(session, backend_url, phone_number, otp_code)
        if response.get("status") == "approved":
            door_controller.unlock_door()
            flash("OTP verified, door unlocked", "success")
            return render_template("door_unlocked.html")
        elif response.get("status") == "error":
            flash(response.get("message", "An error occurred during verification"), "danger")
            return render_template("otp.html", phone_number=phone_number)
        else:
            flash("Invalid OTP or unexpected response", "danger")
            return render_template("otp.html", phone_number=phone_number)

    @app.route('/update_schedule', methods=['POST'])
    def update_schedule():
        try:
            data = request.get_json()
            return mqtt_handler.update_schedule(data)
        except Exception as e:
            print(f"Error updating schedule: {e}")
            return {"status": "error", "message": str(e)}, 500
    
    @app.route('/door-entry', methods=['GET', 'POST'])
    def door_entry():
        now = datetime.now()
        weekday = now.strftime("%A")

        # Check if the current time is within the schedule
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
                return render_template("door_unlocked.html")

            if open_time_str and close_time_str:
                try:
                    open_time = datetime.strptime(open_time_str, "%H:%M").time()
                    close_time = datetime.strptime(close_time_str, "%H:%M").time()
                except ValueError as ve:
                    flash("Schedule time format error.", "danger")
                    print(f"[DEBUG] Schedule time format error: {ve}")
                    return redirect(url_for("door_entry"))

                current_time = now.time().replace(second=0, microsecond=0)

                if force_unlocked or (open_time <= current_time <= close_time):
                    door_controller.unlock_door()
                    flash("Door unlocked based on schedule.", "success")
                    return render_template("door_unlocked.html")

        # Otherwise proceed with OTP verification
        phone_number = request.form.get('phone_number')
        if not phone_number:
            flash("Phone number is required for verification.", "danger")
            return redirect(url_for("door_entry"))

        try:
            resp = session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
            print(f"[DEBUG] Backend response: {resp.status_code} - {resp.text}")
            data = resp.json()
        except Exception as e:
            flash("Error connecting to backend.", "danger")
            print(f"[DEBUG] Error connecting to backend: {e}")
            return redirect(url_for("door_entry"))

        if data.get("status") == "OTP sent":
            flash("OTP sent to your phone. Please enter the OTP.", "success")
            return render_template("otp.html", phone_number=phone_number)
        elif data.get("status") == "pending":
            flash(data.get("message", "Access pending."), "warning")
            return render_template("pending.html", phone_number=phone_number)
        else:
            flash(data.get("error", "An error occurred."), "danger")
            return redirect(url_for("door_entry"))

    return render_template("door_entry.html")

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