from flask import render_template, request, redirect, url_for, flash, jsonify
from gpio import unlock_door
from utils import verify_otp_rest, update_schedule
from datetime import datetime

schedule = {}

def register_routes(app, mqtt):
    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/door-entry', methods=['GET', 'POST'])
    def door_entry():
        if request.method == "POST":
            now = datetime.now()
            weekday = now.strftime("%A")

            # Check if the current time is within the schedule
            if weekday in schedule:
                entry = schedule[weekday]
                open_time_str = entry.get("open_time")
                close_time_str = entry.get("close_time")
                force_unlocked = entry.get("forceUnlocked", False)

                if force_unlocked:
                    unlock_door()
                    flash("Door unlocked based on schedule.", "success")
                    return render_template("door_unlocked.html")

                if open_time_str and close_time_str:
                    try:
                        open_time = datetime.strptime(open_time_str, "%H:%M").time()
                        close_time = datetime.strptime(close_time_str, "%H:%M").time()
                    except ValueError as ve:
                        flash("Schedule time format error.", "danger")
                        return redirect(url_for("door_entry"))

                    current_time = now.time().replace(second=0, microsecond=0)
                    if open_time <= current_time <= close_time:
                        unlock_door()
                        flash("Door unlocked based on schedule.", "success")
                        return render_template("door_unlocked.html")

            # Otherwise proceed with OTP verification
            phone_number = request.form.get('phone_number')
            if not phone_number:
                flash("Phone number is required for verification.", "danger")
                return redirect(url_for("door_entry"))

            try:
                resp = verify_otp_rest(phone_number, None)
                if resp.get("status") == "approved":
                    unlock_door()
                    flash("OTP verified, door unlocked", "success")
                    return redirect(url_for('index'))
                else:
                    flash(resp.get("message", "Verification failed"), "danger")
            except Exception as e:
                flash("Error connecting to backend.", "danger")
                return redirect(url_for("door_entry"))

        return render_template("door_entry.html")

    @app.route('/update_schedule', methods=['POST'])
    def update_schedule_route():
        global schedule
        data = request.get_json()
        schedule = update_schedule(data)
        return jsonify({"status": "success"}), 200