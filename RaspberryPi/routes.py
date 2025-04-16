from flask import render_template, request, redirect, url_for, flash, session
from datetime import datetime
import time
from utils import verify_otp_rest
from face_recognition_service import FaceRecognitionService

def check_schedule(door_controller, mqtt_handler, session=None, backend_url=None):
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
            
            # Log door unlock via schedule if session is available
            if session and backend_url:
                try:
                    session.post(f"{backend_url}/log-door-access", json={
                        "method": "Schedule",
                        "status": "Unlocked",
                        "details": "Force unlocked"
                    })
                except Exception as e:
                    print(f"[DEBUG] Error logging door access: {e}")
            
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
                
                # Log door unlock via schedule if session is available
                if session and backend_url:
                    try:
                        session.post(f"{backend_url}/log-door-access", json={
                            "method": "Schedule",
                            "status": "Unlocked",
                            "details": f"{weekday} {open_time_str}-{close_time_str}"
                        })
                    except Exception as e:
                        print(f"[DEBUG] Error logging door access: {e}")
                
                return True
    return False

def setup_routes(app, door_controller, mqtt_handler, session, backend_url):
    # Initialize facial recognition service
    face_service = FaceRecognitionService(backend_session=session, backend_url=backend_url)
    
    @app.route('/')
    def index():
        return render_template('entry_options.html')

    @app.route('/verify', methods=['POST'])
    def verify():
        phone_number = request.form['phone_number']
        otp_code = request.form['otp_code']
        response = verify_otp_rest(session, backend_url, phone_number, otp_code)
        if response.get("status") == "approved":
            door_controller.unlock_door()
            flash("OTP verified, door unlocked", "success")
            
            # Log door unlock via OTP verification
            try:
                session.post(f"{backend_url}/log-door-access", json={
                    "user": phone_number,
                    "method": "OTP Verification",
                    "status": "Unlocked",
                    "details": "Door unlocked after OTP verification"
                })
            except Exception as e:
                print(f"[DEBUG] Error logging door access: {e}")
                
            return render_template("door_unlocked.html")
        elif response.get("status") == "error":
            flash(response.get("message", "Incorrect OTP code. Please try again."), "danger")
            return render_template("otp.html", phone_number=phone_number)
        else:
            flash("Incorrect OTP code. Please try again.", "danger")
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
        # Check schedule first
        if check_schedule(door_controller, mqtt_handler, session, backend_url):
            return render_template("door_unlocked.html")

        # For GET requests, show the entry options page
        if request.method == "GET":
            return render_template("entry_options.html")

        # If we get here, schedule check didn't unlock the door
        if request.method == "POST":
            # Handle POST request for OTP verification
            phone_number = request.form.get('phone_number')
            if not phone_number:
                flash("Phone number is required for verification.", "danger")
                return redirect(url_for("door_entry"))

            try:
                # Send request to door-entry endpoint
                resp = session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
                print(f"[DEBUG] Backend response: {resp.status_code} - {resp.text}")
                data = resp.json()

                if data.get("status") == "OTP sent":
                    flash("OTP sent to your phone. Please enter the OTP.", "success")
                    return render_template("otp.html", phone_number=phone_number)
                elif data.get("status") == "pending":
                    flash(data.get("message", "Access pending."), "warning")
                    return render_template("pending.html", phone_number=phone_number)
                else:
                    flash(data.get("error", "An error occurred."), "danger")
                    return redirect(url_for("door_entry"))

            except Exception as e:
                flash("Error connecting to backend.", "danger")
                print(f"[DEBUG] Error connecting to backend: {e}")
                return redirect(url_for("door_entry"))

        return render_template("door_entry.html")

    @app.route('/phone-entry', methods=['GET', 'POST'])
    def phone_entry():
        """Handle the phone number entry flow"""
        # Check schedule first
        if check_schedule(door_controller, mqtt_handler, session, backend_url):
            return render_template("door_unlocked.html")

        # If we get here, schedule check didn't unlock the door
        if request.method == "POST":
            # Handle POST request for OTP verification
            phone_number = request.form.get('phone_number')
            if not phone_number:
                flash("Phone number is required for verification.", "danger")
                return redirect(url_for("phone_entry"))

            try:
                # Send request to door-entry endpoint
                resp = session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
                print(f"[DEBUG] Backend response: {resp.status_code} - {resp.text}")
                data = resp.json()

                if data.get("status") == "OTP sent":
                    flash("OTP sent to your phone. Please enter the OTP.", "success")
                    return render_template("otp.html", phone_number=phone_number)
                elif data.get("status") == "pending":
                    flash(data.get("message", "Access pending."), "warning")
                    return render_template("pending.html", phone_number=phone_number)
                else:
                    flash(data.get("error", "An error occurred."), "danger")
                    return redirect(url_for("phone_entry"))

            except Exception as e:
                flash("Error connecting to backend.", "danger")
                print(f"[DEBUG] Error connecting to backend: {e}")
                return redirect(url_for("phone_entry"))

        # Show the phone entry form
        return render_template("door_entry.html")
    
    @app.route('/face-recognition', methods=['GET'])
    def face_recognition():
        """Display the face recognition page"""
        # Force cleanup of any existing resources
        print("[DEBUG] face_recognition route called - ensuring a fresh start")
        
        try:
            # Kill all OpenCV processes that might be running
            import cv2
            import os
            import signal
            import subprocess
            import platform
            
            # First close any OpenCV windows
            cv2.destroyAllWindows()
            
            # Force release of any camera that might be open
            for i in range(5):  # Try multiple camera indices
                try:
                    cap = cv2.VideoCapture(i)
                    if cap.isOpened():
                        cap.release()
                        print(f"[DEBUG] Released camera {i} during cleanup")
                except:
                    pass
            
            # Extra thorough cleanup - attempt to kill any hung OpenCV processes
            if platform.system() != 'Windows':
                try:
                    # For Linux/Mac - find and kill any stuck cv2 processes
                    ps_process = subprocess.Popen(['ps', 'aux'], stdout=subprocess.PIPE)
                    output, _ = ps_process.communicate()
                    
                    for line in output.splitlines():
                        if b'python' in line and b'cv2' in line:
                            # Extract PID
                            pid = int(line.split()[1])
                            # Don't kill our own process
                            if pid != os.getpid():
                                try:
                                    os.kill(pid, signal.SIGTERM)
                                    print(f"[DEBUG] Killed hung OpenCV process with PID {pid}")
                                except:
                                    pass
                except Exception as e:
                    print(f"[DEBUG] Error cleaning up processes: {e}")
            
            # After process cleanup, reinitialize the face recognition service
            face_service.reset_system()
            
            # One more release attempt
            face_service.release_camera()
        except Exception as e:
            print(f"[DEBUG] Error during cleanup: {e}")
        
        # Return the template with manual activation
        return render_template("face_recognition.html")
    
    @app.route('/process-face', methods=['POST'])
    def process_face():
        """Process facial recognition and redirect based on result"""
        try:
            # Attempt to identify the user with liveness check
            user_match = face_service.identify_user(timeout=30)
            
            if user_match:
                # User recognized, get their phone number
                phone_number = user_match[0]['name']  # Access the first element of the list
                
                try:
                    # Send OTP directly since user is recognized
                    resp = session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
                    data = resp.json()
                    
                    if data.get("status") == "OTP sent":
                        flash("Face recognized! OTP sent to your phone.", "success")
                        return render_template("otp.html", phone_number=phone_number)
                    elif data.get("status") == "pending":
                        flash(data.get("message", "Access pending."), "warning")
                        return render_template("pending.html", phone_number=phone_number)
                    else:
                        flash(data.get("error", "An error occurred."), "danger")
                        # Always clean up before redirecting
                        face_service.release_camera()
                        return redirect(url_for("door_entry"))
                        
                except Exception as e:
                    flash("Error connecting to backend.", "danger")
                    print(f"[DEBUG] Error connecting to backend: {e}")
                    # Always clean up before redirecting
                    face_service.release_camera()
                    return redirect(url_for("door_entry"))
            else:
                # User not recognized, capture image and collect phone number
                face_image = face_service.capture_face_with_liveness()
                if face_image is not None:
                    # Register the face and save encoding temporarily
                    face_encoding = face_service.register_face(face_image)
                    if face_encoding:
                        # Store the encoding in the Flask session
                        from flask import session as flask_session
                        flask_session['face_encoding'] = face_encoding
                            
                        flash("Face captured successfully. Please enter your phone number to register.", "info")
                        return render_template("register_face.html")
                
                flash("Face recognition failed. Please try again or use phone number.", "danger")
                # Always clean up before redirecting
                face_service.release_camera()
                return redirect(url_for("door_entry"))
                
        except Exception as e:
            flash(f"Error during face recognition: {str(e)}", "danger")
            print(f"[DEBUG] Face recognition error: {e}")
            # Always clean up on exception
            try:
                face_service.release_camera()
            except:
                pass
            return redirect(url_for("door_entry"))
    
    @app.route('/register-face', methods=['POST'])
    def register_face():
        """Register a captured face with a phone number"""
        phone_number = request.form.get('phone_number')
        
        # Get the face encoding from Flask session
        from flask import session as flask_session
        face_encoding = flask_session.get('face_encoding')
        
        if not phone_number or not face_encoding:
            flash("Phone number and face image required.", "danger")
            return redirect(url_for("door_entry"))
            
        try:
            # Send the face encoding and phone number to the backend
            resp = session.post(f"{backend_url}/register-face", json={
                "phone_number": phone_number,
                "face_encoding": face_encoding
            })
            data = resp.json()
            
            if data.get("status") == "success":
                # Clear the session data
                if 'face_encoding' in flask_session:
                    flask_session.pop('face_encoding')
                
                flash("Face registered successfully. Please wait for OTP or admin approval.", "success")
                # Try to send OTP to the newly registered user
                resp = session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
                data = resp.json()
                
                if data.get("status") == "OTP sent":
                    return render_template("otp.html", phone_number=phone_number)
                elif data.get("status") == "pending":
                    return render_template("pending.html", phone_number=phone_number)
                else:
                    return redirect(url_for("door_entry"))
            else:
                flash(data.get("error", "Error registering face"), "danger")
        except Exception as e:
            flash("Error connecting to backend.", "danger")
            print(f"[DEBUG] Error: {e}")
            
        return redirect(url_for("door_entry"))
    
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