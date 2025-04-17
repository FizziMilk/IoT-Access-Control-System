from flask import render_template, request, redirect, url_for, flash, session as flask_session, jsonify
from datetime import datetime
import time
from utils import verify_otp_rest
from Web_Recognition.web_face_service import WebFaceService
import cv2
import logging
import os
import json
import base64
import threading
import traceback
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebRoutes")

def setup_qt_environment():
    """Set up appropriate Qt environment variables based on available plugins"""
    try:
        # Use xcb as it's the most reliable on Linux
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        logger.info("Using Qt platform plugin: xcb")
    except Exception as e:
        logger.error(f"Error setting up Qt environment: {e}")

def check_schedule(door_controller, mqtt_handler, backend_session=None, backend_url=None):
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
            if backend_session and backend_url:
                try:
                    backend_session.post(f"{backend_url}/log-door-access", json={
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
                if backend_session and backend_url:
                    try:
                        backend_session.post(f"{backend_url}/log-door-access", json={
                            "method": "Schedule",
                            "status": "Unlocked",
                            "details": f"{weekday} {open_time_str}-{close_time_str}"
                        })
                    except Exception as e:
                        print(f"[DEBUG] Error logging door access: {e}")
                
                return True
    return False

def setup_routes(app, door_controller, mqtt_handler, backend_session, backend_url):
    # Import the WebFaceService for use in routes
    from Web_Recognition.web_face_service import WebFaceService
    
    # Set up Qt environment once at startup
    setup_qt_environment()
    
    @app.route('/')
    def index():
        """
        Root route that ensures proper cleanup when returning from face recognition
        """
        # Check if we're coming from face recognition by checking the referrer
        referrer = request.referrer or ""
        
        if 'face-recognition' in referrer or 'process-face' in referrer:
            try:
                # Just make sure OpenCV windows are destroyed
                import cv2
                cv2.destroyAllWindows()
                cv2.waitKey(1)
            except Exception as e:
                logger.error(f"Error cleaning up resources: {e}")
        
        return render_template('entry_options.html')

    @app.route('/verify', methods=['POST'])
    def verify():
        phone_number = request.form['phone_number']
        otp_code = request.form['otp_code']
        response = verify_otp_rest(backend_session, backend_url, phone_number, otp_code)
        if response.get("status") == "approved":
            door_controller.unlock_door()
            flash("OTP verified, door unlocked", "success")
            
            # Log door unlock via OTP verification
            try:
                backend_session.post(f"{backend_url}/log-door-access", json={
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
        if check_schedule(door_controller, mqtt_handler, backend_session, backend_url):
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
                resp = backend_session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
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
        if check_schedule(door_controller, mqtt_handler, backend_session, backend_url):
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
                resp = backend_session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
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
        logger.info("Face recognition page requested")
        
        # Clean up any existing OpenCV windows
        try:
            import cv2
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception as e:
            logger.error(f"Error destroying windows: {e}")
        
        # Set Qt environment
        setup_qt_environment()
        
        # Return the template
        return render_template("face_recognition.html")
    
    @app.route('/process-face', methods=['POST'])
    def process_face():
        """Process facial recognition and redirect based on result"""
        face_image = None
        match = None
        
        # Add debugging to see what's happening with camera access
        logger.info("Starting face recognition process")
        
        # Check camera device directly
        import subprocess
        import time  # Ensure time is imported
        try:
            # Check if camera is in use
            result = subprocess.run(['lsof', '/dev/video0'], capture_output=True, text=True)
            logger.info(f"Camera device status before processing: {'In use' if result.stdout else 'Available'}")
            
            if result.stdout:
                # Camera appears to be in use, attempt to force-release
                logger.warning("Camera appears to be in use, will attempt to release")
                try:
                    cv2.destroyAllWindows()
                    # Add a longer wait to ensure full cleanup
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in pre-cleanup: {e}")
        except Exception as e:
            logger.error(f"Could not check camera status: {e}")
        
        # Create a fresh service instance for each request
        local_face_service = None
        
        try:
            # Determine headless mode
            headless_mode = os.environ.get('DISPLAY') is None
            
            # Create a fresh service instance for this request only
            logger.info("Creating fresh WebFaceService instance for recognition")
            local_face_service = WebFaceService(
                backend_session=backend_session, 
                backend_url=backend_url,
                headless=headless_mode
            )
            
            # Add timeout/retry for initialization
            max_retries = 3
            retry_count = 0
            success = False
            
            while retry_count < max_retries and not success:
                try:
                    # Initialize the service (load face data)
                    success = local_face_service.initialize()
                    if not success:
                        logger.warning(f"Service initialization failed, attempt {retry_count+1}/{max_retries}")
                        retry_count += 1
                        import time
                        time.sleep(1)  # Wait before retrying
                    else:
                        logger.info("Service initialized successfully")
                except Exception as e:
                    logger.error(f"Error during initialization: {e}")
                    retry_count += 1
                    import time
                    time.sleep(1)  # Wait before retrying
            
            if not success:
                flash("Error initializing face recognition. Please try again.", "danger")
                return redirect(url_for("door_entry"))
            
            # Capture face with liveness detection
            logger.info("Starting face capture with liveness detection")
            face_image = local_face_service.capture_face_with_liveness(timeout=30)
            
            if face_image is None:
                logger.warning("Face verification failed - no face detected or liveness check failed")
                flash("Face verification failed. Please try again.", "danger")
                return redirect(url_for("door_entry"))
            
            logger.info("Face captured successfully, attempting recognition")
            # Try to identify the person from known faces
            match = local_face_service.recognition.identify_face(face_image)
            
            # User recognized
            if match:
                logger.info(f"Face recognized as: {match.get('name')}")
                # Get their phone number
                phone_number = match['name']
                
                try:
                    # Check if this user is allowed direct access
                    resp = backend_session.get(f"{backend_url}/check-face-access/{phone_number}")
                    data = resp.json()
                    
                    if data.get("status") == "approved":
                        # Unlock door and log access
                        door_controller.unlock_door()
                        
                        # Log door unlock via facial recognition
                        try:
                            backend_session.post(f"{backend_url}/log-door-access", json={
                                "user": phone_number,
                                "method": "Facial Recognition",
                                "status": "Unlocked",
                                "details": f"Door unlocked via facial recognition, confidence: {match.get('confidence', 'N/A')}"
                            })
                        except Exception as e:
                            print(f"[DEBUG] Error logging door access: {e}")
                            
                        flash("Face recognized! Door unlocked.", "success")
                        return render_template("door_unlocked.html")
                    else:
                        # No direct access - provide options
                        flash("Face recognized but additional verification needed.", "warning")
                        return render_template("verify_options.html", phone_number=phone_number,
                                              user_recognized=True, face_recognized=True)
                except Exception as e:
                    logger.error(f"Error checking face access: {e}")
                    
                    # Default to requiring additional verification
                    flash("Face recognized but verification needed. Please use OTP.", "warning")
                    return render_template("verify_options.html", phone_number=phone_number,
                                         user_recognized=True, face_recognized=True)
            else:
                # Not recognized - allow registration or other verification
                # Generate face encoding for potential registration
                encoding = local_face_service.recognition.process_face_encoding(face_image)
                
                if encoding is not None:
                    # Store encoding in session for potential registration
                    # Check if it's a numpy array or already a list
                    try:
                        if isinstance(encoding, np.ndarray):
                            flask_session['face_encoding'] = encoding.tolist()
                        else:
                            # Already a list, store directly
                            flask_session['face_encoding'] = encoding
                    except Exception as e:
                        logger.error(f"Error storing face encoding: {e}")
                        # Try direct assignment as fallback
                        flask_session['face_encoding'] = encoding
                    
                    flask_session['has_temp_face'] = True
                    flash("Face not recognized. Please verify identity or register.", "warning")
                    return render_template("verify_options.html", user_recognized=False,
                                         face_recognized=False, has_face=True)
                else:
                    flash("Could not process face features. Please try again.", "danger")
                    return redirect(url_for("door_entry"))
                
        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            logger.error(traceback.format_exc())
            flash("An error occurred during face recognition.", "danger")
            return redirect(url_for("door_entry"))
        finally:
            # Always clean up resources in the local service instance
            if local_face_service:
                try:
                    local_face_service.release_camera()
                except Exception as e:
                    logger.error(f"Error releasing camera resources: {e}")
                
                # Force cleanup of OpenCV windows
                try:
                    import cv2
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                except Exception:
                    pass
                
                # RPI OS-specific: Try resetting the camera device directly
                try:
                    # Check if we're on Linux (RPI OS)
                    if os.path.exists('/dev/video0'):
                        logger.info("Attempting direct camera device reset")
                        
                        # Try killing any processes that might have the camera open
                        try:
                            import subprocess
                            import time  # Make sure time is imported here
                            # This is safer than pkill - it only affects video0 processes
                            subprocess.run('sudo fuser -k /dev/video0 2>/dev/null || true', 
                                          shell=True, timeout=2)
                            time.sleep(0.5)
                        except Exception as e:
                            logger.error(f"Error resetting camera device: {e}")
                except Exception as e:
                    logger.error(f"Error in OS-specific cleanup: {e}")
            
            # Add final check to see if camera was properly released
            try:
                import subprocess
                import time  # Ensure time is imported
                result = subprocess.run(['lsof', '/dev/video0'], capture_output=True, text=True)
                logger.info(f"Camera device status after cleanup: {'Still in use' if result.stdout else 'Released'}")
                
                # Add a small delay to let the system finish any processes
                time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in final camera check: {e}")
    
    @app.route('/register-face', methods=['POST'])
    def register_face():
        """Register a captured face with a phone number"""
        phone_number = request.form.get('phone_number')
        
        # Get the face encoding from Flask session
        face_encoding = flask_session.get('face_encoding')
        
        if not phone_number or not face_encoding:
            flash("Phone number and face image required.", "danger")
            return redirect(url_for("door_entry"))
            
        try:
            # Convert face encoding to base64 string
            # Make sure face_encoding is serializable (should be a list at this point)
            if not isinstance(face_encoding, list):
                try:
                    # Try to convert if it's a numpy array
                    if isinstance(face_encoding, np.ndarray):
                        face_encoding = face_encoding.tolist()
                except Exception as e:
                    logger.error(f"Error converting face encoding: {e}")
                    flash("Error processing face data.", "danger")
                    return redirect(url_for("door_entry"))
            
            # Convert face encoding list to JSON string
            encoding_json = json.dumps(face_encoding)
            
            # Encode JSON string as base64
            encoding_base64 = base64.b64encode(encoding_json.encode('utf-8')).decode('utf-8')
            
            # Send the face encoding and phone number to the backend
            resp = backend_session.post(f"{backend_url}/register-face", json={
                "phone_number": phone_number,
                "face_encoding": encoding_base64
            })
            data = resp.json()
            
            if data.get("status") == "success":
                # Clear the session data
                if 'face_encoding' in flask_session:
                    flask_session.pop('face_encoding')
                
                flash("Face registered successfully. Please wait for OTP or admin approval.", "success")
                # Try to send OTP to the newly registered user
                resp = backend_session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
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
            resp = backend_session.post(f"{backend_url}/update-user-name", json={"phone_number": phone_number, "name": name})
            data = resp.json()
            if data.get("status") == "success":
                flash("Name updated succesffuly. Please wait for admin approval.", "info")
            else:
                flash(data.get("error", "Error updating name"), "danger")
        except Exception as e:
            flash("Error connecting to backend.", "danger")
        return redirect(url_for("door_entry")) 