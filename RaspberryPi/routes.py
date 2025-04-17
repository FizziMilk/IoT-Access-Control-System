from flask import render_template, request, redirect, url_for, flash, session as flask_session, jsonify, send_file, send_from_directory
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
import uuid
import subprocess
import io

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
    
    # Flag to track if backend connectivity has been verified
    backend_available = False
    
    # Function to verify backend connectivity
    def check_backend_availability():
        nonlocal backend_available
        if backend_available:
            return True
            
        try:
            test_resp = backend_session.get(f"{backend_url}/get-face-data", timeout=3)
            if test_resp.status_code == 200:
                backend_available = True
                logger.info("Backend API connectivity verified")
                return True
            else:
                logger.warning(f"Backend API returned status code: {test_resp.status_code}")
                return False
        except Exception as e:
            logger.warning(f"Backend API connectivity test failed: {e}")
            return False
    
    # Function to unlock door without using backend API
    def unlock_door_direct(method="Manual Override", user="Local System"):
        try:
            # Use door controller directly
            door_controller.unlock_door()
            logger.info(f"Door unlocked directly via {method} by {user}")
            
            # Try to log to backend if available, but don't fail if it's not
            if backend_available:
                try:
                    backend_session.post(f"{backend_url}/log-door-access", json={
                        "user": user,
                        "method": method,
                        "status": "Unlocked",
                        "details": f"Door unlocked via {method}"
                    }, timeout=2)
                except Exception as e:
                    logger.error(f"Failed to log door access to backend: {e}")
            
            return True
        except Exception as e:
            logger.error(f"Error unlocking door directly: {e}")
            return False
    
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
        
        # Clean up any existing OpenCV windows and force camera release
        try:
            import cv2
            cv2.destroyAllWindows()
            cv2.waitKey(1)
            
            # Try to force release camera at OS level
            import subprocess
            subprocess.run('sudo fuser -k /dev/video0 2>/dev/null || true', 
                           shell=True, timeout=2)
        except Exception as e:
            logger.error(f"Error cleaning up resources: {e}")
        
        # Check if testing mode - pass this to the template
        testing_mode = request.args.get('testing', 'false').lower() == 'true'
        
        # Return the template
        return render_template("face_recognition.html", testing_mode=testing_mode)
    
    @app.route('/face-recognition-feed')
    def face_recognition_feed():
        """Return the latest frame from the face recognition process with detection overlays"""
        try:
            # Make sure the UPLOAD_FOLDER is defined
            if 'UPLOAD_FOLDER' not in app.config:
                logger.error("UPLOAD_FOLDER not defined in app configuration")
                return send_placeholder_image()
            
            # Make sure the uploads directory exists
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                logger.warning(f"Upload folder does not exist: {upload_folder}, creating it")
                os.makedirs(upload_folder, exist_ok=True)
            
            debug_frame_path = os.path.join(upload_folder, 'debug_frame.jpg')
            
            # If no debug frame exists yet, return a placeholder
            if not os.path.exists(debug_frame_path):
                return send_placeholder_image()
            
            # Return the latest frame
            return send_file(debug_frame_path, mimetype='image/jpeg', max_age=0)
        except Exception as e:
            app.logger.error(f"Error serving face recognition feed: {str(e)}")
            return send_placeholder_image()
    
    def send_placeholder_image():
        """Create and send a placeholder image when no frame is available"""
        # Create a blank image with text
        import cv2
        import numpy as np
        import io
        
        # Create a blank image with text
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(img, "Waiting for camera...", (150, 240),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert to JPEG
        _, buffer = cv2.imencode('.jpg', img)
        
        # Convert to bytes and create a file-like object
        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        
        return send_file(io_buf, mimetype='image/jpeg', max_age=0)
        
    @app.route('/process-face', methods=['POST'])
    def process_face():
        """Process facial recognition using a separate process"""
        logger.info("Starting face recognition in separate process")
        
        # Check backend availability, but proceed regardless
        backend_is_available = check_backend_availability()
        logger.info(f"Backend API availability: {backend_is_available}")
        
        try:
            # Check if UPLOAD_FOLDER is configured
            if 'UPLOAD_FOLDER' not in app.config:
                logger.error("UPLOAD_FOLDER not configured in app settings")
                flash("System configuration error. Please contact administrator.", "danger")
                return redirect(url_for("door_entry"))
            
            # Ensure upload folder exists
            upload_folder = app.config['UPLOAD_FOLDER']
            if not os.path.exists(upload_folder):
                logger.warning(f"Upload folder does not exist: {upload_folder}, creating it")
                os.makedirs(upload_folder, exist_ok=True)
            
            # Generate a unique output file
            output_dir = '/tmp/face_recognition_results'
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"face_result_{uuid.uuid4()}.json")
            
            # Check if we should skip liveness detection (for testing only)
            skip_liveness = request.form.get('skip_liveness', 'false').lower() == 'true'
            if skip_liveness:
                logger.warning("Skipping liveness detection - TESTING MODE ONLY")
            
            # Set up frames directory for debug display
            frames_dir = '/tmp/face_recognition_frames'
            os.makedirs(frames_dir, exist_ok=True)
            
            # Build the command
            cmd = [
                'python3', 
                os.path.join(os.path.dirname(os.path.abspath(__file__)), 'face_recognition_process.py'),
                '--output', output_file,
                '--debug-frames',
                '--frames-dir', frames_dir,
                '--upload-folder', upload_folder,
                '--timeout', '45',  # Increased timeout for better recognition
            ]
            
            # Only add backend URL if backend is available
            if backend_is_available:
                cmd.extend(['--backend-url', backend_url])
            
            # Add skip-liveness flag if needed
            if skip_liveness:
                cmd.append('--skip-liveness')
            
            logger.info(f"Executing: {' '.join(cmd)}")
            
            # Run the standalone script as a separate process
            process = subprocess.run(cmd, 
                                    timeout=60,  # Increased timeout from 45 to 60 seconds
                                    check=False,
                                    capture_output=True)
            
            logger.info(f"Process returned with code {process.returncode}")
            
            if process.stderr:
                logger.warning(f"Process stderr: {process.stderr.decode('utf-8', errors='replace')}")
                
                # Log the stderr output to help with debugging
                stderr_output = process.stderr.decode('utf-8', errors='replace')
                if "liveness check" in stderr_output.lower():
                    logger.info("Liveness check details found in stderr - helpful for debugging")
            
            # Wait briefly to ensure the result file is written
            time.sleep(0.5)
            
            # Check if the output file exists
            if not os.path.exists(output_file):
                logger.error("Output file not created")
                flash("Face recognition failed. Please try again.", "danger")
                return redirect(url_for("door_entry"))
            
            # Read the results
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            # Clean up the output file
            try:
                os.remove(output_file)
            except Exception as e:
                logger.warning(f"Could not remove output file: {e}")
            
            # Process the results
            if results.get('success', False):
                # Get the first match if there are any results
                if results.get('results') and len(results['results']) > 0:
                    # We have a direct match from the standalone process
                    match = results['results'][0]
                    phone_number = match['name']  # Name field actually contains phone number
                    confidence = match['confidence']
                    
                    logger.info(f"Face directly matched with {phone_number}, confidence: {confidence:.2f}")
                    
                    # Directly unlock the door without backend check
                    if unlock_door_direct(method="Facial Recognition", user=phone_number):
                        flash("Face recognized! Door unlocked.", "success")
                        return render_template("door_unlocked.html")
                    else:
                        flash("Face recognized but failed to unlock door.", "danger")
                        return redirect(url_for("door_entry"))
                
                # We detected a live face, but no match was found
                # Extract face encoding if available for potential registration
                if results.get('face_detected') and results.get('is_live'):
                    face_encodings = results.get('face_encodings')
                    if face_encodings and len(face_encodings) > 0:
                        # Store the face encoding for potential registration
                        flask_session['face_encoding'] = face_encodings[0]
                        flask_session['has_temp_face'] = True
                        
                        flash("Face not recognized. Please verify identity or register.", "warning")
                        return render_template("verify_options.html", user_recognized=False,
                                            face_recognized=False, has_face=True)
                        
                    else:
                        # No face encodings found but the face was detected
                        flash("Face detected but couldn't be processed properly. Please try again.", "warning")
                        return redirect(url_for("door_entry"))
                else:
                    # Handle cases where face was detected but not live or couldn't be processed
                    if results.get('face_detected') and not results.get('is_live'):
                        flash("Could not verify that this is a real face. Please try again with better lighting.", "warning")
                    else:
                        flash("No face detected. Please try again and ensure good lighting.", "warning")
                    return redirect(url_for("door_entry"))
            else:
                # Error during face recognition
                error_msg = results.get('error', "Face recognition failed. Please try again.")
                logger.warning(f"Face recognition error: {error_msg}")
                
                # Provide more helpful error messages
                if "liveness" in error_msg.lower():
                    flash("Couldn't verify this is a real face. Please try with better lighting or enable testing mode.", "warning")
                elif "timeout" in error_msg.lower():
                    flash("Face recognition took too long. Please try again in better lighting conditions.", "warning")
                else:
                    flash(error_msg, "danger")
                    
                return redirect(url_for("door_entry"))
                
        except subprocess.TimeoutExpired:
            logger.error("Face recognition process timed out")
            flash("Face recognition took too long. Please try again.", "danger")
            return redirect(url_for("door_entry"))
        except Exception as e:
            logger.error(f"Error processing face recognition: {e}")
            logger.error(traceback.format_exc())
            flash("An error occurred during face recognition.", "danger")
            return redirect(url_for("door_entry"))
    
    @app.route('/register-face', methods=['POST'])
    def register_face():
        """Register a captured face with a phone number"""
        phone_number = request.form.get('phone_number')
        
        # Get the face encoding from Flask session
        face_encoding = flask_session.get('face_encoding')
        
        if not phone_number or not face_encoding:
            flash("Phone number and face image required.", "danger")
            return redirect(url_for("door_entry"))
        
        # Check backend availability before attempting registration
        if not check_backend_availability():
            flash("Backend service unavailable. Registration will be processed when connection is restored.", "warning")
            
            # Save registration locally for future processing
            try:
                # Store the face data in a local file for later sync
                local_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pending_registrations')
                os.makedirs(local_dir, exist_ok=True)
                
                # Create a unique filename with timestamp and phone
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = os.path.join(local_dir, f"face_{timestamp}_{phone_number}.json")
                
                # Save the face data
                with open(filename, 'w') as f:
                    json.dump({
                        "phone_number": phone_number,
                        "face_encoding": face_encoding,
                        "timestamp": timestamp
                    }, f)
                
                logger.info(f"Saved pending face registration for {phone_number} to {filename}")
                flash("Face saved locally. Registration will be processed when backend connection is restored.", "info")
                
                # Clear the session data
                if 'face_encoding' in flask_session:
                    flask_session.pop('face_encoding')
                    
                return redirect(url_for("door_entry"))
            except Exception as e:
                logger.error(f"Error saving local face data: {e}")
                flash("Error saving face data. Please try again later.", "danger")
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
            
            # Validate response status code
            if resp.status_code != 200:
                logger.error(f"Backend returned non-200 status: {resp.status_code}")
                logger.error(f"Response content: {resp.text}")
                flash("Error connecting to backend service.", "danger")
                return redirect(url_for("door_entry"))
                
            # Check for empty response
            if not resp.text.strip():
                logger.error("Backend returned empty response for face registration")
                flash("Backend service returned empty response. Please try again.", "danger")
                return redirect(url_for("door_entry"))
                
            # Parse response with error handling
            try:
                data = resp.json()
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to decode JSON response from face registration: {json_err}")
                logger.error(f"Response content: {resp.text}")
                flash("Invalid response from backend service. Please try again.", "danger")
                return redirect(url_for("door_entry"))
            
            if data.get("status") == "success":
                # Clear the session data
                if 'face_encoding' in flask_session:
                    flask_session.pop('face_encoding')
                
                flash("Face registered successfully. Please wait for OTP or admin approval.", "success")
                
                # Skip OTP flow if backend is not fully functional
                return redirect(url_for("door_entry"))
            else:
                flash(data.get("error", "Error registering face"), "danger")
        except Exception as e:
            logger.error(f"Error connecting to backend for face registration: {e}")
            flash("Error connecting to backend.", "danger")
            
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

    @app.route('/upload/<filename>')
    def uploaded_file(filename):
        return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

    @app.route('/video_feed')
    def video_feed():
        """Stream video directly from camera as multipart/x-mixed-replace content"""
        def generate_frames():
            # Create a camera capture object
            camera = None
            try:
                camera = cv2.VideoCapture(0)  # Use default camera
                if not camera.isOpened():
                    raise Exception("Could not open camera")
                    
                # Set resolution
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                # Set FPS to improve performance
                camera.set(cv2.CAP_PROP_FPS, 30)
                # Reduce buffer size to minimize latency
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Load face and eye detection models
                try:
                    # Look for haarcascades in common locations
                    cascade_paths = [
                        cv2.data.haarcascades,  # OpenCV's built-in path
                        '/usr/local/share/opencv4/haarcascades/',  # Common Linux path
                        '/usr/share/opencv/haarcascades/',  # Another Linux path
                        'C:/opencv/data/haarcascades/',  # Common Windows path
                    ]
                    
                    # Try to find the face cascade file
                    face_cascade_path = None
                    for path in cascade_paths:
                        test_path = os.path.join(path, 'haarcascade_frontalface_default.xml')
                        if os.path.exists(test_path):
                            face_cascade_path = test_path
                            break
                    
                    if face_cascade_path is None:
                        logger.warning("Couldn't find face cascade file, using default path")
                        face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    
                    # Try to find the eye cascade file
                    eye_cascade_path = None
                    for path in cascade_paths:
                        test_path = os.path.join(path, 'haarcascade_eye.xml')
                        if os.path.exists(test_path):
                            eye_cascade_path = test_path
                            break
                    
                    if eye_cascade_path is None:
                        logger.warning("Couldn't find eye cascade file, using default path")
                        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
                    
                    # Load the cascades
                    face_cascade = cv2.CascadeClassifier(face_cascade_path)
                    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
                    
                    # Check if the cascades loaded correctly
                    if face_cascade.empty():
                        logger.error(f"Failed to load face cascade from {face_cascade_path}")
                        raise Exception("Failed to load face cascade")
                    
                    if eye_cascade.empty():
                        logger.error(f"Failed to load eye cascade from {eye_cascade_path}")
                        # Continue without eye detection if it fails
                        logger.warning("Will continue without eye detection")
                    
                    logger.info("Camera and detection models initialized for video feed")
                except Exception as e:
                    logger.error(f"Error loading detection models: {e}")
                    # Continue without detection if models fail to load
                    face_cascade = None
                    eye_cascade = None
            except Exception as e:
                logger.error(f"Error opening camera: {e}")
                
            # Create a placeholder for when camera isn't working
            placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder, "Camera not available", (150, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            _, placeholder_buffer = cv2.imencode('.jpg', placeholder, [cv2.IMWRITE_JPEG_QUALITY, 90])
            placeholder_bytes = placeholder_buffer.tobytes()
            
            # Processing flag
            is_processing = False
            
            # For face detection performance
            frame_counter = 0
            detection_interval = 2  # Only detect faces every N frames
            last_faces = []  # Store the last detected faces
            last_eyes = []  # Store the last detected eyes
            
            try:
                while True:
                    if camera is None or not camera.isOpened():
                        # Return placeholder if camera not available
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
                        # Try to reopen camera
                        try:
                            if camera is not None:
                                camera.release()
                            camera = cv2.VideoCapture(0)
                        except Exception as e:
                            logger.error(f"Failed to reopen camera: {e}")
                        time.sleep(0.5)
                        continue
                    
                    # Read frame from camera
                    ret, frame = camera.read()
                    
                    if not ret or frame is None or frame.size == 0:
                        # Return placeholder if frame reading failed
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
                        time.sleep(0.1)
                        continue
                    
                    frame_with_detections = frame.copy()
                    
                    # Perform face and eye detection every few frames for better performance
                    if face_cascade is not None:
                        frame_counter += 1
                        if frame_counter % detection_interval == 0:
                            # Convert to grayscale for detection
                            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                            
                            # Apply histogram equalization to improve detection in varying lighting
                            gray = cv2.equalizeHist(gray)
                            
                            # Detect faces
                            faces = face_cascade.detectMultiScale(
                                gray,
                                scaleFactor=1.1,
                                minNeighbors=5,
                                minSize=(30, 30),
                                flags=cv2.CASCADE_SCALE_IMAGE
                            )
                            
                            # Update last faces if we found any
                            if len(faces) > 0:
                                last_faces = faces
                                
                                # Only detect eyes if we found faces
                                last_eyes = []
                                if eye_cascade is not None:
                                    for (x, y, w, h) in faces:
                                        # Region of interest for eyes
                                        roi_gray = gray[y:y+h, x:x+w]
                                        
                                        # Detect eyes within the face region
                                        eyes = eye_cascade.detectMultiScale(
                                            roi_gray,
                                            scaleFactor=1.1,
                                            minNeighbors=3,
                                            minSize=(15, 15),
                                            maxSize=(w//3, h//3)  # Eyes shouldn't be too large
                                        )
                                        
                                        # Store the eyes with their face position
                                        for (ex, ey, ew, eh) in eyes:
                                            last_eyes.append((x+ex, y+ey, ew, eh))
                    
                    # Draw the last detected faces on every frame
                    for (x, y, w, h) in last_faces:
                        # Draw face rectangle
                        cv2.rectangle(frame_with_detections, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        
                        # Add face label
                        cv2.putText(frame_with_detections, "Face", (x, y-10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        
                    # Draw eye rectangles
                    for (x, y, w, h) in last_eyes:
                        cv2.rectangle(frame_with_detections, (x, y), (x+w, y+h), (255, 0, 0), 2)
                        
                    # Add information about detected faces at the bottom of the frame
                    if len(last_faces) > 0:
                        cv2.putText(frame_with_detections, f"Faces: {len(last_faces)}, Eyes: {len(last_eyes)}", 
                                  (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    # Check if facial recognition is in progress
                    if 'UPLOAD_FOLDER' in app.config:
                        debug_frame_path = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_frame.jpg')
                        
                        # If debug frame exists and is fresh (modified in last 0.5 seconds), use it
                        try:
                            if (os.path.exists(debug_frame_path) and 
                                time.time() - os.path.getmtime(debug_frame_path) < 0.5):
                                
                                is_processing = True
                                # Use the debug frame instead of the camera frame
                                with open(debug_frame_path, 'rb') as f:
                                    frame_bytes = f.read()
                                    yield (b'--frame\r\n'
                                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                                    time.sleep(0.033)  # ~30fps
                                    continue
                        except Exception as e:
                            # If there's an error reading the debug frame, continue with camera frame
                            logger.error(f"Error reading debug frame: {e}")
                            
                    # If we get here, we'll use the camera frame directly
                    if is_processing:
                        # Add "Processing..." text if we were in processing mode
                        cv2.putText(frame_with_detections, "Processing...", (10, 30),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        is_processing = False
                    
                    # Convert frame to JPEG
                    try:
                        _, buffer = cv2.imencode('.jpg', frame_with_detections, [cv2.IMWRITE_JPEG_QUALITY, 90])
                        frame_bytes = buffer.tobytes()
                        
                        # Only yield if we got valid data
                        if frame_bytes and len(frame_bytes) > 0:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        else:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
                    except Exception as e:
                        logger.error(f"Error encoding frame: {e}")
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + placeholder_bytes + b'\r\n')
                    
                    # Small delay for frame rate control
                    time.sleep(0.033)  # ~30fps
            
            except Exception as e:
                logger.error(f"Error in video streaming: {e}")
            finally:
                # Clean up camera resources when done
                if camera is not None and camera.isOpened():
                    camera.release()
        
        # Return the streaming response
        return app.response_class(
            generate_frames(),
            mimetype='multipart/x-mixed-replace; boundary=frame'
        )

    return app 