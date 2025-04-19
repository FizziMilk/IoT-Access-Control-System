from flask import render_template, request, redirect, url_for, flash, session as flask_session, jsonify, Response
from datetime import datetime
import time
from utils import verify_otp_rest
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
import platform

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
    # Set up Qt environment once at startup
    setup_qt_environment()
    
    # Add integrated video feed with face recognition
    camera = None
    face_recognition_active = False
    face_recognition_result = None
    camera_lock = threading.Lock()
    
    def get_camera():
        """Get or initialize camera with proper error handling and fallbacks"""
        nonlocal camera, camera_needs_cleanup
        
        # If we already have a valid camera, return it
        if camera is not None and hasattr(camera, 'isOpened') and camera.isOpened():
            try:
                # Do a test read to make sure camera is still working
                ret, frame = camera.read()
                if ret and frame is not None:
                    return camera
                else:
                    logger.warning("Existing camera failed test read, will reinitialize")
                    # Only release_camera internally, not setting cleanup flag
                    with camera_lock:
                        if camera is not None:
                            try:
                                if hasattr(camera, 'isOpened') and camera.isOpened():
                                    camera.release()
                            except Exception as e:
                                logger.error(f"Error releasing camera: {e}")
                            finally:
                                camera = None
            except Exception as e:
                logger.error(f"Error testing existing camera: {e}")
                # Only release_camera internally, not setting cleanup flag
                with camera_lock:
                    if camera is not None:
                        try:
                            if hasattr(camera, 'isOpened') and camera.isOpened():
                                camera.release()
                        except Exception as e:
                            logger.error(f"Error releasing camera: {e}")
                        finally:
                            camera = None
            
        # Import the camera index from config
        from camera_config import CAMERA_INDEX
            
        # Try multiple camera indices, starting with the configured one
        camera_indices = [CAMERA_INDEX, 0, 1, 2]  # Try these indices in order
        
        # Make each index unique to avoid duplicates
        camera_indices = list(dict.fromkeys(camera_indices))
        
        for idx in camera_indices:
            try:
                logger.info(f"Attempting to open camera at index {idx}")
                cam = cv2.VideoCapture(idx)
                
                # Give camera time to initialize
                time.sleep(1.0)
                
                # Check if camera opened
                if cam.isOpened():
                    # Set camera properties
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                    
                    # Read a test frame to confirm camera is working
                    ret, test_frame = cam.read()
                    if not ret or test_frame is None:
                        logger.warning(f"Camera {idx} opened but failed to provide frame, trying next index")
                        cam.release()
                        continue
                    
                    logger.info(f"Successfully opened camera at index {idx}")
                    camera = cam
                    return camera
                else:
                    # Try to close it before moving to next index
                    cam.release()
                    logger.warning(f"Failed to open camera {idx}, trying next index")
                    
            except Exception as e:
                logger.error(f"Error initializing camera at index {idx}: {e}")
        
        # If we get here, all camera attempts failed
        logger.warning("Could not open any camera, creating fake camera feed")
        
        # Create a fake camera feed for UI testing
        class FakeCamera:
            def __init__(self):
                self.frame_count = 0
                self.width = 640
                self.height = 480
                
            def isOpened(self):
                return True
                
            def read(self):
                # Create a black frame with text
                frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                
                # Add some animation to show it's running
                self.frame_count += 1
                
                # Add timestamp and frame count
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(frame, f"NO CAMERA AVAILABLE", (50, 50), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Time: {timestamp}", (50, 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Frame: {self.frame_count}", (50, 150), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Draw a moving element
                angle = (self.frame_count % 360) * (np.pi / 180)
                cx, cy = self.width // 2, self.height // 2
                radius = 100
                x = int(cx + radius * np.cos(angle))
                y = int(cy + radius * np.sin(angle))
                cv2.circle(frame, (x, y), 20, (0, 255, 0), -1)
                
                return True, frame
                
            def release(self):
                logger.info("Fake camera released")
                pass
        
        # Use the fake camera
        camera = FakeCamera()
        return camera
    
    def release_camera():
        """Release camera resources safely"""
        nonlocal camera, camera_needs_cleanup
        
        with camera_lock:
            if camera is not None:
                try:
                    logger.info("Releasing camera resources")
                    if hasattr(camera, 'isOpened') and camera.isOpened():
                        # Try to read any remaining frames to clear buffer
                        for _ in range(5):
                            camera.read()
                        camera.release()
                        logger.info("Camera released successfully")
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
                finally:
                    camera = None
                    
                # On Linux, we won't forcibly kill processes - it's too aggressive
                # and can crash the application
                if platform.system() == 'Linux':
                    # Just log that we're waiting for resources to be freed
                    logger.info("Waiting for camera resources to be freed naturally")
                    # Add a small delay to allow OS to reclaim resources
                    time.sleep(0.5)
                
                # Camera has been released, so we don't need cleanup anymore
                camera_needs_cleanup = False
    
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
        nonlocal camera_needs_cleanup
        
        logger.info("Face recognition page requested")
        
        # Set the cleanup flag since we're initializing camera resources
        camera_needs_cleanup = True
        
        # Clean up any existing OpenCV windows and force camera release
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error cleaning up CV resources: {e}")
        
        # Check if testing mode - pass this to the template
        testing_mode = request.args.get('testing', 'false').lower() == 'true'
        
        # Return the template
        return render_template("face_recognition.html", testing_mode=testing_mode)
    
    @app.route('/video_feed')
    def video_feed():
        """
        Video streaming route for face recognition.
        """
        def generate_frames():
            nonlocal camera, camera_lock, face_recognition_active, camera_needs_cleanup
            
            # Initialize camera if needed, but don't set cleanup flag
            # as the video feed will manage its own resources
            with camera_lock:
                if camera is None:
                    camera = get_camera()
            
            # Create placeholder when needed
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)
            line_type = 2
            
            # Track camera reconnection attempts
            last_camera_attempt = time.time()
            reconnect_delay = 3.0  # seconds
            
            try:
                while True:
                    frame = None
                    current_time = time.time()
                    
                    # If face recognition is active, don't try to use the camera
                    if face_recognition_active:
                        # Create a placeholder frame showing face recognition is in progress
                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame, "Face Recognition in Progress...", (50, 240), 
                                  font, font_scale, (0, 255, 0), line_type)
                        cv2.putText(frame, "Please look at the camera and keep still", (50, 280), 
                                  font, font_scale, font_color, line_type)
                    else:
                        # Normal camera operation when face recognition is not active
                        with camera_lock:
                            # Only try to reconnect camera after delay
                            if (camera is None or (hasattr(camera, 'isOpened') and not camera.isOpened())) and \
                               (current_time - last_camera_attempt > reconnect_delay):
                                logger.info("Attempting to reconnect to camera")
                                last_camera_attempt = current_time
                                try:
                                    # If we already have a camera object that failed, try to release it first
                                    if camera is not None and hasattr(camera, 'release'):
                                        try:
                                            camera.release()
                                            # Don't set cleanup flag here as we're managing our own resources
                                        except:
                                            pass
                                    
                                    # Get a new camera
                                    camera = get_camera()
                                except Exception as e:
                                    logger.error(f"Error reconnecting to camera: {e}")
                                    camera = None
                            
                            # Try to get a frame if camera exists and is open
                            if camera is not None and hasattr(camera, 'isOpened') and camera.isOpened():
                                try:
                                    success, frame = camera.read()
                                    if not success or frame is None:
                                        frame = np.zeros((480, 640, 3), dtype=np.uint8)
                                        cv2.putText(frame, "Failed to read frame", (50, 240), 
                                                   font, font_scale, font_color, line_type)
                                except Exception as e:
                                    logger.error(f"Error reading frame: {e}")
                                    frame = None
                            else:
                                # No valid camera
                                frame = None
                        
                        # If frame is None, create a placeholder
                        if frame is None:
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame, "Camera not available", (50, 240), 
                                       font, font_scale, font_color, line_type)
                    
                    # Add timestamp
                    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                               (10, 30), font, font_scale, font_color, 1)
                    
                    # Convert to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in HTTP multipart response
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # Slight delay to reduce CPU usage
                    time.sleep(0.04)  # ~25 FPS
                    
            except Exception as e:
                logger.error(f"Error in video feed: {e}")
                
            finally:
                # Don't release the camera here as it's shared
                # Also don't set cleanup flag as this is a streaming endpoint
                pass
                        
        # Return the response
        return Response(generate_frames(),
                       mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/start-face-recognition', methods=['POST'])
    def start_face_recognition():
        """Start face recognition process on the video feed"""
        nonlocal face_recognition_active, face_recognition_result, camera, camera_needs_cleanup
        
        # Reset result
        face_recognition_result = None
        face_recognition_active = True
        
        # Set cleanup flag since we'll be using camera resources
        camera_needs_cleanup = True
        
        # Skip liveness check if in testing mode
        skip_liveness = request.form.get('skip_liveness', 'false').lower() == 'true'
        if skip_liveness:
            logger.warning("Skipping liveness detection - TESTING MODE ONLY")
        
        # Start recognition process in a background thread
        debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
        os.makedirs(debug_dir, exist_ok=True)
        
        def run_recognition_background():
            nonlocal face_recognition_result, face_recognition_active, camera, camera_needs_cleanup
            try:
                # Import directly here for cleaner import structure
                from face_recognition_process import run_face_recognition
                from camera_config import CAMERA_INDEX
                
                # Set a flag to stop the video feed
                with camera_lock:
                    if camera is not None:
                        logger.info("Releasing camera for face recognition")
                        # Release camera here before passing to face recognition process
                        release_camera()
                
                # Add a longer safety delay to ensure camera is fully released
                logger.info("Waiting for camera resources to be freed completely...")
                time.sleep(3.0)
                
                # Run face recognition (which will handle its own camera)
                result = run_face_recognition(
                    camera_index=CAMERA_INDEX, 
                    backend_url=backend_url, 
                    skip_liveness=skip_liveness,
                    debug_dir=debug_dir
                )
                
                # Store result
                face_recognition_result = result
                logger.info(f"Face recognition completed with result: {result.get('success')}")
                
                # Add a delay before allowing video feed to reacquire the camera
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Error running face recognition: {e}")
                face_recognition_result = {"success": False, "error": str(e)}
            finally:
                face_recognition_active = False
                # Set cleanup flag to false since we're done with camera resources
                camera_needs_cleanup = False
        
        # Start background thread
        recognition_thread = threading.Thread(target=run_recognition_background)
        recognition_thread.daemon = True
        recognition_thread.start()
        
        return jsonify({'status': 'started'})
    
    @app.route('/check-face-recognition-status')
    def check_face_recognition_status():
        """Check if face recognition is complete and get results"""
        nonlocal face_recognition_active, face_recognition_result
        
        if face_recognition_active:
            return jsonify({'status': 'processing'})
        
        if face_recognition_result:
            result = face_recognition_result
            return jsonify({'status': 'complete', 'result': result})
        
        return jsonify({'status': 'not_started'})
    
    @app.route('/process-face', methods=['POST'])
    def process_face():
        """Process facial recognition using the integrated video feed"""
        nonlocal face_recognition_result, camera_needs_cleanup
        
        # Set cleanup flag since we'll be working with camera resources
        camera_needs_cleanup = True
        
        try:
            # Skip liveness check if in testing mode
            skip_liveness = request.form.get('skip_liveness', 'false').lower() == 'true'
            
            # Check if we have results from the video feed
            if not face_recognition_result:
                flash("No face recognition results. Please try again.", "danger")
                return redirect(url_for("door_entry"))
            
            # Process results
            results = face_recognition_result
            
            # Clear results
            face_recognition_result = None
            
            # Process the results (same logic as before)
            if results['success']:
                if results['face_detected'] and 'face_encodings' in results and results['face_encodings']:
                    # Check if face passed liveness detection
                    if 'is_live' in results and not results['is_live'] and not skip_liveness:
                        logger.warning("Face failed liveness detection")
                        flash("Could not verify that this is a real face. Please try again with better lighting.", "warning")
                        return redirect(url_for("door_entry"))
                    
                    # We captured a live face, check if it was directly matched
                    if 'match' in results and results['match']:
                        # We have a direct match
                        match = results['match']
                        phone_number = match['name']  # Name field actually contains phone number
                        confidence = match['confidence']
                        
                        logger.info(f"Face directly matched with {phone_number}, confidence: {confidence:.2f}")
                        
                        # Check if this user is allowed direct access
                        resp = backend_session.get(f"{backend_url}/check-face-access/{phone_number}")
                        access_data = resp.json()
                        
                        if access_data.get("status") == "approved":
                            # Unlock door and log access
                            door_controller.unlock_door()
                            
                            # Log door unlock via facial recognition
                            try:
                                backend_session.post(f"{backend_url}/log-door-access", json={
                                    "user": phone_number,
                                    "method": "Facial Recognition",
                                    "status": "Unlocked",
                                    "details": f"Door unlocked via facial recognition, confidence: {confidence:.2f}"
                                })
                            except Exception as e:
                                logger.error(f"Error logging door access: {e}")
                                
                            flash("Face recognized! Door unlocked.", "success")
                            return render_template("door_unlocked.html")
                        else:
                            # No direct access - provide options
                            flash("Face recognized but additional verification needed.", "warning")
                            return render_template("verify_options.html", phone_number=phone_number,
                                                user_recognized=True, face_recognized=True)
                    
                    # No direct match, try to identify using backend
                    face_encoding = results['face_encodings'][0]  # Use the first face if multiple detected
                    
                    try:
                        # Send the face encoding to backend for identification
                        encoding_json = json.dumps(face_encoding)
                        encoding_base64 = base64.b64encode(encoding_json.encode('utf-8')).decode('utf-8')
                        
                        resp = backend_session.post(f"{backend_url}/identify-face", json={
                            "face_encoding": encoding_base64
                        })
                        data = resp.json()
                        
                        if data.get("identified") and data.get("user"):
                            # User recognized
                            phone_number = data.get("user").get("phone_number")
                            confidence = data.get("confidence", "N/A")
                            
                            # Check if this user is allowed direct access
                            resp = backend_session.get(f"{backend_url}/check-face-access/{phone_number}")
                            access_data = resp.json()
                            
                            if access_data.get("status") == "approved":
                                # Unlock door and log access
                                door_controller.unlock_door()
                                
                                # Log door unlock via facial recognition
                                try:
                                    backend_session.post(f"{backend_url}/log-door-access", json={
                                        "user": phone_number,
                                        "method": "Facial Recognition",
                                        "status": "Unlocked",
                                        "details": f"Door unlocked via facial recognition, confidence: {confidence}"
                                    })
                                except Exception as e:
                                    logger.error(f"Error logging door access: {e}")
                                    
                                flash("Face recognized! Door unlocked.", "success")
                                return render_template("door_unlocked.html")
                            else:
                                # No direct access - provide options
                                flash("Face recognized but additional verification needed.", "warning")
                                return render_template("verify_options.html", phone_number=phone_number,
                                                    user_recognized=True, face_recognized=True)
                        else:
                            # Face not recognized, allow registration
                            flask_session['face_encoding'] = face_encoding
                            flask_session['has_temp_face'] = True
                            
                            flash("Face not recognized. Please verify identity or register.", "warning")
                            return render_template("verify_options.html", user_recognized=False,
                                                face_recognized=False, has_face=True)
                    except Exception as e:
                        logger.error(f"Error identifying face: {e}")
                        logger.error(traceback.format_exc())
                        flash("Error identifying face. Please try again.", "danger")
                        return redirect(url_for("door_entry"))
                else:
                    # No face detected
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
    
    # Add a flag to track if cleanup is needed
    camera_needs_cleanup = False
    
    # Clean up resources when app exits
    @app.teardown_appcontext
    def cleanup_resources(exception=None):
        """Clean up all resources when application context tears down"""
        nonlocal camera, camera_needs_cleanup
        
        # Skip full cleanup on regular requests
        if not camera_needs_cleanup and request.endpoint not in [
            'face_recognition', 'start_face_recognition', 'process_face'
        ]:
            return
        
        logger.info("Cleaning up resources on app context teardown")
        
        # Release camera only if it exists and we're in a critical endpoint
        with camera_lock:
            if camera is not None and camera_needs_cleanup:
                try:
                    logger.info("Releasing camera resources")
                    if hasattr(camera, 'isOpened') and camera.isOpened():
                        # Try to read any remaining frames to clear buffer
                        for _ in range(5):
                            camera.read()
                        camera.release()
                        logger.info("Camera released successfully")
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
                finally:
                    camera = None
                    
                # On Linux, we won't forcibly kill processes - it's too aggressive
                # and can crash the application
                if platform.system() == 'Linux':
                    # Just log that we're waiting for resources to be freed
                    logger.info("Waiting for camera resources to be freed naturally")
                    # Add a small delay to allow OS to reclaim resources
                    time.sleep(0.5)
                
                # Reset the cleanup flag
                camera_needs_cleanup = False
                
        # Only destroy windows on specific endpoints
        if request.endpoint in ['face_recognition', 'process_face']:
            try:
                cv2.destroyAllWindows()
                # Force window cleanup
                cv2.waitKey(1)
            except Exception as e:
                logger.error(f"Error destroying windows: {e}")
                
        logger.info("Resource cleanup completed")

    return app 