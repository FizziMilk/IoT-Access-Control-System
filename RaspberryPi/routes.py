from flask import render_template, request, redirect, url_for, flash, session as flask_session, jsonify, Response, current_app
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
import copy
import requests
import face_recognition
from recognition_state import recognition_state

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
    camera_lock = threading.Lock()
    
    # Flag to track if cleanup is needed
    camera_needs_cleanup = False
    
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
        """Release camera resources safely and thoroughly"""
        nonlocal camera, camera_needs_cleanup
        
        with camera_lock:
            if camera is not None:
                try:
                    logger.info("Releasing camera resources")
                    if hasattr(camera, 'isOpened') and camera.isOpened():
                        # Try to read any remaining frames to clear buffer
                        for _ in range(3):  # Reduced from 5
                            try:
                                camera.read()
                            except Exception:
                                # Ignore errors during buffer clearing
                                pass
                        camera.release()
                        logger.info("Camera released successfully")
                except Exception as e:
                    logger.error(f"Error releasing camera: {e}")
                finally:
                    camera = None
                    
                # Platform-specific cleanup for more reliable camera release
                if platform.system() == 'Linux':
                    # Just log that we're waiting for resources to be freed
                    logger.info("Linux: Waiting for camera resources to be freed naturally")
                    # Add a delay to allow OS to reclaim resources
                    time.sleep(0.5)  # Reduced from 2.0
                elif platform.system() == 'Windows':
                    # Windows often needs more aggressive cleanup
                    logger.info("Windows: Performing additional cleanup steps")
                    time.sleep(0.3)  # Reduced from 1.0
                    try:
                        # Force OpenCV to release any hanging resources
                        cv2.destroyAllWindows()
                        # Multiple waitKey calls needed for better cleanup
                        for _ in range(3):  # Reduced from 5
                            cv2.waitKey(1)
                    except Exception as e:
                        logger.error(f"Error during Windows camera cleanup: {e}")
                
                # Camera has been released, update the cleanup flag
                camera_needs_cleanup = False
                
                # Additional cleanup to ensure resources are properly released
                try:
                    # Explicitly destroy any OpenCV windows
                    cv2.destroyAllWindows()
                    cv2.waitKey(1)
                except Exception as e:
                    logger.error(f"Error during additional camera cleanup: {e}")
                
                # Call garbage collection to help free up resources
                import gc
                gc.collect()
    
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
                
            # Check if we have recent recognition data to show (if from face recognition)
            if 'recent_recognition' in flask_session and flask_session['recent_recognition'].get('debug_frame'):
                # Keep the recognition data for display in the door_unlocked template
                return render_template("door_unlocked.html")
            elif 'debug_frame' in flask_session:
                # Create basic recognition data if only debug frame is available
                flask_session['recent_recognition'] = {
                    'user': phone_number,
                    'debug_frame': flask_session['debug_frame'],
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                return render_template("door_unlocked.html")
            else:
                # No recognition data available, just render the template
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

        # For GET requests, show the entry form
        if request.method == 'GET':
            # Get recent recognition data from session if available
            recent_recognition = None
            if 'recent_recognition' in flask_session:
                recent_recognition = flask_session.get('recent_recognition')
                # Clear from session after retrieving
                flask_session.pop('recent_recognition', None)
            
            return render_template('door_entry.html', 
                                  recent_recognition=recent_recognition)

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
        nonlocal camera, camera_needs_cleanup
        
        logger.info("Face recognition page requested")
        
        # Reset recognition state
        recognition_state.face_recognition_active = False
        recognition_state.face_recognition_result = None
        
        # Set the cleanup flag since we're initializing camera resources
        camera_needs_cleanup = True
        
        # Force camera release and cleanup
        release_camera()
        
        # Clean up any existing OpenCV windows
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)
        except Exception as e:
            logger.error(f"Error cleaning up CV resources: {e}")
        
        # Return the template
        return render_template("face_recognition.html")
    
    @app.route('/video_feed')
    def video_feed():
        """
        Video streaming route for face recognition.
        """
        def generate_frames():
            nonlocal camera, camera_lock
            
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
                    if recognition_state.face_recognition_active:
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
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cv2.putText(frame, timestamp, 
                               (10, 30), font, font_scale, font_color, 1)
                    
                    # Convert to JPEG
                    ret, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    
                    # Yield the frame in HTTP multipart response
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    
                    # Slight delay to reduce CPU usage - adjusted for better responsiveness
                    time.sleep(0.02)  # ~50 FPS (reduced from 0.04 which was ~25 FPS)
                    
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
        nonlocal camera, camera_needs_cleanup
        
        # If camera has been released, get a new one
        with camera_lock:
            if camera is None or not hasattr(camera, 'isOpened') or not camera.isOpened():
                logger.info("Getting new camera for face recognition")
                camera = get_camera()
                camera_needs_cleanup = True
        
        # Don't start if already running
        if recognition_state.recognition_running:
            return jsonify({
                "status": "error",
                "message": "Recognition process already running"
            })
        
        recognition_state.recognition_running = True
        recognition_state.face_recognition_active = True
        
        # Define how many frames to capture
        num_frames = 4  # Using 4 frames for better face encodings
        
        # Capture multiple frames
        frames = []
        try:
            for i in range(num_frames):
                # Check if camera is still valid before each frame capture
                with camera_lock:
                    if camera is None or not hasattr(camera, 'isOpened') or not camera.isOpened():
                        logger.warning("Camera was released, getting a new one")
                        camera = get_camera()
                        camera_needs_cleanup = True
                    
                    # Read frame while holding the lock
                    ret, frame = camera.read()
                
                if ret and frame is not None:
                    logger.info(f"Captured frame {i+1}/{num_frames}")
                    frames.append(frame)
                else:
                    logger.warning(f"Failed to capture frame {i+1}")
                    time.sleep(0.5)  # Wait a bit before retrying
            
            # Start recognition in background thread
            recognition_state.recognition_thread = threading.Thread(
                target=run_recognition_background, 
                args=(frames,)
            )
            recognition_state.recognition_thread.daemon = True
            recognition_state.recognition_thread.start()
            
            return jsonify({
                "status": "success", 
                "message": "Face recognition started"
            })
        except Exception as e:
            logger.error(f"Error starting face recognition: {e}")
            recognition_state.recognition_running = False
            recognition_state.face_recognition_active = False
            return jsonify({
                "status": "error",
                "message": f"Error starting face recognition: {e}"
            })
    
    @app.route('/check-face-recognition-status')
    def check_face_recognition_status():
        """Check the status of face recognition process"""
        
        if recognition_state.face_recognition_result:
            # If we have a result, return it and clear for next time
            result = recognition_state.face_recognition_result.copy()
            
            # Check if face was detected but not recognized, and passed liveness
            if (result.get("face_detected", False) and 
                not result.get("face_recognized", False) and 
                result.get("is_live", False) and 
                not result.get("face_too_small", False)):
                
                # Store the face encoding in session for registration
                if "face_encodings" in result and result["face_encodings"]:
                    flask_session["temp_face_encoding"] = result["face_encodings"][0]
                    flask_session["needs_registration"] = True
                    
                    # Add redirection information to the result
                    result["redirect"] = "/register-face"
                    result["redirect_reason"] = "face_not_recognized"
            
            # Store any error in the session for display
            if "error" in result:
                flask_session["error_message"] = result["error"]
            
            recognition_state.face_recognition_result = None
            recognition_state.face_recognition_active = False
            return jsonify(result)
        
        # If recognition is still active but no result yet
        if recognition_state.face_recognition_active:
            return jsonify({
                "status": "processing", 
                "message": "Face recognition in progress..."
            })
        
        # No recognition happening
        return jsonify({
            "status": "idle", 
            "message": "No face recognition in progress"
        })
    
    @app.route('/process-face/<face_id>', methods=['POST'])
    def process_face(face_id):
        """Process a detected face and check access rights"""
        try:
            nonlocal camera_needs_cleanup
            
            if not face_id:
                logger.warning("No face ID provided")
                flash('No face ID provided', 'error')
                return redirect(url_for('face_recognition'))
            
            logger.info(f"Processing face with ID: {face_id}")
            
            # Get debug frame if passed
            debug_frame = request.form.get('debug_frame')
            if debug_frame:
                logger.info(f"Debug frame provided: {debug_frame}")
                # Store in session for display in results
                flask_session['debug_frame'] = debug_frame
            
            if not recognition_state.face_recognition_result:
                logger.warning("No face recognition results available")
                flash('No face recognition results available', 'error')
                return redirect(url_for('face_recognition'))
            
            try:
                # Check if face was detected
                if not recognition_state.face_recognition_result.get('face_detected', False):
                    logger.warning("No face detected in result")
                    flash('No face was detected. Please try again.', 'error')
                    return redirect(url_for('face_recognition'))
                
                # Check if face encodings are available
                if 'face_encodings' not in recognition_state.face_recognition_result or not recognition_state.face_recognition_result['face_encodings']:
                    logger.warning("No face encodings found in result")
                    flash('No face encodings available', 'error')
                    return redirect(url_for('face_recognition'))
                
                # Store all the collected encodings in the session
                # We'll use these for registration if needed
                face_encodings = recognition_state.face_recognition_result['face_encodings']
                flask_session['face_encodings'] = face_encodings
                
                # Also store the primary encoding
                flask_session['face_encoding'] = face_encodings[0] if face_encodings else None
                
                # Check if the face was recognized
                if recognition_state.face_recognition_result.get('face_recognized', False) and recognition_state.face_recognition_result.get('match'):
                    match = recognition_state.face_recognition_result['match']
                    matched_phone = match.get('name')
                    confidence = match.get('confidence', 0)
                    
                    logger.info(f"Face matched with {matched_phone} (confidence: {confidence:.2f})")
                    flash(f"Face recognized! Welcome, {matched_phone}.", 'success')
                    
                    # Log the successful face recognition
                    try:
                        backend_session.post(f"{backend_url}/log-door-access", json={
                            "user": matched_phone,
                            "method": "Face Recognition",
                            "status": "Successful",
                            "details": f"Face recognized with confidence {confidence:.2f}"
                        })
                    except Exception as e:
                        logger.error(f"Error logging face access: {e}")
                    
                    # Redirect to door entry page
                    return redirect(url_for('door_entry'))
                else:
                    # Face wasn't recognized - offer registration
                    logger.info("Face not recognized, redirecting to registration")
                    return redirect(url_for('register_face'))
                
            except Exception as e:
                logger.error(f"Error processing face match: {str(e)}", exc_info=True)
                flash("An error occurred while processing your face. Please try again.", "error")
                return redirect(url_for('face_recognition'))
                
        except Exception as e:
            logger.error(f"Error processing face: {e}", exc_info=True)
            flash('An unexpected error occurred. Please try again.', 'error')
            return redirect(url_for('face_recognition'))
    
    @app.route('/register-face', methods=['GET', 'POST'])
    def register_face():
        # Check if we have a temp face encoding from recognition
        has_face_encoding = 'temp_face_encoding' in flask_session and flask_session['temp_face_encoding'] is not None
        needs_registration = flask_session.get('needs_registration', False)
        
        if request.method == 'POST':
            # Handle form submission
            name = request.form.get('name')
            phone = request.form.get('phone')
            
            if not name or not phone:
                flash('Please provide both name and phone number', 'error')
                return render_template('register_face.html', has_face_encoding=has_face_encoding)
            
            # Get the face encoding from session
            face_encoding = flask_session.get('temp_face_encoding')
            
            if not face_encoding:
                flash('No face encoding found. Please try again.', 'error')
                return redirect(url_for('face_recognition'))
            
            try:
                # Register the face with the backend
                result = register_face_with_backend(name, phone, face_encoding)
                
                if result.get('success'):
                    # Clear session data
                    flask_session.pop('temp_face_encoding', None)
                    flask_session.pop('needs_registration', None)
                    
                    flash('Face registered successfully!', 'success')
                    return redirect(url_for('index'))
                else:
                    flash(f'Error registering face: {result.get("error", "Unknown error")}', 'error')
                    return render_template('register_face.html', has_face_encoding=has_face_encoding)
                
            except Exception as e:
                logger.error(f"Error registering face: {e}")
                flash(f'Error registering face: {e}', 'error')
                return render_template('register_face.html', has_face_encoding=has_face_encoding)
        
        # GET request
        if not has_face_encoding and not needs_registration:
            # No face encoding in session, redirect to face recognition
            flash('Please complete face recognition first', 'warning')
            return redirect(url_for('face_recognition'))
        
        return render_template('register_face.html', 
                              has_face_encoding=has_face_encoding,
                              needs_registration=needs_registration)

    def register_face_with_backend(name, phone, face_encoding):
        """Register a new face with the backend server"""
        try:
            # Convert face encoding to list if it's a numpy array
            if isinstance(face_encoding, np.ndarray):
                face_encoding = face_encoding.tolist()
            
            # Prepare the payload
            payload = {
                'name': name,
                'phone': phone,
                'face_encoding': face_encoding
            }
            
            # Make request to backend
            response = requests.post(f"{backend_url}/api/register-face", json=payload)
            
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Backend returned status {response.status_code}: {response.text}")
                return {'success': False, 'error': f"Backend error: {response.text}"}
            
        except Exception as e:
            logger.error(f"Error communicating with backend: {e}")
            return {'success': False, 'error': str(e)}
    
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
    
    @app.route('/reset-camera-resources', methods=['POST'])
    def reset_camera_resources():
        """
        Endpoint to force reset camera resources - useful when camera gets stuck
        """
        nonlocal camera, camera_needs_cleanup
        
        try:
            logger.info("Manually resetting camera resources")
            
            # Reset recognition state
            recognition_state.face_recognition_active = False
            recognition_state.face_recognition_result = None
            
            # Set cleanup flag
            camera_needs_cleanup = True
            
            # Force camera release - this will also destroy OpenCV windows
            release_camera()
            
            # Reduced delay to ensure resources are freed
            time.sleep(1.0)  # Reduced from 3.0 seconds
            
            # Platform-specific cleanup approaches
            if platform.system() == 'Linux':
                try:
                    # List video devices to check their availability
                    os.system('ls -la /dev/video* > /dev/null 2>&1')
                except Exception as e:
                    logger.error(f"Error checking video devices: {e}")
            elif platform.system() == 'Windows':
                try:
                    # Windows-specific cleanup
                    logger.info("Windows: Performing additional resource cleanup")
                    # Force destroy all windows multiple times
                    for _ in range(2):  # Reduced from 3
                        cv2.destroyAllWindows()
                        cv2.waitKey(1)
                    # Force garbage collection
                    import gc
                    gc.collect()
                    time.sleep(0.3)  # Reduced from 1.0
                except Exception as e:
                    logger.error(f"Error during Windows cleanup: {e}")
            
            # Try to open and close the camera multiple times as a reset mechanism
            attempts = 0
            max_attempts = 2  # Reduced from 3
            success = False
            
            while attempts < max_attempts and not success:
                try:
                    logger.info(f"Performing camera open/close cycle (attempt {attempts+1}/{max_attempts})")
                    # Import camera index from config
                    from camera_config import CAMERA_INDEX
                    
                    # Try configured index first, then fallback to 0
                    for idx in [CAMERA_INDEX, 0]:
                        try:
                            temp_camera = cv2.VideoCapture(idx)
                            if temp_camera.isOpened():
                                # Read a few frames to clear buffers
                                for _ in range(2):  # Reduced from 3
                                    temp_camera.read()
                                temp_camera.release()
                                logger.info(f"Camera open/close cycle completed successfully with index {idx}")
                                success = True
                                break
                        except Exception as e:
                            logger.error(f"Error in camera cycle with index {idx}: {e}")
                    
                    # If we didn't succeed with any index, wait before retrying
                    if not success:
                        time.sleep(0.3)  # Reduced from 1.0
                    
                except Exception as e:
                    logger.error(f"Error in camera open/close cycle: {e}")
                    time.sleep(0.3)  # Reduced from 1.0
                
                attempts += 1
            
            # Final cleanup
            cv2.destroyAllWindows()
            for _ in range(2):  # Reduced from 5
                cv2.waitKey(1)
            
            # Force camera to None to ensure full reinitialization on next access
            with camera_lock:
                camera = None
            
            return jsonify({
                "status": "success", 
                "message": "Camera resources reset",
                "success": success
            })
            
        except Exception as e:
            logger.error(f"Error resetting camera resources: {e}", exc_info=True)
            return jsonify({"status": "error", "message": str(e)}), 500
    
    @app.route('/capture-preview-frame', methods=['POST'])
    def capture_preview_frame():
        """Capture a single frame to display during processing"""
        nonlocal camera, camera_needs_cleanup
        
        try:
            logger.info("Capturing preview frame before facial recognition")
            
            # Ensure debug frames directory exists
            debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate timestamp for the frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Get camera instance with proper lock
            with camera_lock:
                if camera is None or not hasattr(camera, 'isOpened') or not camera.isOpened():
                    camera = get_camera()
                    # Don't set cleanup flag here - we need the camera for face recognition
                
                if camera is None or not hasattr(camera, 'read'):
                    logger.error("Camera not available for preview capture")
                    return jsonify({"success": False, "error": "Camera not available"}), 500
                    
                # Read frame while still holding the lock
                ret, frame = camera.read()
                if not ret or frame is None:
                    logger.error("Failed to capture preview frame")
                    return jsonify({"success": False, "error": "Failed to capture frame"}), 500
            
            # Add timestamp to the frame
            cv2.putText(
                frame, 
                timestamp, 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 255, 0), 
                2
            )
            
            # Save the frame
            filename = f"{debug_dir}/preview_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            logger.info(f"Preview frame saved to {filename}")
            
            # Return success with timestamp
            return jsonify({
                "success": True, 
                "timestamp": timestamp,
                "filename": f"preview_{timestamp}.jpg"
            })
            
        except Exception as e:
            logger.error(f"Error capturing preview frame: {e}", exc_info=True)
            return jsonify({"success": False, "error": str(e)}), 500
    
    # Clean up resources when app exits
    @app.teardown_appcontext
    def cleanup_resources(exception=None):
        """Clean up all resources when application context tears down"""
        nonlocal camera, camera_needs_cleanup
        
        # Skip full cleanup on regular requests or if outside request context
        if not camera_needs_cleanup:
            # Check if we're in a request context before accessing request.endpoint
            from flask import has_request_context
            if not has_request_context() or request.endpoint not in [
                'face_recognition', 'start_face_recognition', 'process_face'
            ]:
                return
        
        logger.info("Cleaning up resources on app context teardown")
        
        # Release camera only if it exists and we're in a critical endpoint or cleanup is needed
        with camera_lock:
            if camera is not None and camera_needs_cleanup:
                try:
                    logger.info("Releasing camera resources")
                    if hasattr(camera, 'isOpened') and camera.isOpened():
                        # Try to read any remaining frames to clear buffer - reduced read count
                        for _ in range(2):  # Reduced from 5
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
                    # Add a small delay to allow OS to reclaim resources - reduced wait time
                    time.sleep(0.3)  # Reduced from 0.5
                
                # Reset the cleanup flag
                camera_needs_cleanup = False
                
        # Only destroy windows on specific endpoints if in request context
        from flask import has_request_context
        if has_request_context() and request.endpoint in ['face_recognition', 'process_face']:
            try:
                cv2.destroyAllWindows()
                # Force window cleanup
                cv2.waitKey(1)
            except Exception as e:
                logger.error(f"Error destroying windows: {e}")
                
        logger.info("Resource cleanup completed")

    # Define the run_recognition_background function inside setup_routes
    def run_recognition_background(frames):
        try:
            # Set up debug directory for frame saving
            debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Import camera config values
            from camera_config import MIN_FACE_WIDTH, MIN_FACE_HEIGHT
            
            # Timestamp for debugging
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Import face recognition modules
            from face_recognition_process import WebRecognition, load_known_faces, save_debug_frame
            import face_recognition
            import numpy as np
            
            # Initialize recognition system and load known faces
            recognition = WebRecognition()
            encodings, names = load_known_faces(backend_url)
            if encodings and names:
                recognition.load_encodings(encodings, names)
            else:
                logger.warning("No face encodings loaded from backend")
            
            # Skip liveness check flag (for testing)
            skip_liveness = False
            
            # Process each frame to find the best one for recognition
            best_frame_index = -1
            best_face_width = 0
            best_face_height = 0
            
            # Find the best frame with the largest face
            try:
                for i, frame in enumerate(frames):
                    # Get face locations for this specific frame
                    face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
                    
                    # Skip if no faces found in this frame
                    if not face_locations:
                        continue
                        
                    # Use the first (largest) face in the frame
                    face_location = face_locations[0]
                    
                    # Extract coordinates (top, right, bottom, left)
                    top, right, bottom, left = face_location
                    
                    # Calculate face dimensions
                    face_width = right - left
                    face_height = bottom - top
                    
                    # Handle potential array types
                    if isinstance(face_width, np.ndarray):
                        face_width = float(face_width.item()) if face_width.size == 1 else float(face_width.mean())
                    if isinstance(face_height, np.ndarray):
                        face_height = float(face_height.item()) if face_height.size == 1 else float(face_height.mean())
                    
                    # Compare as scalars - find the largest face
                    if float(face_width) > float(best_face_width) or float(face_height) > float(best_face_height):
                        best_frame_index = i
                        best_face_width = face_width
                        best_face_height = face_height
                        
                    logger.info(f"Frame {i}: Face dimensions {face_width}x{face_height}")
            except Exception as e:
                logger.error(f"Error finding best frame: {e}")
                import traceback
                logger.error(f"Best frame selection traceback: {traceback.format_exc()}")
            
            logger.info(f"Best frame index: {best_frame_index}")
            logger.info(f"Best face dimensions: {best_face_width}x{best_face_height} pixels")
            
            # Check if we found any usable faces
            if best_frame_index == -1:
                logger.warning("No faces detected in any frame")
                recognition_state.face_recognition_result = {
                    "success": False,
                    "face_detected": False,
                    "error": "No face detected"
                }
                recognition_state.recognition_running = False
                recognition_state.face_recognition_active = False
                return
            
            # Check if the face is too small for reliable recognition
            if best_face_width < MIN_FACE_WIDTH or best_face_height < MIN_FACE_HEIGHT:
                logger.warning(f"Face too small for reliable recognition: {best_face_width}x{best_face_height}")
                result = {
                    "success": False,
                    "face_detected": True,
                    "face_recognized": False,
                    "face_too_small": True,
                    "error": "Face is too small for reliable recognition"
                }
                
                # Add distance feedback
                if best_face_width < MIN_FACE_WIDTH * 0.5 or best_face_height < MIN_FACE_HEIGHT * 0.5:
                    result["distance_feedback"] = "much_too_far"
                else:
                    result["distance_feedback"] = "too_far"
                
                # Save debug frame
                frame_filename = f"{debug_dir}/frame_face_too_small_{timestamp}.jpg"
                best_frame = frames[best_frame_index]
                face_loc = face_recognition.face_locations(best_frame, number_of_times_to_upsample=2)[0]
                cv2.rectangle(best_frame, (face_loc[3], face_loc[0]), 
                              (face_loc[1], face_loc[2]), (0, 255, 255), 2)
                cv2.putText(best_frame, "Face too small - Please move closer", 
                            (face_loc[3], face_loc[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imwrite(frame_filename, best_frame)
                
                recognition_state.face_recognition_result = result
                recognition_state.recognition_running = False
                recognition_state.face_recognition_active = False
                return
            
            # Get the best frame and face location
            best_frame = frames[best_frame_index]
            all_face_locations = face_recognition.face_locations(best_frame, number_of_times_to_upsample=2)
            best_face_location = all_face_locations[0] if all_face_locations else None
            
            # Extract face encoding from the best frame
            face_encodings = []
            if best_face_location:
                best_face_encoding = face_recognition.face_encodings(best_frame, [best_face_location])
                if best_face_encoding:
                    face_encodings.append(best_face_encoding[0])
            
            # Helper function to sanitize values from arrays
            def sanitize_value(value):
                if isinstance(value, np.ndarray):
                    if value.size == 1:
                        return value.item()
                    else:
                        return float(value.mean())
                return value
            
            # Perform liveness check on ONLY 2 frames for speed
            liveness_results = []
            frame_liveness_results = []
            
            try:
                if not skip_liveness:
                    logger.info("Performing liveness detection on 2 of 4 frames")
                    
                    # Select only 2 frames for liveness - first and last for maximum time difference
                    liveness_frames = [frames[0], frames[-1]]
                    frame_indices = [0, len(frames)-1]
                    
                    # Process the selected frames
                    for idx, i in enumerate(frame_indices):
                        frame = frames[i]
                        try:
                            if frame is not None:
                                face_locs = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
                                if face_locs:
                                    face_loc = face_locs[0]
                                    # Check liveness for this frame
                                    try:
                                        liveness_result = recognition.liveness_detector.check_face_liveness(frame, face_loc)
                                        logger.info(f"Frame {i+1} liveness result: {liveness_result}")
                                        
                                        # Ensure is_live is a scalar boolean
                                        is_live = liveness_result.get("is_live", False)
                                        if isinstance(is_live, np.ndarray):
                                            is_live = is_live.any()  # Use .any() explicitly for array evaluation
                                        
                                        # Store the result with scalar values
                                        frame_liveness_results.append({
                                            "frame_index": i,
                                            "is_live": bool(is_live),
                                            "confidence": float(sanitize_value(liveness_result.get("confidence", 0))),
                                            "texture_score": float(sanitize_value(liveness_result.get("texture_score", 0)))
                                        })
                                        
                                        # For debug frame visualization
                                        liveness_results.append({
                                            "is_live": bool(is_live),
                                            "confidence": float(sanitize_value(liveness_result.get("confidence", 0))),
                                            "texture_score": float(sanitize_value(liveness_result.get("texture_score", 0)))
                                        })
                                    except Exception as e:
                                        logger.error(f"Error in liveness detection for frame {i+1}: {e}")
                                        import traceback
                                        logger.error(f"Liveness detection traceback: {traceback.format_exc()}")
                        except Exception as e:
                            logger.error(f"Error processing frame {i+1}: {e}")
                            logger.error(f"Frame processing traceback: {traceback.format_exc()}")
                    
                    # Determine overall liveness using a voting system
                    live_votes = sum(1 for result in frame_liveness_results if bool(result.get("is_live", False)))
                    total_votes = len(frame_liveness_results)
                    
                    logger.info(f"Liveness votes: {live_votes}/{total_votes}")
                    
                    # Calculate average confidence
                    avg_confidence = 0
                    if total_votes > 0:
                        confidence_sum = 0
                        for result in frame_liveness_results:
                            confidence = sanitize_value(result.get("confidence", 0))
                            confidence_sum += confidence
                        avg_confidence = confidence_sum / total_votes
                    
                    # Need majority of frames to show liveness
                    is_live = (live_votes > total_votes / 2) if total_votes > 0 else False
                    
                    logger.info(f"Overall liveness result: {live_votes}/{total_votes} frames live, avg confidence {avg_confidence:.2f}")
                    
                    # Create combined liveness result
                    combined_liveness_result = {
                        "is_live": bool(is_live),
                        "confidence": float(avg_confidence),
                        "live_frame_count": int(live_votes),
                        "total_frame_count": int(total_votes),
                        "frame_results": frame_liveness_results
                    }
                else:
                    logger.info("Skipping liveness detection")
                    # Create a dummy result for skipped liveness
                    combined_liveness_result = {
                        "is_live": True,
                        "confidence": 1.0,
                        "skipped": True,
                        "live_frame_count": 1,  # Count all faces as live
                        "total_frame_count": 1
                    }
            except Exception as e:
                logger.error(f"Error in liveness check processing: {e}")
                import traceback
                logger.error(f"Liveness processing traceback: {traceback.format_exc()}")
                # Provide a fallback liveness result
                combined_liveness_result = {
                    "is_live": True,  # Default to True to avoid blocking recognition
                    "confidence": 0.5,
                    "error": str(e),
                    "skipped": True
                }
            
            # Process all 4 frames for face encodings to get better recognition
            # Extract face encodings from all frames where a face is detected
            all_face_encodings = []
            for i, frame in enumerate(frames):
                try:
                    frame_face_locations = face_recognition.face_locations(frame, number_of_times_to_upsample=2)
                    if frame_face_locations:
                        frame_face_encodings = face_recognition.face_encodings(frame, [frame_face_locations[0]])
                        if frame_face_encodings:
                            all_face_encodings.append(frame_face_encodings[0])
                            logger.info(f"Added face encoding from frame {i+1}")
                except Exception as e:
                    logger.error(f"Error extracting face encoding from frame {i+1}: {e}")
            
            # Use all encodings for improved recognition accuracy
            match_results = []
            for face_encoding in all_face_encodings:
                try:
                    match_result = recognition.identify_face(face_encoding=face_encoding)
                    
                    # Ensure all values in match are scalars
                    if match_result and match_result.get("match"):
                        match_data = match_result["match"]
                        scalar_match = {
                            "name": str(match_data.get("name", "")),
                            "confidence": float(sanitize_value(match_data.get("confidence", 0))),
                            "distance": float(sanitize_value(match_data.get("distance", 1.0)))
                        }
                        
                        # Add other fields if present
                        if "best_single_confidence" in match_data:
                            scalar_match["best_single_confidence"] = float(sanitize_value(match_data.get("best_single_confidence", 0)))
                        if "num_encodings" in match_data:
                            scalar_match["num_encodings"] = int(match_data.get("num_encodings", 1))
                            
                        match_result["match"] = scalar_match
                        
                    match_results.append(match_result)
                except Exception as e:
                    logger.error(f"Error during face identification: {e}")
                    match_results.append(None)
            
            # Find the best match
            best_match = None
            best_confidence = 0
            for match in match_results:
                # Make sure we're dealing with scalar values when comparing
                if match and match.get("match"):
                    confidence = match["match"].get("confidence", 0)
                    # Ensure confidence is a scalar value
                    if isinstance(confidence, np.ndarray):
                        if confidence.size == 1:
                            confidence = float(confidence.item())
                        else:
                            confidence = float(confidence.mean())
                    # Now compare with scalar values
                    if confidence > best_confidence:
                        best_match = match
                        best_confidence = confidence
            
            # Save final debug frame
            faces = [best_face_location] if best_face_location else []
            
            # Create a properly serialized match object for visualization
            match_for_visualization = None
            if best_match and best_match.get("match"):
                # Create a copy of the match with scalar values
                match_data = best_match["match"]
                match_for_visualization = {
                    "name": str(match_data.get("name", "")),
                    "confidence": float(match_data.get("confidence", 0)),
                    "distance": float(match_data.get("distance", 0))
                }
                
            matches = [{"match": match_for_visualization}] if match_for_visualization else []
            frame_filename = f"{debug_dir}/frame_final_{timestamp}.jpg"
            
            # Ensure liveness result is properly serialized for visualization
            visualization_liveness = None
            if liveness_results and len(liveness_results) > 0:
                liveness_result = liveness_results[0]
                # Create a serialized copy with scalar values
                visualization_liveness = {
                    "is_live": bool(liveness_result.get("is_live", False)),
                    "confidence": float(liveness_result.get("confidence", 0)),
                    "texture_score": float(liveness_result.get("texture_score", 0))
                }
            else:
                # Use combined result if available
                visualization_liveness = {
                    "is_live": bool(combined_liveness_result.get("is_live", False)),
                    "confidence": float(combined_liveness_result.get("confidence", 0))
                }
            
            save_debug_frame(best_frame, frame_filename, 
                            faces=faces, 
                            liveness_results=[visualization_liveness] if visualization_liveness else None,
                            matches=matches)
            
            # Prepare final result
            result = {
                "success": True,
                "face_detected": True,
                "face_recognized": best_match is not None,
                "liveness_check_passed": combined_liveness_result.get("is_live", False),
                "face_too_small": False,
                "face_encodings": [encoding.tolist() for encoding in all_face_encodings],
                "face_locations": [best_face_location] if best_face_location else [],
                "debug_frame": os.path.basename(frame_filename)
            }
            
            # Add match information if found
            if best_match and best_match.get("match"):
                result["match"] = best_match["match"]
            
            # Add liveness results
            result["is_live"] = combined_liveness_result.get("is_live", False)
            result["liveness_results"] = combined_liveness_result
            
            # Store the result
            recognition_state.face_recognition_result = result
            
        except Exception as e:
            logger.error(f"Error during face recognition: {e}")
            import traceback
            logger.error(f"Face recognition traceback: {traceback.format_exc()}")
            recognition_state.face_recognition_result = {
                "success": False,
                "error": str(e)
            }
        
        recognition_state.recognition_running = False
        recognition_state.face_recognition_active = False 

    return app 