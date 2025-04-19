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
# Rename this import to avoid shadowing with a potential function
import face_recognition as face_recog
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
    
    # Define API URL for backend requests
    API_URL = backend_url
    
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
    def face_recognition_page():
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
                    # Convert to RGB for face_recognition library
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
    
    @app.route('/check-face-recognition-status', methods=['GET'])
    def check_face_recognition_status():
        """Check the status of face recognition process and return results"""
        try:
            # Get the current recognition result
            result = {
                "success": True,
                "recognition_active": recognition_state.face_recognition_active,
                "result": None,
                "redirect": None
            }
            
            # If recognition is not active and we have a result
            if not recognition_state.face_recognition_active and recognition_state.face_recognition_result:
                result["result"] = recognition_state.face_recognition_result
                
                # Check if we have a face that passed liveness but was not recognized
                if (result["result"].get("success") and 
                    result["result"].get("face_detected") and 
                    result["result"].get("is_live") and 
                    not result["result"].get("face_recognized")):
                    
                    logger.info("Unknown live face detected - redirecting to registration")
                    
                    # Store face encoding in session for registration if available
                    face_encodings = result["result"].get("face_encodings")
                    if face_encodings:
                        flask_session["pending_face_encoding"] = face_encodings
                        logger.info("Face encoding stored in session for registration")
                    
                    # Add redirect information to the result
                    result["redirect"] = "/register-face"
                    result["message"] = "Unknown face detected - redirecting to registration"
            
            return jsonify(result)
        except Exception as e:
            logger.error(f"Error checking face recognition status: {str(e)}")
            return jsonify({
                "success": False,
                "error": f"Error checking face recognition status: {str(e)}"
            })
    
    @app.route('/process-face/<face_id>', methods=['POST'])
    def process_face(face_id):
        """Process a detected face and check access rights"""
        try:
            nonlocal camera_needs_cleanup
            
            if not face_id:
                logger.warning("No face ID provided")
                flash('No face ID provided', 'error')
                return redirect(url_for('face_recognition_page'))
            
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
                return redirect(url_for('face_recognition_page'))
            
            try:
                # Check if face was detected
                if not recognition_state.face_recognition_result.get('face_detected', False):
                    logger.warning("No face detected in result")
                    flash('No face was detected. Please try again.', 'error')
                    return redirect(url_for('face_recognition_page'))
                
                # Check if face encodings are available
                if 'face_encodings' not in recognition_state.face_recognition_result or not recognition_state.face_recognition_result['face_encodings']:
                    logger.warning("No face encodings found in result")
                    flash('No face encodings available', 'error')
                    return redirect(url_for('face_recognition_page'))
                
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
                return redirect(url_for('face_recognition_page'))
                
        except Exception as e:
            logger.error(f"Error processing face: {e}", exc_info=True)
            flash('An unexpected error occurred. Please try again.', 'error')
            return redirect(url_for('face_recognition_page'))
    
    @app.route('/register-face', methods=['GET', 'POST'])
    def register_face():
        """Handle face registration for unknown detected faces."""
        if request.method == 'GET':
            # Render the registration form
            return render_template('register_face.html')
        
        # For POST requests (form submission)
        try:
            # Get data from request
            data = request.json if request.is_json else request.form
            name = data.get('name')
            device_id = data.get('device_id')
            access_level = int(data.get('access_level', 1))
            
            if not name or not device_id:
                return jsonify({'success': False, 'error': 'Missing required fields'})
            
            # Get face encoding from session
            face_encoding = flask_session.get('face_encoding')
            if face_encoding is None:
                logger.error("No face encoding found in session")
                return jsonify({'success': False, 'error': 'No face data available'})
            
            # Convert face encoding from string to numpy array
            try:
                face_encoding = np.fromstring(face_encoding, dtype=np.float64)
            except Exception as e:
                logger.error(f"Error converting face encoding: {str(e)}")
                return jsonify({'success': False, 'error': 'Invalid face data'})
            
            # Save to database
            try:
                # Connect to database
                conn = get_db_connection()
                cursor = conn.cursor()
                
                # Insert new user
                cursor.execute(
                    "INSERT INTO users (name, face_encoding) VALUES (?, ?)",
                    (name, face_encoding.tobytes())
                )
                user_id = cursor.lastrowid
                
                # Add access for the specified device
                cursor.execute(
                    "INSERT INTO access_rights (user_id, device_id, access_level) VALUES (?, ?, ?)",
                    (user_id, device_id, access_level)
                )
                
                # Commit changes
                conn.commit()
                conn.close()
                
                # Clear session data
                flask_session.pop('face_encoding', None)
                
                logger.info(f"Successfully registered new user: {name} with access to device {device_id}")
                return jsonify({'success': True, 'message': 'Face registered successfully'})
                
            except Exception as e:
                logger.error(f"Database error during registration: {str(e)}")
                return jsonify({'success': False, 'error': f'Database error: {str(e)}'})
                
        except Exception as e:
            logger.error(f"Error in face registration: {str(e)}")
            return jsonify({'success': False, 'error': str(e)})
    
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
                'face_recognition_page', 'start_face_recognition', 'process_face'
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
        if has_request_context() and request.endpoint in ['face_recognition_page', 'process_face']:
            try:
                cv2.destroyAllWindows()
                # Force window cleanup
                cv2.waitKey(1)
            except Exception as e:
                logger.error(f"Error destroying windows: {e}")
                
        logger.info("Resource cleanup completed")

    # Define the run_recognition_background function inside setup_routes
    def run_recognition_background(frames):
        """Run face recognition in the background"""
        try:
            # Set recognition state to active
            recognition_state.face_recognition_active = True
            recognition_state.face_recognition_result = None
            
            logger.info("Starting face recognition background process")
            start_time = time.time()
            
            # Initialize result structure
            result = {
                "success": False,
                "face_detected": False,
                "face_recognized": False,
                "is_live": False,
                "matched_users": [],
                "timestamp": datetime.now().isoformat()
            }
            
            # Detect faces in the frames that were provided
            face_locations = []
            rgb_frames = []
            
            for frame in frames:
                # Convert to RGB for face_recognition library if needed
                if frame.shape[2] == 3 and frame[0,0].size == 3:  # Check if BGR
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    rgb_frame = frame  # Already RGB
                
                # Detect faces in this frame (using HOG which is faster than CNN)
                locations = face_recog.face_locations(rgb_frame, model="hog")
                
                if locations:
                    rgb_frames.append(rgb_frame)
                    face_locations.append(locations)
            
            if not rgb_frames:
                logger.warning("No faces detected in any frames")
                result["error"] = "No faces detected in frames"
                recognition_state.face_recognition_result = result
                return
            
            # Find the frame with the largest face for better recognition
            try:
                best_frame_start = time.time()
                best_frame_index = -1
                largest_face_area = 0
                best_face_location = None
                
                for i, locations in enumerate(face_locations):
                    if not locations:
                        continue
                        
                    for location in locations:
                        # Get face dimensions
                        top, right, bottom, left = location
                        face_width = right - left
                        face_height = bottom - top
                        face_area = face_width * face_height
                        
                        if face_area > largest_face_area:
                            largest_face_area = face_area
                            best_frame_index = i
                            best_face_location = location
                
                best_frame_time = time.time() - best_frame_start
                logger.info(f"Found best frame in {best_frame_time:.2f} seconds")
                
                if best_frame_index == -1 or not best_face_location:
                    logger.warning("No faces detected in frames")
                    result["error"] = "No faces detected in frames"
                    recognition_state.face_recognition_result = result
                    return
                    
                # Get the best frame and its face location
                best_frame = rgb_frames[best_frame_index]
                
                # Check if face is too small (min 100x100 pixels for reliable recognition)
                top, right, bottom, left = best_face_location
                face_width = right - left
                face_height = bottom - top
                
                # Log face dimensions
                logger.info(f"Best face dimensions: {face_width}x{face_height} pixels")
                
                if face_width < 100 or face_height < 100:
                    logger.warning(f"Face too small for reliable recognition: {face_width}x{face_height}")
                    result["face_detected"] = True
                    result["face_too_small"] = True
                    result["error"] = "Face is too small for reliable recognition"
                    
                    # Save a debug frame with rectangle and text
                    debug_frame = rgb_frames[best_frame_index].copy()
                    debug_frame = cv2.cvtColor(debug_frame, cv2.COLOR_RGB2BGR)
                    cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(debug_frame, f"Too small: {face_width}x{face_height}", 
                               (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"static/debug/face_too_small_{timestamp}.jpg", debug_frame)
                    
                    recognition_state.face_recognition_result = result
                    return
                
                # Check liveness (only on the best frame to save time)
                liveness_start = time.time()
                is_live = check_liveness(best_frame, best_face_location)
                liveness_time = time.time() - liveness_start
                logger.info(f"Liveness check completed in {liveness_time:.2f} seconds: {is_live}")
                
                result["face_detected"] = True
                result["is_live"] = is_live
                
                if not is_live:
                    result["error"] = "Face liveness check failed"
                    result["success"] = True  # Process completed successfully, just not a live face
                    recognition_state.face_recognition_result = result
                    return
                
                # Extract face encodings from best frame (only encode the best face)
                encoding_start = time.time()
                face_encodings = face_recog.face_encodings(best_frame, [best_face_location])
                encoding_time = time.time() - encoding_start
                logger.info(f"Face encoding extracted in {encoding_time:.2f} seconds")
                
                if not face_encodings:
                    logger.warning("Failed to extract face encodings")
                    result["error"] = "Failed to extract face features"
                    recognition_state.face_recognition_result = result
                    return
                
                # Store the face encoding in the result
                result["face_encodings"] = face_encodings[0].tolist()
                
                # Load known face encodings from backend
                try:
                    known_faces_response = requests.get(f"{API_URL}/users/encodings")
                    if known_faces_response.status_code != 200:
                        logger.error(f"Failed to get known faces: {known_faces_response.status_code}")
                        result["error"] = "Failed to retrieve registered users"
                        recognition_state.face_recognition_result = result
                        return
                    
                    known_faces_data = known_faces_response.json()
                    
                    # Check if we have any registered users
                    if not known_faces_data or not known_faces_data.get("users", []):
                        logger.info("No registered users found")
                        result["message"] = "No registered users found"
                        result["success"] = True
                        recognition_state.face_recognition_result = result
                        return
                    
                    # Prepare face recognition data
                    known_face_encodings = []
                    known_face_ids = []
                    known_face_names = []
                    
                    for user in known_faces_data.get("users", []):
                        if user.get("face_encoding"):
                            known_face_encodings.append(np.array(user["face_encoding"]))
                            known_face_ids.append(user.get("id", ""))
                            known_face_names.append(user.get("name", "Unknown"))
                    
                    if not known_face_encodings:
                        logger.warning("No valid face encodings found in registered users")
                        result["error"] = "No valid face encodings found in registered users"
                        recognition_state.face_recognition_result = result
                        return
                    
                    # Compare faces
                    matches = face_recog.compare_faces(known_face_encodings, face_encodings[0], tolerance=0.6)
                    face_distances = face_recog.face_distance(known_face_encodings, face_encodings[0])
                    
                    if not any(matches):
                        logger.info("Face not recognized - no matches found")
                        result["message"] = "Face not recognized"
                        result["success"] = True
                        recognition_state.face_recognition_result = result
                        return
                    
                    # Find the best match
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        matched_user_id = known_face_ids[best_match_index]
                        matched_user_name = known_face_names[best_match_index]
                        match_confidence = 1.0 - float(face_distances[best_match_index])
                        
                        logger.info(f"Face recognized as {matched_user_name} with {match_confidence:.2%} confidence")
                        
                        result["face_recognized"] = True
                        result["matched_users"] = [{
                            "id": matched_user_id,
                            "name": matched_user_name,
                            "confidence": match_confidence
                        }]
                        result["success"] = True
                
                except Exception as e:
                    logger.error(f"Error during face matching: {str(e)}")
                    result["error"] = f"Error during face matching: {str(e)}"
                    recognition_state.face_recognition_result = result
                    return
                
            except Exception as e:
                logger.error(f"Error finding best frame: {str(e)}")
                result["error"] = f"Error during face processing: {str(e)}"
                recognition_state.face_recognition_result = result
                return
            
            # Set the final result
            total_time = time.time() - start_time
            logger.info(f"Face recognition completed in {total_time:.2f} seconds")
            result["success"] = True
            recognition_state.face_recognition_result = result
            
        except Exception as e:
            logger.error(f"Unexpected error in face recognition: {str(e)}")
            recognition_state.face_recognition_result = {
                "success": False,
                "error": f"Unexpected error in face recognition: {str(e)}",
                "timestamp": datetime.now().isoformat()
            }
        finally:
            # Always set recognition state to inactive when done
            recognition_state.face_recognition_active = False

    # Define a simple liveness check function in routes.py
    def check_liveness(frame, face_location):
        """Simple placeholder liveness detection to check if a face is real
        
        In a production system, you would implement a more robust liveness detection
        using blink detection, texture analysis, etc.
        """
        try:
            # Get face dimensions
            top, right, bottom, left = face_location
            face_img = frame[top:bottom, left:right]
            
            # Ensure the face region is valid
            if face_img.size == 0:
                logger.warning("Invalid face region for liveness check")
                return False
            
            # Basic checks - in a real system, you'd have much more sophisticated checks
            
            # Check 1: Face has some variation in color/texture (non-flat)
            gray = cv2.cvtColor(face_img, cv2.COLOR_RGB2GRAY) if len(face_img.shape) == 3 else face_img
            std_dev = np.std(gray)
            texture_check = std_dev > 25  # Arbitrary threshold
            
            # Check 2: Face doesn't have unnaturally uniform regions
            # Simplified check for demonstration
            _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            white_pixels = np.count_nonzero(thresh)
            white_ratio = white_pixels / (gray.shape[0] * gray.shape[1])
            uniform_check = white_ratio < 0.15  # Real faces shouldn't have large uniform white regions
            
            is_live = texture_check and uniform_check
            
            logger.info(f"Liveness check: texture_std={std_dev:.2f}, white_ratio={white_ratio:.3f}, is_live={is_live}")
            return is_live
            
        except Exception as e:
            logger.error(f"Error in liveness check: {str(e)}")
            return False

    return app 