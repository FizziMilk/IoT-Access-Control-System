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
import socket
import pickle
import gc

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
            if 'recent_recognition' in flask_session:
                # Keep the recognition data for display in the door_unlocked template
                return render_template("door_unlocked.html")
            elif 'debug_frame' in flask_session:
                # Create basic recognition data if only debug frame is available
                flask_session['recent_recognition'] = {
                    'user': phone_number,
                    'debug_frame_id': flask_session['debug_frame'],
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
    
    @app.route('/door_entry', methods=['GET', 'POST'])
    def door_entry():
        # First check if the door should be unlocked according to schedule
        should_unlock = check_schedule(door_controller, mqtt_handler, backend_session, backend_url)
        
        if should_unlock:
            # If scheduled unlock, open the door directly
            door_controller.unlock_door()
            flash('Door unlocked per schedule', 'success')
            return redirect(url_for('index'))
        
        # If not scheduled, proceed with facial recognition
        return redirect(url_for('face_recognition_page'))

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
        reset_recognition_state()
        
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
                    time.sleep(0.01)  # ~100 FPS (reduced from 0.02 for faster response)
                    
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
        
        # Define how many frames to capture - reduced for faster processing
        num_frames = 5  # Reduced from 10 to 5 frames for faster processing
        
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
                time.sleep(0.3)  # Reduced from 0.5 seconds to 0.3 seconds for faster capture
        
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
        """Check the status of the face recognition process."""
        response = {
            "active": recognition_state.face_recognition_active,
            "status": "not_started"
        }
        
        if not recognition_state.face_recognition_active and recognition_state.face_recognition_result:
            # Create a serializable copy of the result
            serializable_result = {}
            for key, value in recognition_state.face_recognition_result.items():
                # Convert numpy arrays to lists
                if isinstance(value, np.ndarray):
                    serializable_result[key] = value.tolist()
                # Ensure all values are JSON serializable
                elif isinstance(value, (bool, int, float, str, list, dict)) or value is None:
                    serializable_result[key] = value
                else:
                    # Convert other types to string
                    serializable_result[key] = str(value)
            
            response["result"] = serializable_result
            
            # Add explicit status based on result contents
            if serializable_result.get("backend_error"):
                response["status"] = "backend_error"
            elif serializable_result.get("face_too_small"):
                response["status"] = "face_too_small"
            elif (serializable_result.get("face_detected") and 
                  "is_live" in serializable_result and 
                  serializable_result["is_live"] is False):
                response["status"] = "liveness_failed"
            elif serializable_result.get("registration_needed"):
                response["registration_needed"] = True
                response["status"] = "registration_needed"
                
                # Store the face encoding in the recognition state instead of session
                if serializable_result.get("save_face_encoding") and ("face_encoding" in serializable_result or "face_encodings" in serializable_result):
                    # Add detailed logging about the face encoding
                    encoding_data = serializable_result.get("face_encoding") or serializable_result.get("face_encodings")
                    if encoding_data is not None:
                        try:
                            # Generate a unique ID for this face data
                            face_id = str(uuid.uuid4())[:20]
                            logger.info(f"Storing face encoding in server memory for registration (type: {type(encoding_data)}, length: {len(encoding_data) if isinstance(encoding_data, list) else 'unknown'})")
                            
                            # Store in the recognition state
                            recognition_state.store_face_encoding(encoding_data, face_id)
                            
                            # Also set a simple flag in the session that face data is available
                            flask_session['has_valid_face_data'] = True
                            flask_session['face_id'] = face_id
                            
                            logger.info(f"Successfully stored face encoding with ID: {face_id}")
                        except Exception as storage_error:
                            logger.error(f"Error storing face encoding: {storage_error}", exc_info=True)
                    else:
                        logger.error("Cannot store face encoding: encoding data is None")
                else:
                    logger.warning("No face encoding available to store or save_face_encoding flag not set")
                
                # Important - mark recognition as complete to stop the face recognition bg process
                recognition_state.face_recognition_active = False
            
            elif serializable_result.get("recognized"):
                # Check if the recognized user is approved
                if serializable_result.get("is_allowed", False):
                    response["status"] = "recognized"
                    response["is_allowed"] = True
                    
                    # Ensure face_encodings is specifically preserved in the result
                    if "face_encoding" in serializable_result and not "face_encodings" in recognition_state.face_recognition_result:
                        # Store face_encoding as face_encodings to prevent the error
                        logger.info("Ensuring face_encodings are available for process_face")
                        recognition_state.face_recognition_result['face_encodings'] = recognition_state.face_recognition_result.get('face_encoding')
                else:
                    # User is recognized but not approved
                    response["status"] = "pending_approval"
                    response["is_allowed"] = False
                    response["user_name"] = serializable_result.get("user_name", "Unknown User")
                    logger.info(f"User recognized but pending approval: {response['user_name']}")
            elif serializable_result.get("success"):
                response["status"] = "complete"
            
            # Always perform camera cleanup when recognition is complete
            # whether it's a success, failure, or needs registration
            if not response["active"]:
                logger.info(f"Recognition completed with status: {response.get('status', 'unknown')}")
                try:
                    # Clean up camera resources
                    release_camera()
                    
                    # Wait a bit longer after release for better stability
                    time.sleep(1.0)
                    
                    # Add an additional step to ensure camera is fully released
                    try:
                        # Try to reset the video subsystem by opening and closing a temporary camera
                        logger.info("Performing additional cleanup to ensure camera is released")
                        
                        # Force reinitialize camera if needed
                        if platform.system() == 'Linux':
                            logger.info("Forcing additional Linux camera cleanup")
                            # Add a longer delay after release
                            time.sleep(1.0)
                        elif platform.system() == 'Windows':
                            # Force Windows cleanup
                            logger.info("Forcing additional Windows cleanup")
                            cv2.destroyAllWindows()
                            cv2.waitKey(1)
                            # Force garbage collection
                            gc.collect()
                        
                    except Exception as e:
                        logger.error(f"Error during additional cleanup: {e}")
                    
                    logger.info("Camera cleanup completed after recognition")
                except Exception as e:
                    logger.error(f"Error during camera cleanup: {str(e)}")
        elif recognition_state.face_recognition_active:
            # Currently processing, get the progress if available
            if hasattr(recognition_state, 'face_recognition_progress') and recognition_state.face_recognition_progress is not None:
                response["progress"] = recognition_state.face_recognition_progress
            response["status"] = "processing"
                
        return jsonify(response)
    
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
                
                # Verify liveness check passed before proceeding
                if not recognition_state.face_recognition_result.get('is_live', False):
                    logger.warning("Liveness check failed - rejecting registration attempt")
                    flash('Security check failed: Only live faces can be registered.', 'error')
                    return redirect(url_for('face_recognition_page'))
                
                # Check if face encodings are available
                if 'face_encodings' not in recognition_state.face_recognition_result or not recognition_state.face_recognition_result['face_encodings']:
                    logger.warning("No face encodings found in result")
                    flash('No face encodings available', 'error')
                    return redirect(url_for('face_recognition_page'))
                
                # Instead of storing face encodings in the session, store in server memory
                face_encodings = recognition_state.face_recognition_result['face_encodings']
                logger.info("Face encodings available for processing")
                
                # Store only a flag indicating we have valid face data
                flask_session['has_valid_face_data'] = True
                
                # Store the face encodings in server memory
                recognition_state.store_face_encoding(face_encodings, face_id)
                logger.info(f"Stored face encoding in server memory with ID: {face_id}")
                
                # Check if the face was recognized
                if recognition_state.face_recognition_result.get('face_recognized', False) and recognition_state.face_recognition_result.get('matched_users'):
                    matched_users = recognition_state.face_recognition_result.get('matched_users', [])
                    if matched_users:
                        matched_user = matched_users[0]
                        matched_phone = matched_user.get('phone_number', matched_user.get('user_id', matched_user.get('name')))
                        user_name = matched_user.get('name', matched_phone)  # Get name or fall back to phone
                        confidence = matched_user.get('confidence', 0)
                    
                    logger.info(f"Face matched with {matched_phone} (confidence: {confidence:.2f})")
                    flash(f"Face recognized! Welcome, {user_name}.", 'success')
                    
                    # Log the successful face recognition
                    try:
                        backend_session.post(f"{API_URL}/log-door-access", json={
                            "user": matched_phone,
                            "method": "Face Recognition",
                            "status": "Successful",
                            "details": f"Face recognized with confidence {confidence:.2f}"
                        })
                    except Exception as e:
                        logger.error(f"Error logging face access: {e}")
                    
                    # Get low_security flag from recognition result
                    low_security = recognition_state.face_recognition_result.get('low_security', False)
                    
                    # Store only essential recognition data for display
                    flask_session['recent_recognition'] = {
                        'user': user_name,  # Store name for display
                        'phone': matched_phone,  # Store phone for verification
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        'low_security': low_security
                    }
                    
                    # Only store the debug frame reference, not the whole frame
                    if 'debug_frame' in flask_session:
                        flask_session['recent_recognition']['debug_frame_id'] = flask_session.get('debug_frame')
                    
                    # Check if the user is in low security mode (can bypass OTP)
                    if low_security:
                        logger.info(f"User {matched_phone} has low security mode enabled - bypassing OTP verification")
                        # Log the direct door access
                        try:
                            backend_session.post(f"{API_URL}/log-door-access", json={
                                "user": matched_phone,
                                "method": "Face Recognition (Low Security)",
                                "status": "Door Unlocked",
                                "details": "Direct access via face recognition - OTP bypassed"
                            })
                        except Exception as e:
                            logger.error(f"Error logging direct door access: {e}")
                        
                        # Directly unlock the door
                        door_controller.unlock_door()
                        
                        # Flash a message indicating direct access
                        flash(f"Welcome, {user_name}! The door has been unlocked.", "success")
                        
                        # Redirect to door_unlocked page
                        return render_template("door_unlocked.html")
                    else:
                        # Normal security mode - send OTP and redirect to OTP verification
                        logger.info(f"Sending OTP to user {matched_phone} after face recognition")
                        try:
                            # Send request to door-entry endpoint to trigger OTP
                            resp = backend_session.post(f"{API_URL}/door-entry", json={"phone_number": matched_phone})
                            logger.info(f"Backend OTP response: {resp.status_code} - {resp.text}")
                            data = resp.json()

                            if data.get("status") == "OTP sent":
                                flash("OTP sent to your phone. Please enter the code to complete verification.", "success")
                                # Redirect to OTP verification page
                                return render_template("otp.html", phone_number=matched_phone)
                            elif data.get("status") == "pending":
                                # User exists but is not approved yet
                                logger.warning(f"User recognized but not approved: {matched_phone}")
                                flash(data.get("message", "Your account is pending approval. Please contact administrator."), "warning")
                                return redirect(url_for('door_entry'))
                            else:
                                # For any other error, create the OTP page anyway with the user's phone number
                                # This allows us to try OTP verification directly
                                logger.warning(f"OTP status unclear, defaulting to OTP page: {data}")
                                flash("Please enter the verification code sent to your phone.", "info")
                                return render_template("otp.html", phone_number=matched_phone)
                        except Exception as e:
                            logger.error(f"Error sending OTP after face recognition: {e}")
                            # Even if an error occurs, direct to OTP page instead of door_entry
                            flash("Please enter the verification code sent to your phone.", "info")
                            return render_template("otp.html", phone_number=matched_phone)
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
        if request.method == 'POST':
            # Log all form fields for debugging
            logger.info(f"Form submitted with fields: {list(request.form.keys())}")
            
            name = request.form.get('name')
            phone = request.form.get('phone_number')
            
            logger.info(f"Extracted name: '{name}', phone: '{phone}'")
            
            # Validate input
            if not name or not phone:
                logger.warning(f"Missing required fields - name: {bool(name)}, phone: {bool(phone)}")
                flash('Please provide both name and phone number', 'error')
                return redirect(url_for('register_face'))
            
            # Get face encoding from server memory instead of session
            face_encoding = recognition_state.get_face_encoding()
            
            # Detailed logs to diagnose issues
            if face_encoding is None:
                logger.error("Face encoding not found for registration form submission")
                flash('No face data found. Please try the face recognition process again.', 'error')
                return redirect(url_for('face_recognition_page'))
            
            logger.info(f"Retrieved face encoding for registration: Type={type(face_encoding)}")
            
            # Check if face_encoding is valid
            if isinstance(face_encoding, np.ndarray):
                logger.info(f"Converting numpy array to list. Shape: {face_encoding.shape}")
                face_encoding = face_encoding.tolist()
            elif not isinstance(face_encoding, list):
                logger.error(f"Unexpected face encoding type: {type(face_encoding)}")
                flash('Invalid face data format. Please try again.', 'error')
                return redirect(url_for('face_recognition_page'))
                
            # Verify list size if possible
            if isinstance(face_encoding, list):
                logger.info(f"Face encoding list length: {len(face_encoding)}")
                if len(face_encoding) != 128:
                    logger.warning(f"Unexpected face encoding length: {len(face_encoding)} (expected 128)")
            
            # Convert face encoding to base64 encoded JSON string
            try:
                # Convert face encoding to JSON string
                logger.info("Converting face encoding to JSON string")
                face_encoding_json = json.dumps(face_encoding)
                
                # Encode to base64
                logger.info("Encoding JSON string to base64")
                face_encoding_base64 = base64.b64encode(face_encoding_json.encode('utf-8')).decode('utf-8')
                
                logger.info("Successfully encoded face data in base64 format")
            except Exception as e:
                logger.error(f"Error encoding face data: {e}", exc_info=True)
                flash('Error processing face data. Please try again.', 'error')
                return redirect(url_for('register_face'))
            
            # Prepare data for API
            user_data = {
                'name': name,
                'phone_number': phone,  # Updated to match backend API field name
                'face_encoding': face_encoding_base64
            }
            
            try:
                # Send registration request to backend API
                logger.info(f"Sending registration request to backend for user: {name}")
                logger.info(f"API URL: {API_URL}/register-face")
                logger.debug(f"Request data: {json.dumps({**user_data, 'face_encoding': '(BASE64_DATA)'})}")
                
                # Log request details without sensitive data
                logger.info(f"Request payload size: {len(json.dumps(user_data))} bytes")
                
                # Make the API request with detailed error handling
                try:
                    response = requests.post(f"{API_URL}/register-face", json=user_data, timeout=10)
                    logger.info(f"API response status code: {response.status_code}")
                    
                    if response.status_code == 200 or response.status_code == 201:
                        logger.info(f"Successfully registered user: {name}")
                        # Clear the session
                        if 'has_valid_face_data' in flask_session:
                            flask_session.pop('has_valid_face_data', None)
                        if 'face_id' in flask_session:
                            flask_session.pop('face_id', None)
                            
                        # Clear the face encoding from server memory
                        recognition_state.clear_face_encoding()
                        
                        flash('Registration successful!', 'success')
                        reset_recognition_state()
                        return redirect(url_for('index'))
                    else:
                        # Try to get detailed error information
                        try:
                            error_data = response.json()
                            error_msg = f"Registration failed. Status code: {response.status_code}"
                            if 'error' in error_data:
                                error_msg += f". Error: {error_data['error']}"
                            logger.error(error_msg)
                            # Also log the first 100 chars of the response for debugging
                            logger.error(f"Response content (truncated): {response.text[:100]}...")
                            flash(error_msg, 'error')
                        except Exception as json_error:
                            # Handle cases where the response isn't JSON
                            logger.error(f"Failed to parse error response: {json_error}. Response text: {response.text[:200]}")
                            flash(f"Registration failed. Status code: {response.status_code}", 'error')
                except requests.RequestException as req_error:
                    # Handle connection errors specifically
                    logger.error(f"Request error during registration: {req_error}", exc_info=True)
                    flash(f"Connection error: {str(req_error)}", 'error')
            except Exception as e:
                # Handle any other errors
                error_msg = f"Error during registration: {str(e)}"
                logger.error(error_msg, exc_info=True)
                flash(error_msg, 'error')
            
            return redirect(url_for('register_face'))
        
        # GET request - render the registration form
        # Check if we have valid face data available
        encoding_available = False
        liveness_failed = False
        
        # Check if there was a liveness check failure in the recognition result
        if hasattr(recognition_state, 'face_recognition_result') and recognition_state.face_recognition_result:
            result = recognition_state.face_recognition_result
            if result.get("face_detected") and not result.get("is_live", True):
                liveness_failed = True
                logger.warning("Liveness check failed - showing liveness error in registration page")
        
        # Check if face encoding is available in server memory
        if recognition_state.get_face_encoding() is not None:
            encoding_available = True
            logger.info("Face encoding available in server memory")
        # Also check for the flag in the session as a backup
        elif flask_session.get('has_valid_face_data', False):
            encoding_available = True
            logger.info("Face encoding availability confirmed by session flag")
            
        return render_template('register_face.html', 
                              encoding_available=encoding_available,
                              liveness_failed=liveness_failed)
    
    @app.route('/update-name', methods=['POST'])
    def update_name():
        phone_number = request.form.get('phone_number')
        name = request.form.get('name')
        if not phone_number or not name:
            flash("Name and phone number are required.", "danger")
            return redirect(url_for("door_entry"))
        try:
            resp = backend_session.post(f"{API_URL}/update-user-name", json={"phone_number": phone_number, "name": name})
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
            max_attempts = 2  # Reduced from 3 to 2 for faster reset
            success = False
            
            while attempts < max_attempts and not success:
                attempts += 1
                logger.info(f"Performing camera open/close cycle (attempt {attempts}/{max_attempts})")
                
                # Try different camera indices for more thorough reset
                for idx in range(2):  # Try indices 0 and 1
                    try:
                        temp_camera = cv2.VideoCapture(idx)
                        if temp_camera.isOpened():
                            # Read a frame to ensure it's working
                            ret, _ = temp_camera.read()
                            # Release immediately
                            temp_camera.release()
                            logger.info(f"Camera open/close cycle completed successfully with index {idx}")
                            success = True
                    except Exception as e:
                        logger.error(f"Error in camera cycle with index {idx}: {e}")
                
                # Small delay between attempts
                if not success and attempts < max_attempts:
                    time.sleep(0.5)  # Reduced from 1.0
            
            # Now try to initialize a new camera for the video feed
            logger.info("Attempting to open camera at index 0")
            camera = get_camera()
            if camera is not None and camera.isOpened():
                logger.info("Successfully opened camera at index 0")
                camera_needs_cleanup = True
            else:
                logger.error("Failed to open camera after reset")
            
            return jsonify({"status": "success", "message": "Camera resources reset"}), 200
        except Exception as e:
            logger.error(f"Error resetting camera resources: {e}")
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
    
    @app.route('/log-pending-access', methods=['POST'])
    def log_pending_access():
        """Log access attempts by users who are recognized but awaiting approval"""
        try:
            data = request.json
            user_id = data.get('user_id')
            user_name = data.get('user_name', 'Unknown User')
            
            logger.info(f"Pending access attempt logged for user: {user_name} (ID: {user_id})")
            
            # Log the pending access attempt to the backend
            try:
                backend_session.post(f"{API_URL}/log-door-access", json={
                    "user": user_name,
                    "method": "Face Recognition",
                    "status": "Pending Approval",
                    "details": f"User recognized but pending admin approval"
                }, timeout=5)
                logger.info(f"Successfully logged pending access attempt to backend")
            except Exception as e:
                logger.error(f"Error logging pending access to backend: {e}")
            
            return jsonify({"status": "success"}), 200
        except Exception as e:
            logger.error(f"Error logging pending access: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500
    
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
        """Run face recognition in the background."""
        try:
            logger.info("Starting face recognition in background thread")
            max_retry_attempts = 3  # Maximum number of retries for backend connection
            
            # Initialize result structure
            result = {
                "success": False,
                "face_detected": False,
                "recognized": False,
                "liveness_check": False,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "backend_error": False  # New field to track backend connection errors
            }
            
            # Set initial progress
            recognition_state.face_recognition_progress = 10
            
            # Try to get camera (with retry mechanism built in)
            start_time = time.time()
            camera = get_camera()
            camera_setup_time = time.time() - start_time
            logger.info(f"Camera setup completed in {camera_setup_time:.2f} seconds")
            
            if camera is None or not camera.isOpened():
                logger.error("Failed to open camera")
                result["error_message"] = "Could not open camera"
                recognition_state.face_recognition_result = result
                recognition_state.face_recognition_active = False
                return result

            # Capture frames with timeout to prevent infinite loops
            start_time = time.time()
            frames = []
            face_locations = []
            max_frames = 30  # Maximum number of frames to process
            frame_timeout = 6  # Reduced from 10 to 6 seconds
            
            # Update progress - capturing frames
            recognition_state.face_recognition_progress = 20
            
            # Set a fixed number of frames to capture
            target_frames = 4  # Reduced from 8 to 4 frames for faster processing
            
            while len(frames) < target_frames:
                if time.time() - start_time > frame_timeout:
                    logger.warning(f"Frame capture timeout after {time.time() - start_time:.2f} seconds")
                    break
                    
                ret, frame = camera.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame")
                    continue
                    
                # Add frame to collection
                frames.append(frame)
                
                # Get face locations for this frame
                try:
                    locations = face_recog.face_locations(frame)
                    face_locations.append(locations if locations else [])
                except Exception as e:
                    logger.error(f"Error during face detection: {e}")
                    face_locations.append([])
                
                # Enforce minimum processing time between frames to prevent overload
                time.sleep(0.1)
            
            capture_time = time.time() - start_time
            logger.info(f"Captured {len(frames)} frames in {capture_time:.2f} seconds")
            
            # Release camera
            camera.release()
            
            # Update progress - looking for faces
            recognition_state.face_recognition_progress = 30
            
            if not frames:
                logger.error("No frames captured")
                result["error_message"] = "Failed to capture any frames"
                recognition_state.face_recognition_result = result
                recognition_state.face_recognition_active = False
                return result
            
            # Find the best frame with the largest face
            best_frame_index = None
            best_face_index = 0  # Default to first face
            max_face_area = 0
            
            for i, (frame, locations) in enumerate(zip(frames, face_locations)):
                if not locations:
                    continue
                
                # Check all faces in this frame
                for face_idx, face_loc in enumerate(locations):
                    try:
                        # Calculate face dimensions
                        top, right, bottom, left = face_loc
                        face_width = right - left
                        face_height = bottom - top
                        face_area = face_width * face_height
                        
                        if face_area > max_face_area:
                            max_face_area = face_area
                            best_frame_index = i
                            best_face_index = face_idx
                    except Exception as e:
                        logger.error(f"Error calculating face area: {e}")
                        continue
            
            # No faces found in any frame
            if best_frame_index is None:
                logger.info("No faces detected in any frame")
                result["error_message"] = "No face detected"
                recognition_state.face_recognition_result = result
                recognition_state.face_recognition_active = False
                return result
            
            # Update progress - face found, processing encoding
            recognition_state.face_recognition_progress = 40
            
            # Get the best frame and the corresponding face location
            best_frame = frames[best_frame_index]
            best_face_location = face_locations[best_frame_index][best_face_index]
            
            # Check if face is big enough (minimum 100x100 pixels)
            top, right, bottom, left = best_face_location
            face_width = right - left
            face_height = bottom - top
            
            # Handle different types for face dimensions (numpy array or int)
            is_too_small = False
            try:
                if isinstance(face_width, np.ndarray):
                    is_too_small = (face_width < 100).any()
                elif isinstance(face_height, np.ndarray):
                    is_too_small = is_too_small or (face_height < 100).any()
                else:
                    is_too_small = face_width < 100 or face_height < 100
                    
                logger.debug(f"Face dimensions: width={face_width}, height={face_height}, is_too_small={is_too_small}")
            except Exception as e:
                logger.error(f"Error during face size check: {e}")
                is_too_small = True  # Assume too small if we can't check properly
            
            if is_too_small:
                logger.warning(f"Face too small for reliable recognition: {face_width}x{face_height}")
                result["face_detected"] = True
                result["error_message"] = "Face too small for reliable recognition"
                result["face_too_small"] = True
                
                # Save debug frame with face rectangle and text for small face
                try:
                    debug_frame = best_frame.copy()
                    cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(debug_frame, f"TOO SMALL: {face_width}x{face_height}", 
                                (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    # Save to debug file with timestamp
                    debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
                    os.makedirs(debug_dir, exist_ok=True)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = os.path.join(debug_dir, f"small_face_{timestamp}.jpg")
                    cv2.imwrite(filename, debug_frame)
                    logger.info(f"Saved debug frame for small face to {filename}")
                except Exception as debug_error:
                    logger.error(f"Error saving debug frame: {debug_error}")
                
                recognition_state.face_recognition_result = result
                recognition_state.face_recognition_active = False
                return result
            
            # Face is big enough, proceed with recognition
            result["face_detected"] = True
            
            # Update progress - processing face encoding
            recognition_state.face_recognition_progress = 50
            
            # Get the face encoding
            start_time = time.time()
            try:
                face_encoding = face_recog.face_encodings(best_frame, [best_face_location])[0]
                encoding_time = time.time() - start_time
                logger.info(f"Face encoding completed in {encoding_time:.2f} seconds")
                
                # Store both face_encoding and face_encodings for consistency
                result["face_encoding"] = face_encoding.tolist()
                result["face_encodings"] = face_encoding.tolist()
            except Exception as e:
                logger.error(f"Error during face encoding: {e}")
                result["error_message"] = "Error encoding face"
                recognition_state.face_recognition_result = result
                recognition_state.face_recognition_active = False
                return result
            
            # Update progress - checking liveness
            recognition_state.face_recognition_progress = 60
            
            # Check for liveness using enhanced detection
            start_time = time.time()
            try:
                is_live = False
                
                # Create liveness detector with anti-spoofing features
                from face_recognition_process import LivenessDetector
                liveness_detector = LivenessDetector()
                
                # We need frames with faces to do liveness analysis
                if len(face_locations) > 0 and frames:
                    logger.info(f"Performing enhanced liveness detection with gradient analysis and LBP")
                    
                    # Get the best frame (middle frame usually has the best quality)
                    best_frame_idx = len(frames) // 2
                    best_frame = frames[best_frame_idx]
                    best_face_loc = face_locations[best_frame_idx][0] if len(face_locations[best_frame_idx]) > 0 else None
                    
                    if best_face_loc:
                        # Run enhanced liveness detection on the face
                        liveness_result = liveness_detector.check_face_liveness(best_frame, best_face_loc)
                        
                        # Get results
                        is_live = liveness_result.get("is_live", False)
                        confidence_score = liveness_result.get("confidence_score", 0.0)
                        
                        # Add all liveness metrics to the result
                        for key, value in liveness_result.items():
                            result[f"liveness_{key}"] = value
                        
                        logger.info(f"Enhanced liveness check completed with confidence {confidence_score:.2f}, is_live={is_live}")
                    else:
                        logger.warning("No suitable face location found for liveness check")
                else:
                    logger.warning("No faces detected for liveness analysis")
                
                result["liveness_check"] = is_live
                result["is_live"] = is_live
                
                liveness_time = time.time() - start_time
                logger.info(f"Liveness check completed in {liveness_time:.2f} seconds")
            except Exception as e:
                logger.error(f"Error during liveness check: {e}")
                result["liveness_check"] = False
                result["is_live"] = False
            
            # Update progress - retrieving known user encodings
            recognition_state.face_recognition_progress = 70
            
            # Attempt to get known encodings from backend
            known_encodings = []
            known_names = []
            known_user_ids = []
            known_approval_status = []
            known_user_data = []  # Store full user data objects
            
            retry_count = 0
            max_retries = 2  # Reduced from 3 to 2 for faster failure response
            
            while retry_count < max_retries:
                try:
                    logger.info(f"Attempt {retry_count+1} to get known encodings from backend")
                    response = requests.get(f"{API_URL}/get-user-encodings", timeout=10)
                    
                    if response.status_code == 200:
                        response_data = response.json()
                        logger.info(f"Successfully retrieved {len(response_data.get('users', []))} known face encodings")
                        
                        # Update progress - processing response
                        recognition_state.face_recognition_progress = 80
                        
                        # Extract encodings, names, and user IDs from the 'users' array
                        users = response_data.get('users', [])
                        known_encodings = [user['face_encoding'] for user in users]
                        known_names = [user['name'] for user in users]
                        known_user_ids = [user['id'] for user in users]
                        known_approval_status = [user.get('is_allowed', False) for user in users]
                        known_user_data = users  # Store the full user objects
                        
                        logger.info(f"Processed {len(known_encodings)} face encodings")
                        break  # Successfully got the encodings, exit the loop
                    elif response.status_code == 404:
                        # No registered users with face encodings in the system
                        logger.info("No users with registered faces found in the system")
                        known_encodings = []
                        known_names = []
                        known_user_ids = []
                        known_approval_status = []
                        
                        # Set registration needed flag when no users exist
                        result["registration_needed"] = True
                        result["save_face_encoding"] = True
                        result["is_live"] = True  # Set to true to allow registration
                        logger.info("Setting registration_needed flag as no users exist in the system")
                        break  # Exit the loop, no need to retry
                    else:
                        logger.error(f"Failed to get known encodings: {response.status_code}, {response.text}")
                        retry_count += 1
                        if retry_count < max_retries:
                            sleep_time = retry_count * 2  # progressive backoff
                            logger.info(f"Retrying in {sleep_time} seconds...")
                            time.sleep(sleep_time)
                except Exception as e:
                    logger.error(f"Error getting known encodings: {str(e)}")
                    retry_count += 1
                    if retry_count < max_retries:
                        sleep_time = retry_count * 2
                        logger.info(f"Retrying in {sleep_time} seconds...")
                        time.sleep(sleep_time)
            
            # Update progress - comparing faces
            recognition_state.face_recognition_progress = 90
            
            if retry_count >= max_retries and not known_encodings:
                logger.error("Failed to get known encodings after maximum retries")
                result = {
                    "success": False,
                    "error": "Failed to get known encodings from backend",
                    "face_detected": result["face_detected"],
                    "face_too_small": is_too_small,
                    "liveness_check": {
                        "success": result["liveness_check"],
                        "error": result["error_message"]
                    }
                }
                recognition_state.face_recognition_result = result
                recognition_state.face_recognition_active = False
                return result
            
            # Compare face with known encodings
            face_recognized = False
            if known_encodings:
                try:
                    # Convert encodings to numpy arrays if they aren't already
                    np_known_encodings = []
                    for encoding in known_encodings:
                        try:
                            if isinstance(encoding, list):
                                np_known_encodings.append(np.array(encoding))
                            elif isinstance(encoding, np.ndarray):
                                np_known_encodings.append(encoding)
                            else:
                                logger.warning(f"Unexpected encoding type: {type(encoding)}")
                                continue
                        except Exception as conversion_error:
                            logger.error(f"Error converting encoding: {conversion_error}")
                            continue
                    
                    # Check if we have any valid encodings after conversion
                    if not np_known_encodings:
                        logger.error("No valid known encodings after conversion")
                        raise ValueError("No valid known encodings after conversion")
                    
                    # Log encoding formats for debugging
                    logger.info(f"Known encoding type: {type(np_known_encodings[0]) if np_known_encodings else 'None'}")
                    logger.info(f"Face encoding type: {type(face_encoding)}")
                    
                    # Ensure face_encoding is also a numpy array
                    if isinstance(face_encoding, list):
                        face_encoding = np.array(face_encoding)
                    
                    # Compare faces with tolerance
                    matches = face_recog.compare_faces(np_known_encodings, face_encoding, tolerance=0.6)
                    
                    if True in matches:
                        # Find the match with minimum distance
                        face_distances = face_recog.face_distance(np_known_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        
                        if matches[best_match_index]:
                            recognized_name = known_names[best_match_index]
                            user_id = known_user_ids[best_match_index]
                            is_approved = known_approval_status[best_match_index]
                            # Get low_security status from response
                            low_security = False
                            if isinstance(known_user_data, list) and len(known_user_data) > best_match_index:
                                low_security = known_user_data[best_match_index].get('low_security', False)
                            
                            result["recognized"] = True
                            result["user_name"] = recognized_name
                            result["user_id"] = user_id
                            result["is_allowed"] = is_approved
                            result["low_security"] = low_security
                            result["confidence"] = float(1 - face_distances[best_match_index])
                            
                            # Store the matched user data in the result
                            matched_user = {
                                'name': recognized_name,
                                'confidence': float(1 - face_distances[best_match_index]),
                                'user_id': user_id,
                                'phone_number': known_user_data[best_match_index].get('phone_number', user_id),
                                'is_allowed': is_approved,
                                'low_security': low_security
                            }
                            result['matched_users'] = [matched_user]
                            result['face_recognized'] = True
                            
                            logger.info(f"Face recognized as {recognized_name} (ID: {user_id}) with confidence {result['confidence']:.2f}, approved: {is_approved}, low_security: {low_security}")
                            face_recognized = True
                except Exception as e:
                    logger.error(f"Error during face comparison: {str(e)}", exc_info=True)
                    result["error_message"] = f"Face comparison error: {str(e)}"
                    result["recognized"] = False
                    
                    # Save debug frame with the detected face and error info
                    try:
                        debug_frame = best_frame.copy()
                        top, right, bottom, left = best_face_location
                        cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 0, 255), 2)
                        cv2.putText(debug_frame, f"COMPARISON ERROR", 
                                   (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                        
                        # Save to debug file with timestamp
                        debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
                        os.makedirs(debug_dir, exist_ok=True)
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = os.path.join(debug_dir, f"face_comparison_error_{timestamp}.jpg")
                        cv2.imwrite(filename, debug_frame)
                        logger.info(f"Saved debug frame for face comparison error to {filename}")
                    except Exception as debug_error:
                        logger.error(f"Error saving debug frame: {debug_error}")
            
            # Set registration needed flag if face wasn't recognized but users exist
            if not face_recognized and known_encodings:
                result["registration_needed"] = True
                result["save_face_encoding"] = True
                result["is_live"] = True  # Set to true to allow registration
                logger.info("Setting registration_needed flag as face was not recognized")
            
            # Mark process as successful
            result["success"] = True
            
            # Update progress to 100%
            recognition_state.face_recognition_progress = 100
            
            # Store the final result
            recognition_state.face_recognition_result = result
            
            # Important - explicitly set face recognition as complete
            recognition_state.face_recognition_active = False
            logger.info(f"Face recognition completed: recognition_needed={result.get('registration_needed', False)}, recognized={result.get('recognized', False)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Unexpected error in face recognition: {e}")
            traceback.print_exc()
            result = {
                "success": False,
                "error_message": str(e),
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            # Make sure to set the result and mark as inactive even if there's an error
            recognition_state.face_recognition_result = result
            recognition_state.face_recognition_active = False
            return result

    def reset_recognition_state():
        """Reset all recognition state variables"""
        recognition_state.reset()
        logger.info("Recognition state has been reset")

    return app 