from flask import render_template, request, redirect, url_for, flash, session as flask_session, jsonify, Response, current_app
from datetime import datetime, timedelta
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
from threading import Timer
from camera import WebCamera
import tempfile

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
    
    # Configure the backend URL
    app.config['BACKEND_API_URL'] = backend_url
    
    # Create and clean debug frames directory
    debug_dir = os.path.join(app.root_path, 'static', 'debug_frames')
    os.makedirs(debug_dir, exist_ok=True)
    
    # Clean up old debug frames (keep only the most recent 20)
    try:
        debug_files = [os.path.join(debug_dir, f) for f in os.listdir(debug_dir) if f.endswith('.jpg')]
        if len(debug_files) > 20:
            # Sort by modification time (newest first)
            debug_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            # Delete all but the 20 most recent files
            for file_path in debug_files[20:]:
                try:
                    os.remove(file_path)
                    logging.info(f"Deleted old debug frame: {file_path}")
                except Exception as e:
                    logging.warning(f"Failed to delete debug frame {file_path}: {e}")
    except Exception as e:
        logging.warning(f"Error cleaning debug frames directory: {e}")
        
    # Add integrated video feed with face recognition
    camera = None
    face_recognition_active = False
    face_recognition_result = None
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
                                  recent_recognition=recent_recognition,
                                  testing_mode=TESTING_MODE_ENABLED)

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
        nonlocal camera, camera_needs_cleanup, face_recognition_active, face_recognition_result
        
        logger.info("Face recognition page requested")
        
        # Reset recognition state
        face_recognition_active = False
        face_recognition_result = None
        
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
        
        # Check if testing mode - pass this to the template
        testing_mode = request.args.get('testing', 'false').lower() == 'true'
        
        # Return the template
        return render_template("face_recognition.html", testing_mode=testing_mode)
    
    @app.route('/video_feed')
    def video_feed():
        """Video feed for face recognition."""
        global camera
        camera_lock.acquire()
        try:
            # Check if camera is initialized and working
            if camera is None:
                try:
                    logging.info("Initializing camera for video feed")
                    camera = WebCamera(camera_index=CAMERA_INDEX)
                    
                    # Wait for camera to initialize
                    max_attempts = 10
                    for attempt in range(max_attempts):
                        frame = camera.get_frame()
                        if frame is not None:
                            logging.info(f"Camera initialized successfully after {attempt+1} attempts")
                            break
                        logging.info(f"Waiting for camera to provide a frame (attempt {attempt+1}/{max_attempts})")
                        time.sleep(0.5)
                        
                    if attempt == max_attempts - 1 and frame is None:
                        logging.error("Failed to get frame from camera after multiple attempts")
                        # Return a stream that shows "No camera available"
                        def generate_error():
                            while True:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + 
                                       generate_error_frame("No camera available") + b'\r\n')
                                time.sleep(1)
                        return Response(generate_error(), mimetype='multipart/x-mixed-replace; boundary=frame')
                except Exception as e:
                    logging.error(f"Error initializing camera: {str(e)}")
                    # Return a stream that shows the error
                    def generate_error():
                        while True:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   generate_error_frame(f"Camera error: {str(e)}") + b'\r\n')
                            time.sleep(1)
                    return Response(generate_error(), mimetype='multipart/x-mixed-replace; boundary=frame')
            
            # If camera is available but not providing frames
            frame = camera.get_frame()
            if frame is None:
                logging.warning("Camera is initialized but not providing frames")
                # Try to reset the camera
                try:
                    camera.release()
                    time.sleep(1)
                    camera = WebCamera(camera_index=CAMERA_INDEX)
                    # Check if reset helped
                    frame = camera.get_frame()
                    if frame is None:
                        logging.error("Camera reset didn't help - still no frames")
                        def generate_error():
                            while True:
                                yield (b'--frame\r\n'
                                       b'Content-Type: image/jpeg\r\n\r\n' + 
                                       generate_error_frame("Camera not providing frames") + b'\r\n')
                                time.sleep(1)
                        return Response(generate_error(), mimetype='multipart/x-mixed-replace; boundary=frame')
                except Exception as e:
                    logging.error(f"Error resetting camera: {str(e)}")
                    def generate_error():
                        while True:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + 
                                   generate_error_frame(f"Camera reset error: {str(e)}") + b'\r\n')
                            time.sleep(1)
                    return Response(generate_error(), mimetype='multipart/x-mixed-replace; boundary=frame')
        finally:
            camera_lock.release()
        
        def generate():
            while True:
                try:
                    frame = camera.get_frame()
                    if frame is None:
                        # Provide a placeholder frame
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + 
                               generate_waiting_frame("Waiting for camera...") + b'\r\n')
                        continue
                    
                    ret, jpeg = cv2.imencode('.jpg', frame)
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
                except Exception as e:
                    logging.error(f"Error in video feed: {str(e)}")
                    # Return an error frame
                    yield (b'--frame\r\n'
                           b'Content-Type: image/jpeg\r\n\r\n' + 
                           generate_error_frame(f"Feed error: {str(e)}") + b'\r\n')
                    time.sleep(0.5)

        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def generate_error_frame(message):
        """Generate an image with an error message."""
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a red background with transparency
        cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 100), -1)
        
        # Split message into lines if it's too long
        words = message.split()
        lines = []
        current_line = []
        for word in words:
            if len(' '.join(current_line + [word])) > 40:  # Adjust based on font size and width
                lines.append(' '.join(current_line))
                current_line = [word]
            else:
                current_line.append(word)
        if current_line:
            lines.append(' '.join(current_line))
        
        # Calculate vertical position for text
        y_offset = height // 2 - (len(lines) * 30) // 2
        
        # Add each line of text
        for i, line in enumerate(lines):
            y_pos = y_offset + i * 30
            cv2.putText(frame, line, (width//2 - 20*len(line)//2, y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Convert to JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    def generate_waiting_frame(message="Initializing camera, please wait..."):
        """Generate a waiting message frame."""
        height, width = 480, 640
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        # Add a blue background
        cv2.rectangle(frame, (0, 0), (width, height), (100, 100, 0), -1)
        
        # Add text
        cv2.putText(frame, message, (width//2 - 180, height//2 - 15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add a spinner animation (just a circle)
        cv2.ellipse(frame, (width//2, height//2 + 50), (30, 30), 0, 0, 360, (255, 255, 255), 2)
        
        # Convert to JPEG for streaming
        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()

    @app.route('/start-face-recognition', methods=['POST'])
    def start_face_recognition():
        """Start the face recognition process."""
        global face_recognition_thread, face_recognition_status, face_recognition_result
        global camera_warmup_frames, camera_initialized
        
        # Check if a recognition process is already running
        if face_recognition_thread and face_recognition_thread.is_alive():
            return jsonify({
                'status': 'error',
                'message': 'Face recognition is already in progress'
            })
        
        # Reset status and results
        face_recognition_status = "starting"
        face_recognition_result = None
        
        # Get camera configuration from request
        camera_index = CAMERA_INDEX
        
        # Get skip_liveness from form data
        skip_liveness = request.form.get('skip_liveness', 'false')
        skip_liveness = skip_liveness.lower() in ['true', '1', 'yes', 'on']
        
        # Create debug directory if it doesn't exist
        debug_dir = os.path.join(app.root_path, 'static', 'debug_frames')
        os.makedirs(debug_dir, exist_ok=True)
        
        # Ensure camera is initialized
        if not camera_initialized:
            try:
                app.logger.info("Initializing camera before face recognition")
                webcam = WebCamera(camera_index=camera_index)
                
                # Wait for camera to warm up
                max_attempts = 10
                attempts = 0
                while attempts < max_attempts:
                    attempts += 1
                    frame = webcam.get_frame()
                    if frame is not None:
                        app.logger.info(f"Camera initialized successfully after {attempts} attempts")
                        camera_initialized = True
                        webcam.release()
                        break
                    app.logger.warning(f"Camera warmup attempt {attempts}/{max_attempts} failed to return frame")
                    time.sleep(0.5)
                    
                if not camera_initialized:
                    app.logger.error("Failed to initialize camera after multiple attempts")
                    return jsonify({
                        'status': 'error',
                        'message': 'Failed to initialize camera. Please check your camera connection and try again.'
                    })
                    
            except Exception as e:
                app.logger.error(f"Error initializing camera: {str(e)}")
                return jsonify({
                    'status': 'error',
                    'message': f'Camera initialization error: {str(e)}'
                })
        
        # Start face recognition in a background thread
        def run_recognition():
            global face_recognition_status, face_recognition_result
            try:
                app.logger.info("Starting face recognition process")
                face_recognition_status = "processing"
                
                # Get the configured backend URL
                backend_url = current_app.config.get('BACKEND_API_URL')
                
                # Run the face recognition process
                result = run_face_recognition(
                    camera_index=camera_index,
                    backend_url=backend_url,
                    output_file=None,  # We'll handle the result directly
                    skip_liveness=skip_liveness,
                    debug_dir=debug_dir
                )
                
                # Ensure result is JSON serializable
                json_result = make_serializable(result)
                
                # Save the result for status checking
                face_recognition_result = json_result
                face_recognition_status = "complete"
                app.logger.info(f"Face recognition completed with result: {json_result.get('success', False)}")
                
            except Exception as e:
                app.logger.error(f"Error in face recognition thread: {str(e)}", exc_info=True)
                face_recognition_status = "error"
                face_recognition_result = {
                    "success": False,
                    "message": f"Error: {str(e)}"
                }
        
        face_recognition_thread = threading.Thread(target=run_recognition)
        face_recognition_thread.daemon = True
        face_recognition_thread.start()
        
        return jsonify({
            'status': 'started',
            'message': 'Face recognition started'
        })

    @app.route('/check-face-recognition-status')
    def check_face_recognition_status():
        """Check status of face recognition process."""
        global face_recognition_status, face_recognition_result
        
        # Check if recognition is in progress
        if face_recognition_status is None:
            return jsonify({
                "status": "not_started",
                "message": "Face recognition has not been started"
            })
        
        # Create a copy of the result to avoid modifying the original
        result = {}
        if face_recognition_result:
            result = copy.deepcopy(face_recognition_result)
        
        # Build response object
        response = {
            "status": face_recognition_status
        }
        
        # Add result data if available
        if result:
            # Extract debug frame path and convert to filename only
            if "debug_frame" in result and result["debug_frame"]:
                debug_file = result["debug_frame"]
                # Extract the filename from the full path
                debug_filename = os.path.basename(debug_file)
                result["debug_frame"] = debug_filename
            
            # Ensure all values are JSON serializable
            result = make_serializable(result)
            response["result"] = result
        
        app.logger.info(f"Face recognition status: {face_recognition_status}")
        return jsonify(response)
    
    @app.route('/process-face/<face_id>', methods=['POST'])
    def process_face(face_id):
        """Process a detected face."""
        global face_recognition_result
        
        app.logger.info(f"Processing face: {face_id}")
        
        if not face_recognition_result:
            app.logger.error("No face recognition result available")
            flash("Error: No face recognition result available", "danger")
            return redirect(url_for('face_recognition'))
        
        # Check if recognition was successful
        if not face_recognition_result.get('success', False):
            app.logger.warning(f"Face recognition failed: {face_recognition_result.get('message', 'Unknown error')}")
            flash(f"Face recognition failed: {face_recognition_result.get('message', 'Unknown error')}", "warning")
            return redirect(url_for('face_recognition'))
        
        # Get the best match
        best_match = face_recognition_result.get('best_match')
        if not best_match:
            app.logger.warning("No face match found")
            flash("Your face was not recognized. Please register to gain access.", "warning")
            return redirect(url_for('face_recognition'))
        
        # Parse face match if it's a string
        if isinstance(best_match, str):
            try:
                best_match = json.loads(best_match)
            except json.JSONDecodeError as e:
                app.logger.error(f"Error decoding face match JSON: {str(e)}")
                flash("Error processing face match data", "danger")
                return redirect(url_for('face_recognition'))
        
        # Get user details
        user_id = best_match.get('id')
        user_name = best_match.get('name', 'Unknown')
        
        app.logger.info(f"Face matched to user: {user_name} (ID: {user_id})")
        
        # Log access
        app.logger.info(f"Access granted to: {user_name}")
        flash(f"Welcome, {user_name}!", "success")
        
        # Redirect to user dashboard
        return redirect(url_for('user_dashboard', user_id=user_id))
    
    @app.route('/register-face', methods=['GET', 'POST'])
    def register_face():
        """Register a captured face with a phone number"""
        # For GET requests, show the registration form
        if request.method == 'GET':
            # Get the latest debug frame timestamp
            debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
            timestamp = ""
            
            try:
                # Get list of files in debug_frames directory
                if os.path.exists(debug_dir):
                    files = [f for f in os.listdir(debug_dir) if f.startswith('frame_final_')]
                    if files:
                        # Sort by timestamp (which is part of the filename)
                        files.sort(reverse=True)
                        # Extract the timestamp part from the filename
                        timestamp = files[0].replace('frame_final_', '').replace('.jpg', '')
                        logger.info(f"Using debug frame with timestamp: {timestamp}")
            except Exception as e:
                logger.error(f"Error getting debug frame timestamp: {e}")
                
            return render_template("register_face.html", timestamp=timestamp)
            
        # For POST requests, process the registration
        phone_number = request.form.get('phone_number')
        
        # Get the face encoding from Flask session
        face_encoding = flask_session.get('face_encoding')
        
        if not phone_number or not face_encoding:
            flash("Phone number and face image required.", "danger")
            return redirect(url_for("face_recognition"))
            
        try:
            # Convert face encoding to base64 string
            # Make sure face_encoding is serializable (should be a list at this point)
            if not isinstance(face_encoding, list):
                try:
                    # Try to convert if it's a numpy array
                    if isinstance(face_encoding, np.ndarray):
                        face_encoding = face_encoding.tolist()
                    else:
                        logger.warning(f"Unexpected face_encoding type: {type(face_encoding).__name__}")
                        face_encoding = list(face_encoding) if hasattr(face_encoding, "__iter__") else [float(face_encoding)]
                except Exception as e:
                    logger.error(f"Error converting face encoding: {e}", exc_info=True)
                    flash("Error processing face data.", "danger")
                    return redirect(url_for("face_recognition"))
            
            try:
                # Convert face encoding list to JSON string
                encoding_json = json.dumps(face_encoding)
                
                # Encode JSON string as base64
                encoding_base64 = base64.b64encode(encoding_json.encode('utf-8')).decode('utf-8')
            except (TypeError, json.JSONDecodeError) as e:
                logger.error(f"JSON serialization error: {e}", exc_info=True)
                flash("Error processing face data: invalid format.", "danger")
                return redirect(url_for("face_recognition"))
            
            # Send the face encoding and phone number to the backend
            resp = backend_session.post(f"{backend_url}/register-face", json={
                "phone_number": phone_number,
                "face_encoding": encoding_base64
            })
            
            if resp.status_code == 200 or resp.status_code == 201:
                try:
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
                except Exception as e:
                    logger.error(f"Error processing registration response: {e}")
                    flash("Error processing backend response.", "danger")
            else:
                logger.error(f"Registration failed with status: {resp.status_code}")
                flash("Error registering with backend service.", "danger")
        except Exception as e:
            logger.error(f"Error connecting to backend: {e}")
            flash("Error connecting to backend.", "danger")
            
        return redirect(url_for("face_recognition"))
    
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
    
    @app.route('/reset-camera-resources')
    def reset_camera_resources():
        """Manually reset camera resources."""
        global camera
        
        try:
            logging.info("Manually resetting camera resources")
            
            # Release the current camera
            cleanup_camera_resources()
            
            # Try to initialize a new camera
            camera = WebCamera(camera_index=0, width=640, height=480, face_detection=True)
            
            # Check if initialization was successful
            if camera and camera.is_camera_running():
                logging.info("Camera successfully reset")
                return jsonify({"status": "success", "message": "Camera resources reset successfully"})
            else:
                logging.error("Failed to initialize camera after reset")
                return jsonify({"status": "error", "message": "Failed to initialize camera after reset"})
        
        except Exception as e:
            logging.error(f"Error resetting camera resources: {e}")
            return jsonify({"status": "error", "message": f"Error resetting camera: {str(e)}"})
    
    @app.route('/capture-preview-frame', methods=['POST'])
    def capture_preview_frame():
        """Capture a single frame to display during processing"""
        nonlocal camera
        
        try:
            logger.info("Capturing preview frame before facial recognition")
            
            # Ensure debug frames directory exists
            debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
            os.makedirs(debug_dir, exist_ok=True)
            
            # Generate timestamp for the frame
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            
            # Get camera instance
            curr_camera = get_camera()
            if curr_camera is None or not hasattr(curr_camera, 'read'):
                logger.error("Camera not available for preview capture")
                return jsonify({"success": False, "error": "Camera not available"}), 500
                
            # Read frame
            ret, frame = curr_camera.read()
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

    return app 

def make_serializable(obj):
    """Make an object JSON serializable by converting numpy arrays and other special types."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, (np.ndarray, np.float32, np.float64)):
        if obj.ndim == 0:  # scalar
            return float(obj)
        else:  # array
            return [make_serializable(x) for x in obj]
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        # Try to convert to string or primitive type
        try:
            return str(obj)
        except:
            return None 