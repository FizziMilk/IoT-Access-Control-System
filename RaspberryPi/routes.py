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
        nonlocal camera
        
        # If we already have a valid camera, return it
        if camera is not None and hasattr(camera, 'isOpened') and camera.isOpened():
            try:
                # Do a test read to make sure camera is still working
                ret, frame = camera.read()
                if ret and frame is not None:
                    return camera
                else:
                    logger.warning("Existing camera failed test read, will reinitialize")
                    release_camera()
            except Exception as e:
                logger.error(f"Error testing existing camera: {e}")
                release_camera()
            
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
        nonlocal camera
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
                    
                # Try to make sure system releases camera at OS level
                try:
                    # On Raspberry Pi, we can forcibly release the camera device
                    if platform.system() == 'Linux':
                        # Use fuser to kill processes using video device
                        subprocess.run('sudo fuser -k /dev/video0 2>/dev/null', shell=True)
                        subprocess.run('sudo fuser -k /dev/video1 2>/dev/null', shell=True)
                        subprocess.run('sudo fuser -k /dev/video2 2>/dev/null', shell=True)
                        time.sleep(1.0)  # Give OS time to fully release resources
                        logger.info("Attempted OS-level camera release")
                except Exception as e:
                    logger.error(f"Error during OS-level camera release: {e}")
    
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
    
    @app.route('/video_feed')
    def video_feed():
        """
        Video streaming route for face recognition.
        """
        def generate_frames():
            nonlocal face_recognition_active, face_recognition_result
            
            # Set up face recognition if needed
            recognition = None
            known_faces_loaded = False
            
            # Create placeholder when needed
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            font_color = (255, 255, 255)
            line_type = 2
            
            # Track camera failure count and last attempt time
            camera_failure_count = 0
            last_camera_attempt = time.time()
            camera_retry_interval = 5.0  # Seconds between camera open attempts
            
            # Initialize camera outside the loop
            cam = get_camera()
            
            try:
                while True:
                    frame = None
                    
                    # Check if we should try to reinitialize camera
                    current_time = time.time()
                    should_try_camera = (cam is None or (not hasattr(cam, 'isOpened') or not cam.isOpened())) and \
                                       (current_time - last_camera_attempt > camera_retry_interval)
                    
                    # Acquire lock only for the camera operations
                    with camera_lock:
                        if should_try_camera:
                            # Update last attempt time
                            last_camera_attempt = current_time
                            logger.info("Attempting to reconnect to camera")
                            
                            # Limit retries to avoid log spam
                            camera_failure_count += 1
                            if camera_failure_count > 5:
                                camera_retry_interval = 30.0  # Slow down retries after multiple failures
                            
                            # Try to reinitialize camera
                            cam = get_camera()
                            
                        # Read frame from camera or use placeholder
                        if hasattr(cam, 'read'):
                            success, frame = cam.read()
                            if not success:
                                # Create a placeholder frame
                                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                                cv2.putText(frame, "Failed to read frame", 
                                          (50, 240), font, font_scale, font_color, line_type)
                        else:
                            # Create a placeholder frame
                            frame = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame, "Camera not available", 
                                      (50, 240), font, font_scale, font_color, line_type)
                    
                    # Process frame outside the lock to minimize lock time
                    if frame is not None:
                        # Process face recognition if active
                        if face_recognition_active:
                            # Add status text
                            cv2.putText(frame, "Processing face recognition...", 
                                      (10, 30), font, font_scale, (0, 255, 0), line_type)
                            
                            # Initialize recognition system if needed
                            if recognition is None:
                                try:
                                    # Import directly here to avoid circular imports
                                    from face_recognition_process import WebRecognition as FaceRecognition
                                    recognition = FaceRecognition()
                                    logger.info("Face recognition initialized")
                                except ImportError:
                                    # Fallback to local implementation
                                    try:
                                        # Try to directly use the face_recognition module first
                                        class WebRecognition:
                                            """Simple face recognition class"""
                                            def __init__(self):
                                                self.known_face_encodings = []
                                                self.known_face_names = []
                                                self.detection_threshold = 0.6
                                                logger.info("WebRecognition initialized")
                                                
                                            def load_encodings(self, encodings, names):
                                                self.known_face_encodings = encodings
                                                self.known_face_names = names
                                                return True
                                                
                                            def identify_face(self, frame, face_location=None):
                                                if not self.known_face_encodings:
                                                    return None
                                                    
                                                # Convert to RGB for face_recognition
                                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                                
                                                # Get face locations if not provided
                                                if face_location is None:
                                                    import face_recognition
                                                    face_locations = face_recognition.face_locations(rgb_frame)
                                                    if not face_locations:
                                                        return None
                                                    face_location = face_locations[0]
                                                else:
                                                    face_locations = [face_location]
                                                    
                                                # Get face encodings
                                                import face_recognition
                                                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                                                
                                                if not face_encodings:
                                                    return None
                                                    
                                                face_encoding = face_encodings[0]
                                                
                                                # Compare with known faces
                                                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                                                
                                                if len(face_distances) > 0:
                                                    # Find best match
                                                    best_match_index = np.argmin(face_distances)
                                                    best_match_distance = face_distances[best_match_index]
                                                    
                                                    # Check if match is close enough
                                                    if best_match_distance <= self.detection_threshold:
                                                        match_name = self.known_face_names[best_match_index]
                                                        match_confidence = 1.0 - best_match_distance
                                                        
                                                        return {
                                                            "name": match_name,
                                                            "confidence": float(match_confidence),
                                                            "distance": float(best_match_distance)
                                                        }
                                                
                                                return None
                                        
                                        recognition = WebRecognition()
                                        logger.info("Face recognition initialized")
                                    except Exception as e:
                                        logger.error(f"Error initializing face recognition: {e}")
                                        face_recognition_active = False
                                        face_recognition_result = {
                                            'success': False,
                                            'error': f"Error initializing face recognition: {str(e)}"
                                        }
                            
                            # Load known faces if not already loaded
                            if not known_faces_loaded and backend_url and recognition:
                                try:
                                    from face_recognition_process import load_known_faces
                                    encodings, names = load_known_faces(backend_url)
                                    if encodings:
                                        recognition.load_encodings(encodings, names)
                                        known_faces_loaded = True
                                        logger.info(f"Loaded {len(encodings)} face encodings")
                                except Exception as e:
                                    logger.error(f"Error loading known faces: {e}")
                            
                            # Process face recognition
                            try:
                                # Convert to RGB for face_recognition
                                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                
                                # Find faces
                                import face_recognition
                                face_locations = face_recognition.face_locations(rgb_frame)
                                
                                # Process faces
                                if face_locations:
                                    # Draw rectangles around faces
                                    for top, right, bottom, left in face_locations:
                                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                                    
                                    # Process the first face
                                    face_location = face_locations[0]
                                    top, right, bottom, left = face_location
                                    
                                    # Extract face area
                                    face_image = frame[top:bottom, left:right]
                                    
                                    # Perform liveness check
                                    from face_recognition_process import perform_liveness_check
                                    liveness_result = perform_liveness_check(face_image)
                                    
                                    # Add liveness status to frame
                                    if liveness_result['is_live']:
                                        cv2.putText(frame, "LIVE", (left, top - 10), 
                                                  font, font_scale, (0, 255, 0), line_type)
                                    else:
                                        cv2.putText(frame, "FAKE", (left, top - 10), 
                                                  font, font_scale, (0, 0, 255), line_type)
                                    
                                    # Only continue if liveness check passed
                                    if liveness_result['is_live']:
                                        # Get face encodings
                                        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                                        
                                        # Identify face if we have encodings
                                        if face_encodings and recognition and recognition.known_face_encodings:
                                            match = recognition.identify_face(frame, face_location)
                                            
                                            if match:
                                                # Display match
                                                name = match['name']
                                                confidence = match['confidence']
                                                text = f"{name} ({confidence:.2f})"
                                                cv2.putText(frame, text, (left, bottom + 25), 
                                                          font, font_scale, (0, 255, 0), line_type)
                                                
                                                # Complete recognition
                                                face_recognition_result = {
                                                    'success': True,
                                                    'face_detected': True,
                                                    'is_live': True,
                                                    'match': match,
                                                    'face_encodings': [e.tolist() for e in face_encodings]
                                                }
                                                # Change flag after saving result
                                                face_recognition_active = False
                                            else:
                                                # No match found
                                                cv2.putText(frame, "Unknown", (left, bottom + 25), 
                                                          font, font_scale, (0, 0, 255), line_type)
                                                
                                                # Complete recognition with no match
                                                face_recognition_result = {
                                                    'success': True,
                                                    'face_detected': True,
                                                    'is_live': True,
                                                    'match': None,
                                                    'face_encodings': [e.tolist() for e in face_encodings]
                                                }
                                                # Change flag after saving result
                                                face_recognition_active = False
                                else:
                                    # No faces detected, keep searching
                                    cv2.putText(frame, "No face detected", (10, 60), 
                                              font, font_scale, (0, 0, 255), line_type)
                                    
                            except Exception as e:
                                logger.error(f"Error in face recognition: {e}")
                                logger.error(traceback.format_exc())
                                face_recognition_active = False
                                face_recognition_result = {
                                    'success': False,
                                    'error': f"Error processing face: {str(e)}"
                                }
                        
                        # Convert frame to JPEG for streaming
                        ret, buffer = cv2.imencode('.jpg', frame)
                        frame_bytes = buffer.tobytes()
                        
                        # Yield the frame in the MJPEG format
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        
                    # Add a small delay to control frame rate
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                logger.error(f"Error in video feed: {e}")
                logger.error(traceback.format_exc())
            finally:
                # Clean up - only release the camera at the end of the generator
                with camera_lock:
                    if cam is not None and hasattr(cam, 'release'):
                        cam.release()
                        logger.info("Camera released at end of video feed")
                # Clean up recognition object
                if recognition:
                    del recognition
        
        # Return the streaming response
        return Response(generate_frames(),
                        mimetype='multipart/x-mixed-replace; boundary=frame')
    
    @app.route('/start-face-recognition', methods=['POST'])
    def start_face_recognition():
        """Start face recognition process on the video feed"""
        nonlocal face_recognition_active, face_recognition_result
        
        # Reset result
        face_recognition_result = None
        
        # Skip liveness check if in testing mode
        skip_liveness = request.form.get('skip_liveness', 'false').lower() == 'true'
        if skip_liveness:
            logger.warning("Skipping liveness detection - TESTING MODE ONLY")
        
        # Start recognition process without touching the camera, which is handled by video_feed
        face_recognition_active = True
        
        return jsonify({'status': 'started'})
    
    @app.route('/check-face-recognition-status')
    def check_face_recognition_status():
        """Check if face recognition is complete and get results"""
        nonlocal face_recognition_active, face_recognition_result
        
        # No camera operations here, so no need to use camera_lock
        
        if face_recognition_active:
            return jsonify({'status': 'processing'})
        
        if face_recognition_result:
            result = face_recognition_result
            return jsonify({'status': 'complete', 'result': result})
        
        return jsonify({'status': 'not_started'})
    
    @app.route('/process-face', methods=['POST'])
    def process_face():
        """Process facial recognition using the integrated video feed"""
        nonlocal face_recognition_result
        
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
    
    # Clean up resources when app exits
    @app.teardown_appcontext
    def cleanup_resources(exception=None):
        """Clean up all resources when application context tears down"""
        logger.info("Cleaning up resources on app context teardown")
        
        # Release camera and destroy windows
        release_camera()
        
        # Explicitly destroy all OpenCV windows
        try:
            cv2.destroyAllWindows()
            # Force window cleanup with multiple waitKey calls
            for _ in range(5):
                cv2.waitKey(1)
        except Exception as e:
            logger.error(f"Error destroying windows: {e}")
            
        # Try one more time to ensure cameras are released at OS level
        try:
            if platform.system() == 'Linux':
                subprocess.run('sudo fuser -k /dev/video* 2>/dev/null', shell=True)
                logger.info("Final OS-level camera release completed")
        except Exception as e:
            logger.error(f"Error during final OS-level camera release: {e}")
            
        logger.info("Resource cleanup completed")

    return app 