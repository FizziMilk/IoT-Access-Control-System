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
        
        # Return the template
        return render_template("face_recognition.html")
    
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
        """Start the face recognition process"""
        nonlocal camera, camera_needs_cleanup, face_recognition_active, face_recognition_result
        
        skip_liveness = request.form.get('skip_liveness', 'false').lower() == 'true'
        
        if face_recognition_active:
            return jsonify({'status': 'error', 'message': 'Face recognition already in progress'})
        
        # Set up threading event to signal completion
        recognition_complete = threading.Event()
        
        # Define function to run in the background
        def run_recognition_background():
            nonlocal face_recognition_active, face_recognition_result, camera, camera_needs_cleanup
            
            # Mark as active
            face_recognition_active = True
            
            try:
                # Get camera - we need to force camera creation here in the thread
                # that will use it to avoid threading issues with OpenCV
                local_camera = None
                try:
                    with camera_lock:
                        if camera is not None and hasattr(camera, 'isOpened') and camera.isOpened():
                            # Use existing camera
                            local_camera = camera
                            logger.info("Using existing camera for recognition")
                        else:
                            # Create new camera
                            logger.info("Creating new camera for recognition")
                            local_camera = get_camera()
                            camera = local_camera
                            camera_needs_cleanup = True
                except Exception as e:
                    logger.error(f"Error getting camera: {e}")
                    face_recognition_result = {"success": False, "error": f"Camera error: {str(e)}"}
                    face_recognition_active = False
                    recognition_complete.set()
                    return
                
                # Set up debug directory for frame saving
                debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
                os.makedirs(debug_dir, exist_ok=True)
                
                # Import camera index from config
                try:
                    from camera_config import CAMERA_INDEX
                except ImportError:
                    CAMERA_INDEX = 0
                
                # Get backend URL
                backend_url = os.getenv("BACKEND_URL", "http://localhost:5000")
                
                # Capture multiple frames at the beginning
                frames = []
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                
                try:
                    # Capture 4 frames with small delays between them
                    for i in range(4):
                        ret, frame = local_camera.read()
                        if ret and frame is not None:
                            frames.append(frame)
                            # Save debug frame
                            cv2.imwrite(f"{debug_dir}/frame_capture_{i}_{timestamp}.jpg", frame)
                            logger.info(f"Captured frame {i+1}/4")
                        else:
                            logger.warning(f"Failed to capture frame {i+1}")
                        
                        # Small delay between captures
                        time.sleep(0.5)
                
                    # Release camera right after capturing frames
                    try:
                        logger.info("Releasing camera early to free resources")
                        local_camera.release()
                        logger.info("Camera released after frame capture")
                    except Exception as e:
                        logger.error(f"Error releasing camera after capture: {e}")
                
                except Exception as e:
                    logger.error(f"Error during multi-frame capture: {e}")
                    face_recognition_result = {"success": False, "error": f"Frame capture error: {str(e)}"}
                    face_recognition_active = False
                    recognition_complete.set()
                    return
                
                # Check if we have enough frames
                if not frames:
                    logger.error("No frames captured")
                    face_recognition_result = {"success": False, "error": "No frames captured"}
                    face_recognition_active = False
                    recognition_complete.set()
                    return
                
                # Import face recognition modules
                from face_recognition_process import WebRecognition, load_known_faces, save_debug_frame
                import face_recognition
                import numpy as np
                from camera_config import MIN_FACE_WIDTH, MIN_FACE_HEIGHT
                
                # Initialize recognition system and load known faces
                recognition = WebRecognition()
                encodings, names = load_known_faces(backend_url)
                if encodings and names:
                    recognition.load_encodings(encodings, names)
                else:
                    logger.warning("No face encodings loaded from backend")
                
                # Process each frame to find the best one for recognition
                best_frame_index = -1
                best_face_width = 0
                best_face_height = 0
                
                try:
                    for i, (frame, face_location) in enumerate(zip(frames, face_recognition.face_locations(frames, number_of_times_to_upsample=2))):
                        if face_location is not None:
                            _, right, bottom, left = face_location
                            face_width = right - left
                            face_height = bottom - top
                            
                            # Handle potential array types
                            if isinstance(face_width, np.ndarray):
                                face_width = float(face_width.item()) if face_width.size == 1 else float(face_width.mean())
                            if isinstance(face_height, np.ndarray):
                                face_height = float(face_height.item()) if face_height.size == 1 else float(face_height.mean())
                            
                            # Compare as scalars
                            if float(face_width) > float(best_face_width) or float(face_height) > float(best_face_height):
                                best_frame_index = i
                                best_face_width = face_width
                                best_face_height = face_height
                except Exception as e:
                    logger.error(f"Error finding best frame: {e}")
                    logger.error(f"Best frame selection traceback: {traceback.format_exc()}")
                
                logger.info(f"Best frame index: {best_frame_index}")
                logger.info(f"Best face dimensions: {best_face_width}x{best_face_height} pixels")
                
                # Check if the best face is too small
                face_width_too_small = False
                face_height_too_small = False
                
                if isinstance(best_face_width, np.ndarray):
                    face_width_too_small = (best_face_width < MIN_FACE_WIDTH).any()
                else:
                    face_width_too_small = best_face_width < MIN_FACE_WIDTH
                    
                if isinstance(best_face_height, np.ndarray):
                    face_height_too_small = (best_face_height < MIN_FACE_HEIGHT).any()
                else:
                    face_height_too_small = best_face_height < MIN_FACE_HEIGHT
                
                if best_frame_index == -1 or face_width_too_small or face_height_too_small:
                    logger.warning(f"No suitable face found or face too small: width={best_face_width}, height={best_face_height}")
                    
                    # Prepare result with feedback
                    result = {
                        "success": False,
                        "face_detected": best_frame_index != -1,
                        "face_recognized": False,
                        "face_too_small": face_width_too_small or face_height_too_small,
                        "error": "Face is too small for reliable recognition"
                    }
                    
                    # Determine how far away the face is
                    if best_frame_index != -1:
                        # Check if the width is an array and use .any() for comparison
                        if isinstance(best_face_width, np.ndarray):
                            width_very_small = (best_face_width < MIN_FACE_WIDTH * 0.5).any()
                        else:
                            width_very_small = best_face_width < MIN_FACE_WIDTH * 0.5
                            
                        # Check if the height is an array and use .any() for comparison
                        if isinstance(best_face_height, np.ndarray):
                            height_very_small = (best_face_height < MIN_FACE_HEIGHT * 0.5).any()
                        else:
                            height_very_small = best_face_height < MIN_FACE_HEIGHT * 0.5
                        
                        if width_very_small or height_very_small:
                            result["distance_feedback"] = "much_too_far"
                        else:
                            result["distance_feedback"] = "too_far"
                    
                    # Save a debug frame with a rectangle and text
                    if best_frame_index != -1 and len(frames) > best_frame_index:
                        debug_frame = frames[best_frame_index].copy()
                        face_loc = face_recognition.face_locations(frames, number_of_times_to_upsample=2)[best_frame_index]
                        top, right, bottom, left = face_loc
                        
                        # Draw rectangle around face
                        cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 255, 255), 2)
                        cv2.putText(debug_frame, "Face too small - move closer", (left, top - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Write the frame to file with timestamp
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        debug_filepath = os.path.join(debug_dir, f"too_small_{timestamp}.jpg")
                        cv2.imwrite(debug_filepath, debug_frame)
                    
                    recognition_running = False
                    return result
                
                # Check if we found any usable faces
                if best_frame_index == -1:
                    logger.warning("No faces detected in any frame")
                    face_recognition_result = {
                        "success": False,
                        "face_detected": False,
                        "error": "No face detected"
                    }
                    recognition_running = False
                    return face_recognition_result
                
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
                    
                    recognition_running = False
                    return result
                
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
                
                # Perform liveness check on all frames
                liveness_results = []
                frame_liveness_results = []
                
                # Helper function to sanitize values from arrays
                def sanitize_value(value):
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            return value.item()
                        else:
                            return float(value.mean())
                    return value
                
                try:
                    if not skip_liveness:
                        logger.info("Performing liveness detection on all frames")
                        
                        # Process each frame with a detected face
                        for i, frame in enumerate(frames):
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
                                            
                                            # For debug frame visualization - create a new dict to ensure no arrays are passed
                                            liveness_results.append({
                                                "is_live": bool(is_live),
                                                "confidence": float(sanitize_value(liveness_result.get("confidence", 0))),
                                                "texture_score": float(sanitize_value(liveness_result.get("texture_score", 0)))
                                            })
                                        except Exception as e:
                                            logger.error(f"Error in liveness detection for frame {i+1}: {e}")
                                            # Log full traceback for debugging
                                            import traceback
                                            logger.error(f"Liveness detection traceback: {traceback.format_exc()}")
                            except Exception as e:
                                logger.error(f"Error processing frame {i+1}: {e}")
                                logger.error(f"Frame processing traceback: {traceback.format_exc()}")
                        
                        # Determine overall liveness using a voting system - using explicit scalar comparisons
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
                    logger.error(f"Liveness processing traceback: {traceback.format_exc()}")
                    # Provide a fallback liveness result
                    combined_liveness_result = {
                        "is_live": True,  # Default to True to avoid blocking recognition
                        "confidence": 0.5,
                        "error": str(e),
                        "skipped": True
                    }
                
                # Identify the face
                match_results = []
                for face_encoding in face_encodings:
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
                    
                matches = [{"match": match_for_visualization}] if faces else []
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
                    "face_encodings": [encoding.tolist() for encoding in face_encodings],
                    "face_locations": [best_face_location] if best_face_location else [],
                    "debug_frame": os.path.basename(frame_filename)
                }
                
                # Add match information if found
                if best_match and best_match.get("match"):
                    result["match"] = best_match["match"]
                
                # Add liveness results
                result["is_live"] = combined_liveness_result.get("is_live", False)
                result["liveness_results"] = combined_liveness_result
                    
                recognition_running = False
            except Exception as e:
                logger.error(f"Error during face recognition: {e}")
                face_recognition_result = {"success": False, "error": str(e)}
            
            # Mark as inactive
            face_recognition_active = False
            
            # Signal that we're done
            recognition_complete.set()
            
        # Start the background thread
        threading.Thread(target=run_recognition_background).start()
        
        # Wait briefly to ensure the thread has started
        time.sleep(0.1)
        
        return jsonify({'status': 'started'})
    
    @app.route('/check-face-recognition-status')
    def check_face_recognition_status():
        """Check the status of face recognition process"""
        nonlocal face_recognition_active, face_recognition_result
        
        # Check if recognition is active
        if face_recognition_active:
            status = "processing"
            result = None
        elif face_recognition_result is not None:
            # If face is detected but too small, set special status
            if face_recognition_result.get("face_detected") and face_recognition_result.get("face_too_small"):
                status = "face_too_small"
                # Record that face was detected but was too small for processing
                logger.info("Face detected but too small/far for reliable recognition")
            else:
                status = "complete"
            
            # Deep copy and make serializable to avoid modifying the original
            def make_serializable(obj):
                """Make obj JSON serializable"""
                if isinstance(obj, dict):
                    return {k: make_serializable(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [make_serializable(i) for i in obj]
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (int, float, str, bool, type(None))):
                    return obj
                else:
                    try:
                        # Try to convert to primitive type
                        return str(obj)
                    except Exception:
                        return repr(obj)
            
            try:
                result = make_serializable(face_recognition_result)
                
                # Add debug frame information if available
                debug_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'static', 'debug_frames')
                
                try:
                    # Try to find the final or most relevant frame
                    if os.path.exists(debug_dir):
                        # For face_too_small status, look for that specific frame first
                        if status == "face_too_small":
                            patterns = ['frame_face_too_small_', 'frame_faces_', 'frame_initial_']
                        else:
                            # Look for files in order of preference
                            patterns = ['frame_final_', 'frame_faces_', 'frame_liveness_fail_', 'frame_nofaces_', 'frame_initial_']
                        
                        for pattern in patterns:
                            files = [f for f in os.listdir(debug_dir) if f.startswith(pattern)]
                            if files:
                                # Sort by timestamp (which is part of the filename) to get the most recent
                                files.sort(reverse=True)
                                # Add the filename to the result
                                result['debug_frame'] = files[0]
                                logger.info(f"Found debug frame: {files[0]}")
                                break
                except Exception as e:
                    logger.error(f"Error finding debug frame: {e}")
            except Exception as e:
                logger.error(f"Error making result serializable: {e}")
                result = {"success": False, "error": "Error processing result"}
        else:
            status = "not_started"
            result = None
        
        return jsonify({"status": status, "result": result})
    
    @app.route('/process-face/<face_id>', methods=['POST'])
    def process_face(face_id):
        """Process a detected face and check access rights"""
        try:
            nonlocal face_recognition_result, camera_needs_cleanup
            
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
            
            if not face_recognition_result:
                logger.warning("No face recognition results available")
                flash('No face recognition results available', 'error')
                return redirect(url_for('face_recognition'))
            
            try:
                # Check if face was detected
                if not face_recognition_result.get('face_detected', False):
                    logger.warning("No face detected in result")
                    flash('No face was detected. Please try again.', 'error')
                    return redirect(url_for('face_recognition'))
                
                # Check if face encodings are available
                if 'face_encodings' not in face_recognition_result or not face_recognition_result['face_encodings']:
                    logger.warning("No face encodings found in result")
                    flash('No face encodings available', 'error')
                    return redirect(url_for('face_recognition'))
                
                # Store all the collected encodings in the session
                # We'll use these for registration if needed
                face_encodings = face_recognition_result['face_encodings']
                flask_session['face_encodings'] = face_encodings
                
                # Also store the primary encoding
                flask_session['face_encoding'] = face_encodings[0] if face_encodings else None
                
                # Check if the face was recognized
                if face_recognition_result.get('face_recognized', False) and face_recognition_result.get('match'):
                    match = face_recognition_result['match']
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
        face_encodings = flask_session.get('face_encodings', [])
        
        if not phone_number:
            flash("Phone number is required.", "danger")
            return redirect(url_for("face_recognition"))
            
        if not face_encoding and not face_encodings:
            flash("Face image required.", "danger")
            return redirect(url_for("face_recognition"))
            
        try:
            # If we have multiple encodings, use those
            if face_encodings:
                # Process all collected encodings
                for i, current_encoding in enumerate(face_encodings):
                    # Determine if this is an additional encoding (not the first one)
                    is_additional = (i > 0)
                    
                    try:
                        # Make sure encoding is serializable
                        if not isinstance(current_encoding, list):
                            if isinstance(current_encoding, np.ndarray):
                                current_encoding = current_encoding.tolist()
                            else:
                                logger.warning(f"Unexpected encoding type: {type(current_encoding).__name__}")
                                current_encoding = list(current_encoding) if hasattr(current_encoding, "__iter__") else [float(current_encoding)]
                                
                        # Convert to JSON and then base64
                        encoding_json = json.dumps(current_encoding)
                        encoding_base64 = base64.b64encode(encoding_json.encode('utf-8')).decode('utf-8')
                        
                        # Send to backend
                        resp = backend_session.post(f"{backend_url}/register-face", json={
                            "phone_number": phone_number,
                            "face_encoding": encoding_base64,
                            "is_additional": is_additional
                        })
                        
                        # Process response for the first encoding only
                        if i == 0:
                            if resp.status_code not in (200, 201):
                                logger.error(f"Registration failed with status: {resp.status_code}")
                                flash("Error registering with backend service.", "danger")
                                return redirect(url_for("face_recognition"))
                    except Exception as e:
                        logger.error(f"Error processing encoding {i}: {e}")
                        # Continue with other encodings if one fails
                
                # Clear session data
                if 'face_encodings' in flask_session:
                    flask_session.pop('face_encodings')
                
                # Only flash success and return after all encodings are processed
                flash("Face registered successfully with multiple angles. Please wait for OTP or admin approval.", "success")
                
                # Try to send OTP
                try:
                    resp = backend_session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
                    data = resp.json()
                    
                    if data.get("status") == "OTP sent":
                        return render_template("otp.html", phone_number=phone_number)
                    else:
                        return redirect(url_for("door_entry"))
                except Exception as e:
                    logger.error(f"Error sending OTP: {e}")
                    return redirect(url_for("door_entry"))
            else:
                # Legacy single encoding flow
                # Convert face encoding to base64 string
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
                try:
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
                                try:
                                    resp = backend_session.post(f"{backend_url}/door-entry", json={"phone_number": phone_number})
                                    data = resp.json()
                                    
                                    if data.get("status") == "OTP sent":
                                        return render_template("otp.html", phone_number=phone_number)
                                    else:
                                        return redirect(url_for("door_entry"))
                                except Exception as e:
                                    logger.error(f"Error sending OTP: {e}")
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
    
    @app.route('/reset-camera-resources', methods=['POST'])
    def reset_camera_resources():
        """
        Endpoint to force reset camera resources - useful when camera gets stuck
        """
        nonlocal camera, camera_needs_cleanup, face_recognition_active, face_recognition_result
        
        try:
            logger.info("Manually resetting camera resources")
            
            # Reset recognition state
            face_recognition_active = False
            face_recognition_result = None
            
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
                    camera_needs_cleanup = True
                
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

    return app 