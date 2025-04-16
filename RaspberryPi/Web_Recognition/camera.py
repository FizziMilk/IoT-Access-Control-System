"""
Web-optimized camera handling for face recognition.
Designed to be stateless and resource-efficient for web applications.
"""
import cv2
import time
import numpy as np
import face_recognition
import os
import logging
import traceback
from contextlib import contextmanager

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("WebCamera")

class WebCamera:
    """Camera handling optimized for web applications."""
    
    def __init__(self, headless=False):
        """
        Initialize without claiming resources.
        
        Args:
            headless: If True, no windows will be displayed (for server environments)
        """
        # Defer initialization until needed
        self.camera = None
        self.last_frame = None
        # Configuration
        self.camera_id = 0
        self.resolution = (640, 480)
        self.liveness_resolution = (320, 240)
        # Constants for blink detection
        self.EAR_THRESHOLD = 0.20
        # Operating mode
        self.headless = headless
        
    @contextmanager
    def camera_session(self, resolution=None):
        """
        Context manager for camera usage to ensure proper cleanup.
        
        Args:
            resolution: Optional tuple of (width, height)
            
        Yields:
            The camera object
        """
        try:
            # Setup Qt environment first
            self._setup_qt_environment()
            
            # Initialize camera
            camera = self._initialize_camera(resolution)
            if camera is None:
                raise RuntimeError("Failed to initialize camera")
            
            # Store reference
            self.camera = camera
            
            # Yield camera for use
            yield camera
            
        finally:
            # Always clean up regardless of exceptions
            self._release_camera()
    
    def capture_face(self, resolution=None, max_attempts=3):
        """
        Capture a frame with a face.
        
        Args:
            resolution: Optional tuple of (width, height)
            max_attempts: Maximum number of capture attempts
            
        Returns:
            Tuple of (frame, face_location) or (None, None) if no face is found
        """
        with self.camera_session(resolution or self.resolution):
            # Give the camera some time to adjust
            time.sleep(0.5)
            
            for attempt in range(max_attempts):
                logger.info(f"Capture attempt {attempt+1}/{max_attempts}")
                
                # Capture several frames to allow camera to adjust
                frames = []
                for _ in range(5):
                    ret, frame = self.camera.read()
                    if ret:
                        frames.append(frame)
                    time.sleep(0.1)
                
                if not frames:
                    logger.warning("No frames captured")
                    continue
                
                # Use the last frame (most adjusted)
                frame = frames[-1]
                
                # Find faces in the frame
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame)
                
                if face_locations:
                    # Scale face location back to original size
                    top, right, bottom, left = face_locations[0]
                    face_location = (top*2, right*2, bottom*2, left*2)
                    logger.info(f"Face found at {face_location}")
                    
                    # Return both frame and face location
                    return frame, face_location
            
            logger.warning("No face found after maximum attempts")
            return None, None
    
    def detect_blinks(self, timeout=10):
        """
        Detect eye blinks using the eye aspect ratio (EAR) method.
        
        Args:
            timeout: Maximum seconds to wait for blinks
            
        Returns:
            bool: True if blinks were detected, False otherwise
        """
        logger.info(f"Starting blink detection with timeout={timeout}s")
        
        with self.camera_session(self.liveness_resolution):
            start_time = time.time()
            blink_counter = 0
            ear_values = []
            
            # Continue until timeout or sufficient blinks detected
            while (time.time() - start_time) < timeout and blink_counter < 2:
                ret, frame = self.camera.read()
                if not ret:
                    logger.warning("Failed to capture frame")
                    continue
                
                # Convert frame for face_recognition
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Detect faces
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if not face_locations:
                    continue
                    
                # Get face landmarks
                landmarks = face_recognition.face_landmarks(rgb_small_frame, face_locations)
                if not landmarks:
                    continue
                
                # Scale landmarks back to original size
                scaled_landmarks = {}
                for feature, points in landmarks[0].items():
                    scaled_landmarks[feature] = [(p[0]*2, p[1]*2) for p in points]
                
                # Get eye landmarks
                left_eye = scaled_landmarks.get('left_eye')
                right_eye = scaled_landmarks.get('right_eye')
                
                if not left_eye or not right_eye:
                    continue
                
                # Calculate EAR
                left_ear = self._eye_aspect_ratio(left_eye)
                right_ear = self._eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0
                ear_values.append(ear)
                
                # Check for blink - use a moving average approach for robustness
                if len(ear_values) > 3:
                    # Current EAR compared to recent average
                    recent_avg = sum(ear_values[-4:-1]) / 3
                    
                    if ear < self.EAR_THRESHOLD and recent_avg > (self.EAR_THRESHOLD + 0.05):
                        # This is likely a blink (sudden drop in EAR)
                        blink_counter += 1
                        logger.info(f"Blink detected! EAR: {ear:.3f}, Threshold: {self.EAR_THRESHOLD:.3f}")
                
                # Only show windows in non-headless mode
                if not self.headless:
                    # Display blink count and EAR on frame
                    cv2.putText(frame, f"Blinks: {blink_counter}", (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(frame, f"EAR: {ear:.3f}", (10, frame.shape[0] - 20),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Draw eye contours for visual feedback
                    self._draw_eyes(frame, left_eye, right_eye)
                    
                    # Show frame with feedback
                    cv2.imshow("Blink Detection", frame)
                    key = cv2.waitKey(1) & 0xFF
                    
                    # Allow manual exit
                    if key == 27:  # ESC key
                        break
            
            # Clean up windows
            if not self.headless:
                cv2.destroyAllWindows()
            
            # Result
            result = blink_counter >= 2
            logger.info(f"Blink detection result: {result} ({blink_counter} blinks)")
            return result

    def capture_face_with_liveness(self, timeout=30):
        """
        Capture a face with liveness detection (blink).
        
        Args:
            timeout: Maximum seconds to wait
            
        Returns:
            numpy.ndarray: Face image if liveness check passes, None otherwise
        """
        logger.info(f"Starting face capture with liveness detection (timeout={timeout}s)")
        
        # First detect blinks to verify liveness
        if not self.detect_blinks(timeout=timeout):
            logger.warning("Liveness check failed - no blinks detected")
            return None
        
        # If liveness check passes, capture a face image
        logger.info("Liveness check passed, capturing face image")
        frame, face_location = self.capture_face()
        
        if frame is not None and face_location is not None:
            # Return the captured frame for further processing
            logger.info("Face capture successful")
            return frame
        else:
            logger.warning("Face capture failed after liveness check")
            return None
    
    def _eye_aspect_ratio(self, eye):
        """
        Calculate the eye aspect ratio (EAR) for blink detection.
        
        Args:
            eye: List of 6 (x, y) coordinates of eye landmarks
            
        Returns:
            float: Eye aspect ratio
        """
        # Compute the euclidean distances between the vertical eye landmarks
        A = np.linalg.norm(np.array(eye[1]) - np.array(eye[5]))
        B = np.linalg.norm(np.array(eye[2]) - np.array(eye[4]))
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = np.linalg.norm(np.array(eye[0]) - np.array(eye[3]))
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def _draw_eyes(self, frame, left_eye, right_eye):
        """Draw eye contours on the frame for visual feedback."""
        # Convert eye points to numpy arrays for drawing
        left_eye_np = np.array(left_eye, dtype=np.int32)
        right_eye_np = np.array(right_eye, dtype=np.int32)
        
        # Draw eye contours
        cv2.polylines(frame, [left_eye_np], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye_np], True, (0, 255, 0), 1)
        
    def _initialize_camera(self, resolution=None):
        """
        Initialize camera with robust error handling.
        
        Args:
            resolution: Optional resolution tuple (width, height)
            
        Returns:
            cv2.VideoCapture: Camera object or None if failed
        """
        logger.info(f"Initializing camera with ID={self.camera_id}")
        
        # First release any existing camera
        self._release_camera()
        
        # Try to open camera
        try:
            logger.info(f"Opening camera with index {self.camera_id}")
            cap = cv2.VideoCapture(self.camera_id)
            
            if cap is None or not cap.isOpened():
                logger.warning(f"Failed to open camera with index {self.camera_id}")
                return None
                
            logger.info(f"Successfully opened camera with index {self.camera_id}")
            
            # Set resolution if provided
            if resolution is not None:
                logger.info(f"Setting camera resolution to {resolution}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            # Wait for camera to initialize
            time.sleep(0.5)
            
            return cap
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _release_camera(self):
        """
        Release camera resources safely.
        """
        if self.camera is not None:
            try:
                logger.info("Releasing camera")
                self.camera.release()
                cv2.destroyAllWindows()
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
            finally:
                self.camera = None
                time.sleep(0.5)  # Wait to ensure resources are released
                logger.info("Camera resources released")
    
    def _setup_qt_environment(self):
        """Set up appropriate Qt environment variables based on available plugins."""
        try:
            # Use xcb on Linux (from the error logs)
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            
            # Also set QT_DEBUG_PLUGINS=1 to help diagnose plugin issues
            os.environ["QT_DEBUG_PLUGINS"] = "1"
            
            logger.info("Qt environment configured")
        except Exception as e:
            logger.error(f"Error setting up Qt environment: {e}") 