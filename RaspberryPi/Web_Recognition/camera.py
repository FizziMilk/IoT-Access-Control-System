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
from skimage import feature as skimage_feature

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
        Capture a face with liveness detection (blink and texture analysis).
        
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
        
        # If blink check passes, capture a face image for texture analysis
        logger.info("Blink check passed, capturing face for texture analysis")
        frame, face_location = self.capture_face()
        
        if frame is None or face_location is None:
            logger.warning("Failed to capture face after blink check")
            return None
            
        # Perform texture analysis as a second liveness check
        logger.info("Performing texture analysis")
        texture_result = self.analyze_facial_texture(frame)
        
        if not texture_result:
            logger.warning("Texture analysis failed - possible spoof detected")
            return None
            
        logger.info("Liveness verification complete - both blink and texture checks passed")
        
        # Return the captured frame for further processing
        return frame
    
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
            # Wait a moment to ensure previous camera sessions are fully closed
            time.sleep(1.0)
            
            # Force destroy any lingering windows
            cv2.destroyAllWindows()
            
            # Clear any pre-existing OpenCV state 
            for i in range(5):
                cap = cv2.VideoCapture(self.camera_id)
                if cap is not None and cap.isOpened():
                    cap.release()
                    time.sleep(0.2)
            
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
                self.camera = None
                
                # Ensure all OpenCV windows are destroyed
                cv2.destroyAllWindows()
                
                # Wait to ensure resources are released
                time.sleep(0.5)
                logger.info("Camera resources released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
                logger.error(traceback.format_exc())
                # Still set camera to None even if release fails
                self.camera = None
    
    def _setup_qt_environment(self):
        """Set up appropriate Qt environment variables based on available plugins."""
        try:
            # Use xcb on Linux (from the error logs)
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            
            # Also set QT_DEBUG_PLUGINS=1 to help diagnose plugin issues
            os.environ["QT_DEBUG_PLUGINS"] = "1"
            
            # Add to the _setup_qt_environment method in camera.py
            os.environ["QT_PLUGIN_PATH"] = "/usr/lib/aarch64-linux-gnu/qt5/plugins"  # For Raspberry Pi
            
            # Force single-threaded operation for Qt objects
            os.environ["QT_THREAD_PRIORITY_POLICY"] = "1"
            
            # Reset any lingering OpenCV state
            cv2.destroyAllWindows()
            
            # Sleep to ensure any previous resources are fully released
            time.sleep(0.5)
            
            logger.info("Qt environment configured")
        except Exception as e:
            logger.error(f"Error setting up Qt environment: {e}")

    def analyze_facial_texture(self, frame):
        """
        Analyze facial texture and entropy to detect printed faces and screens.
        Uses multi-scale Local Binary Patterns (LBP) to analyze texture patterns.
        
        Args:
            frame: Image containing a face
            
        Returns:
            bool: True if texture appears to be from a real face, False otherwise
        """
        try:
            logger.info("Starting facial texture analysis")
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to enhance texture
            equalized = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            
            # Calculate Local Binary Pattern at multiple scales
            # This helps capture both micro and macro texture patterns
            
            # Small scale (radius=1) - captures micro-texture
            radius1 = 1
            n_points1 = 8 * radius1
            lbp1 = skimage_feature.local_binary_pattern(blurred, n_points1, radius1, method="uniform")
            
            # Medium scale (radius=2) - captures more structure
            radius2 = 2
            n_points2 = 8 * radius2
            lbp2 = skimage_feature.local_binary_pattern(blurred, n_points2, radius2, method="uniform")
            
            # Larger scale (radius=3) - captures even more structure
            radius3 = 3
            n_points3 = 8 * radius3
            lbp3 = skimage_feature.local_binary_pattern(blurred, n_points3, radius3, method="uniform")
            
            # Compute histograms for each scale
            hist1, _ = np.histogram(lbp1.ravel(), bins=np.arange(0, n_points1 + 3), range=(0, n_points1 + 2))
            hist1 = hist1.astype("float")
            hist1 /= (hist1.sum() + 1e-7)
            
            hist2, _ = np.histogram(lbp2.ravel(), bins=np.arange(0, n_points2 + 3), range=(0, n_points2 + 2))
            hist2 = hist2.astype("float")
            hist2 /= (hist2.sum() + 1e-7)
            
            hist3, _ = np.histogram(lbp3.ravel(), bins=np.arange(0, n_points3 + 3), range=(0, n_points3 + 2))
            hist3 = hist3.astype("float")
            hist3 /= (hist3.sum() + 1e-7)
            
            # Calculate texture entropy for each scale (measure of randomness/complexity)
            entropy1 = -np.sum(hist1 * np.log2(hist1 + 1e-7))
            entropy2 = -np.sum(hist2 * np.log2(hist2 + 1e-7))
            entropy3 = -np.sum(hist3 * np.log2(hist3 + 1e-7))
            
            # Analyze pattern uniformity - printed photos often have more uniform patterns
            uniformity1 = np.sum(hist1 * hist1)
            uniformity2 = np.sum(hist2 * hist2)
            uniformity3 = np.sum(hist3 * hist3)
            
            # Calculate multi-scale uniformity ratio
            uniformity_ratio = (uniformity1 + uniformity2) / (2 * uniformity3 + 1e-7)
            
            # Analyze reflectance properties using gradient information
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            
            # Calculate micro-texture features that differentiate real skin
            gradient_mean = np.mean(magnitude)
            gradient_std = np.std(magnitude)
            gradient_entropy = self._calculate_entropy(magnitude)
            
            # For real skin, gradient distribution is distinctive
            gradient_ratio = gradient_std / (gradient_mean + 1e-7)
            
            # Log metrics for debugging
            logger.info(f"Texture entropy (multi-scale): {entropy1:.2f}/{entropy2:.2f}/{entropy3:.2f}")
            logger.info(f"Uniformity ratio: {uniformity_ratio:.4f}, Gradient ratio: {gradient_ratio:.4f}")
            
            # Simplified decision logic for web mode (same as original implementation)
            # 1. Basic entropy check
            entropy_score = (entropy3 > 3.5)  # Real faces have higher entropy
            
            # 2. Simple gradient check
            gradient_score = (gradient_ratio > 1.0)  # Real faces have more complex gradient patterns
            
            # Pass if either test passes (more lenient for web usage)
            is_real_texture = entropy_score or gradient_score
            
            logger.info(f"Texture analysis results: Entropy={entropy_score}, Gradient={gradient_score}")
            logger.info(f"Texture test overall result: {'PASS' if is_real_texture else 'FAIL'}")
            
            # Create visualization if not in headless mode
            if not self.headless:
                try:
                    self._create_texture_visualization(
                        frame, gray, equalized, lbp1, lbp2, lbp3, magnitude
                    )
                except Exception as e:
                    logger.error(f"Error creating texture visualization: {e}")
            
            return is_real_texture
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            logger.error(traceback.format_exc())
            # Be lenient in case of errors
            return True
    
    def _calculate_entropy(self, image):
        """
        Calculate entropy of an image.
        
        Args:
            image: Grayscale image
            
        Returns:
            float: Entropy value
        """
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        hist = hist.astype(float) / (np.sum(hist) + 1e-7)
        return -np.sum(hist * np.log2(hist + 1e-7))
    
    def _create_texture_visualization(self, frame, gray, equalized, lbp1, lbp2, lbp3, magnitude):
        """
        Create visualization of texture analysis results.
        
        Args:
            frame: Original color frame
            gray: Grayscale version
            equalized: Histogram equalized image
            lbp1: LBP at radius 1
            lbp2: LBP at radius 2
            lbp3: LBP at radius 3
            magnitude: Gradient magnitude
        """
        # Normalize LBP images for visualization
        micro_texture = cv2.normalize(lbp1.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        medium_texture = cv2.normalize(lbp2.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        macro_texture = cv2.normalize(lbp3.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        gradient_viz = cv2.normalize(magnitude.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
        
        # Create a visualization with layout
        h, w = frame.shape[:2]
        viz_size = (w//2, h//2)  # Smaller size for the visualization
        
        # Resize images for the visualization
        frame_resized = cv2.resize(frame, viz_size)
        gray_resized = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), viz_size)
        equalized_resized = cv2.resize(cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR), viz_size)
        
        lbp1_viz = cv2.resize(cv2.cvtColor(micro_texture, cv2.COLOR_GRAY2BGR), viz_size)
        lbp2_viz = cv2.resize(cv2.cvtColor(medium_texture, cv2.COLOR_GRAY2BGR), viz_size)
        lbp3_viz = cv2.resize(cv2.cvtColor(macro_texture, cv2.COLOR_GRAY2BGR), viz_size)
        
        gradient_viz = cv2.resize(cv2.cvtColor(gradient_viz, cv2.COLOR_GRAY2BGR), viz_size)
        
        # Create rows of images
        row1 = np.hstack([frame_resized, gray_resized])
        row2 = np.hstack([lbp1_viz, lbp3_viz])
        row3 = np.hstack([gradient_viz, equalized_resized])
        
        # Combine rows
        visualization = np.vstack([row1, row2, row3])
        
        # Add labels
        cv2.putText(visualization, "TEXTURE ANALYSIS", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Display the visualization
        cv2.imshow("Texture Analysis", visualization)
        cv2.waitKey(100)  # Brief display 