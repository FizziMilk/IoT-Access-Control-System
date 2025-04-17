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
import threading
import queue

# Simplify logging configuration to avoid duplicates
logger = logging.getLogger("WebCamera")

class WebCamera:
    """Camera handling optimized for web applications."""
    
    def __init__(self, config=None):
        """
        Initialize the Camera object.
        
        Args:
            config (dict, optional): Configuration dictionary for the camera.
        """
        # Set up configuration
        self.config = config or {}
        
        # Thread safety
        self._frame_lock = threading.RLock()
        self._camera_lock = threading.RLock()
        
        # Get headless flag from config
        self.headless = self.config.get('headless', True)
        
        # Set Eye Aspect Ratio threshold for blink detection
        self.EAR_THRESHOLD = self.config.get('ear_threshold', 0.20)
        
        # Initialize Qt application if needed
        self._qt_app = None
        self._setup_qt_environment()
        
        # Camera state
        self._camera = None
        self._running = False
        self._current_frame = None
        self._start_time = None
        self._frame_count = 0
        self._last_computed_fps = 0
        self._capture_thread = None
        self._stop_event = threading.Event()
        
        # Frame observers
        self._frame_observers = []
        
        # Default camera settings
        self._width = self.config.get('width', 640)
        self._height = self.config.get('height', 480)
        self._fps = self.config.get('fps', 30)
        self._device_index = self.config.get('device_index', 0)
        
        # Image processing settings
        self._face_recognition_model = self.config.get('face_recognition_model', 'hog')
        self._detection_frequency = self.config.get('detection_frequency', 5)  # every n frames
        
        # Initialize the face encoder
        self._face_encoder = None
        
        # Initialize video capture object
        self.cap = None
        
        # Track if camera is running
        self.is_running = False
        self.is_paused = False
        
        # Threading variables
        self.lock = threading.RLock()
        self.capture_thread = None
        self.stop_event = threading.Event()
        
        # Frame buffers
        self.latest_frame = None
        self.frame_queue = queue.Queue(maxsize=2)  # Small queue to avoid memory issues
        
        # This flag will control visualization during texture analysis
        self.texture_visualization = False
        
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
            self.cap = camera
            
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
        with self.camera_session(resolution or (self._width, self._height)):
            # Give the camera some time to adjust
            time.sleep(0.5)
            
            for attempt in range(max_attempts):
                logger.info(f"Capture attempt {attempt+1}/{max_attempts}")
                
                # Capture several frames to allow camera to adjust
                frames = []
                for _ in range(5):
                    ret, frame = self.cap.read()
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
        
        try:
            with self.camera_session((self._width, self._height)):
                start_time = time.time()
                blink_counter = 0
                ear_values = []
                
                # Only create a window if not in headless mode
                window_created = False
                if not self.headless:
                    try:
                        cv2.namedWindow("Blink Detection", cv2.WINDOW_NORMAL)
                        window_created = True
                    except Exception as e:
                        logger.error(f"Error creating window: {e}")
                
                # Continue until timeout or sufficient blinks detected
                while (time.time() - start_time) < timeout and blink_counter < 2:
                    ret, frame = self.cap.read()
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
                if window_created:
                    try:
                        cv2.destroyWindow("Blink Detection")
                        cv2.waitKey(1)  # Process window destruction
                    except Exception as e:
                        logger.error(f"Error destroying window: {e}")
                
                # Result
                result = blink_counter >= 2
                logger.info(f"Blink detection result: {result} ({blink_counter} blinks)")
                return result
                
        except Exception as e:
            logger.error(f"Error in blink detection: {e}")
            logger.error(traceback.format_exc())
            # Clean up any windows on error
            if not self.headless:
                cv2.destroyAllWindows()
            return False

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
        logger.info(f"Initializing camera with index {self._device_index}")
        
        # First release any existing camera - but only if not already done
        if self.cap is not None:
            self._release_camera()
        
        # Try to open camera
        try:
            logger.info(f"Opening camera with index {self._device_index}")
            cap = cv2.VideoCapture(self._device_index)
            
            if cap is None or not cap.isOpened():
                logger.warning(f"Failed to open camera with index {self._device_index}")
                return None
                
            logger.info(f"Successfully opened camera with index {self._device_index}")
            
            # Set resolution if provided
            if resolution is not None:
                logger.info(f"Setting camera resolution to {resolution}")
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            
            # Wait for camera to initialize
            time.sleep(0.2)  # Reduced from 0.5 to speed up initialization
            
            return cap
            
        except Exception as e:
            logger.error(f"Error initializing camera: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def _release_camera(self):
        """
        Release camera resources efficiently.
        """
        if self.cap is not None:
            try:
                logger.info("Releasing camera")
                self.cap.release()
                self.cap = None
                
                # Destroy windows in non-headless mode
                if not self.headless:
                    cv2.destroyAllWindows()
                
                # Brief wait to ensure resources are released
                time.sleep(0.1)
                logger.info("Camera resources released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
                logger.error(traceback.format_exc())
                # Still set camera to None even if release fails
                self.cap = None
    
    def _setup_qt_environment(self):
        """
        Set up environment variables for Qt/OpenCV compatibility.
        This prevents thread-related crashes in OpenCV's highgui module.
        """
        try:
            # Set QT_QPA_PLATFORM_PLUGIN_PATH to avoid errors
            if 'QT_QPA_PLATFORM_PLUGIN_PATH' not in os.environ:
                cv_path = os.path.dirname(cv2.__file__)
                qt_plugin_path = os.path.join(cv_path, 'qt', 'plugins')
                if os.path.isdir(qt_plugin_path):
                    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = qt_plugin_path
            
            # Force the same thread for Qt operations
            os.environ['QT_THREAD_PRIORITY_POLICY'] = '1'
            
            # Use xcb for all modes as it's the most reliable on Linux
            os.environ['QT_QPA_PLATFORM'] = 'xcb'
            
            # Make Qt run in the main thread
            os.environ['QT_FORCE_STDERR_LOGGING'] = '1'
        except Exception as e:
            logger.error(f"Error setting up Qt environment: {str(e)}")
            logger.error(traceback.format_exc())

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
            
            # Create visualization only if not in headless mode
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
        window_created = False
        try:
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
            
            # Create the window first
            cv2.namedWindow("Texture Analysis", cv2.WINDOW_NORMAL)
            window_created = True
            
            # Display the visualization
            cv2.imshow("Texture Analysis", visualization)
            cv2.waitKey(100)  # Brief display
            
            # Explicitly destroy window after display
            if window_created:
                cv2.destroyWindow("Texture Analysis")
                cv2.waitKey(1)  # Process window destruction
        except Exception as e:
            logger.error(f"Error creating texture visualization: {e}")
            # Ensure window is closed only if it was created
            if window_created:
                try:
                    cv2.destroyWindow("Texture Analysis")
                    cv2.waitKey(1)
                except:
                    pass

    def start(self):
        """
        Start the camera capture thread.
        
        This method initializes the camera and starts the frame capture in a separate thread.
        Returns True if camera started successfully, False otherwise.
        """
        if self._running:
            logger.warning("Camera is already running")
            return True
        
        try:
            # Initialize camera in the main thread before starting background thread
            logger.info(f"Opening camera with index {self._device_index}")
            self.cap = cv2.VideoCapture(self._device_index)
            
            if not self.cap.isOpened():
                logger.error(f"Failed to open camera with index {self._device_index}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
            self.cap.set(cv2.CAP_PROP_FPS, self._fps)
            
            # Get actual camera properties
            actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera initialized with resolution: {actual_width}x{actual_height}, FPS: {actual_fps}")
            
            # Reset control flags
            self._stop_event.clear()
            self._running = True
            
            # Start capture thread
            self._capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self._capture_thread.start()
            
            logger.info("Camera started successfully")
            return True
        
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            logger.error(traceback.format_exc())
            self.release()
            return False 

    def _capture_loop(self):
        """
        Main camera capture loop that runs in a separate thread.
        Continuously captures frames until stopped.
        """
        try:
            logger.info("Frame capture loop started")
            
            while not self._stop_event.is_set() and self.cap and self.cap.isOpened():
                # Capture frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    logger.warning("Failed to capture frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                # Handle the frame in a way that's thread-safe for Qt
                try:
                    # Process the frame without any Qt operations in this thread
                    processed_frame = self._process_frame(frame)
                    
                    # Update the frame in a thread-safe way
                    with self._frame_lock:
                        self._current_frame = processed_frame
                        self._new_frame_available = True
                    
                    # Notify observers about the new frame
                    self._notify_frame_observers(processed_frame)
                    
                except Exception as e:
                    logger.error(f"Error processing frame: {str(e)}")
                    logger.error(traceback.format_exc())
                
                # Sleep to maintain desired frame rate
                time.sleep(1.0 / self._fps)
                
        except Exception as e:
            logger.error(f"Error in capture loop: {str(e)}")
            logger.error(traceback.format_exc())
        finally:
            logger.info("Capture loop ending, releasing resources")
            self._cleanup_resources()
        
    def _process_frame(self, frame):
        """
        Process a frame before sending it to observers.
        This method can be overridden in subclasses to add custom processing.
        
        Args:
            frame (numpy.ndarray): The original frame from camera
            
        Returns:
            numpy.ndarray: The processed frame
        """
        # By default, return a copy of the frame to avoid observers modifying the original
        if frame is not None:
            return frame.copy()
        return None
        
    def _update_frame(self, frame):
        """
        Updates the current frame and notifies observers.
        
        Args:
            frame (numpy.ndarray): The new frame from camera
        """
        if frame is not None:
            with self._frame_lock:
                self._current_frame = frame.copy()
            
            # Process the frame before notifying observers
            processed_frame = self._process_frame(frame)
            self._notify_frame_observers(processed_frame)
    
    def add_frame_observer(self, observer):
        """
        Add a frame observer that will be notified when a new frame is available.
        
        Args:
            observer (callable): A callable that accepts a frame parameter.
                                The callable will be invoked with the latest frame.
        
        Returns:
            bool: True if the observer was added, False otherwise
        """
        if not callable(observer):
            logger.warning("Attempted to add non-callable frame observer")
            return False
            
        with self._frame_lock:
            if observer not in self._frame_observers:
                self._frame_observers.append(observer)
                logger.debug(f"Added frame observer, total observers: {len(self._frame_observers)}")
            return True
    
    def remove_frame_observer(self, observer):
        """
        Remove a previously registered frame observer.
        
        Args:
            observer (callable): The observer to remove
            
        Returns:
            bool: True if the observer was removed, False if it wasn't found
        """
        with self._frame_lock:
            if observer in self._frame_observers:
                self._frame_observers.remove(observer)
                logger.debug(f"Removed frame observer, remaining observers: {len(self._frame_observers)}")
                return True
            return False
    
    def _notify_frame_observers(self, frame):
        """
        Notify all registered observers about a new frame.
        
        This method makes a copy of the observer list under a lock to avoid modification
        during iteration. Each observer is called with the frame as an argument.
        
        Args:
            frame (numpy.ndarray): The processed frame to send to observers
        """
        observers_to_notify = []
        
        # Get a copy of observers under lock
        with self._frame_lock:
            observers_to_notify = self._frame_observers.copy()
            
        # Notify each observer with the new frame
        for observer in observers_to_notify:
            try:
                observer(frame)
            except Exception as e:
                logger.error(f"Error in frame observer: {e}")
                # Consider removing problematic observers here if needed
        
    def _cleanup_resources(self):
        """
        Clean up resources when the capture loop ends.
        """
        try:
            # Release camera resources if they exist
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.cap = None
                
            # Reset state flags
            self._running = False
            self._new_frame_available = False
            self._current_frame = None
            
            logger.info("Camera resources released")
        except Exception as e:
            logger.error(f"Error during resource cleanup: {str(e)}")

    def get_frame(self):
        """
        Safely retrieves the latest frame captured by the camera.
        
        Returns:
            numpy.ndarray or None: The most recent camera frame or None if no frame is available
        """
        if not self._running or not hasattr(self, '_current_frame'):
            logger.warning("Attempted to get frame but camera is not running")
            return None
        
        # Safely get the current frame using the lock
        with self._frame_lock:
            if self._current_frame is None:
                return None
            # Return a copy to avoid thread safety issues
            frame = self._current_frame.copy()
            self._new_frame_available = False
        
        return frame 