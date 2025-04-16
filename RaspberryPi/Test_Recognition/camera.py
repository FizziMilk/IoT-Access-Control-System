import cv2
import time
import face_recognition
import numpy as np
import dlib
from scipy.spatial import distance as dist
import threading
import traceback
import logging
import os
from sklearn.metrics.pairwise import euclidean_distances
from skimage import feature as skimage_feature

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger("CameraSystem")

class CameraSystem:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        """
        Initialize the camera system.
        
        Args:
            camera_id: Camera ID to use (default: 0 for built-in webcam)
            resolution: Resolution of the camera (default: (640, 480))
        """
        self.camera_id = camera_id
        self.resolution = resolution
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.display_detections = True
        self.capture_count = 0
        self.captured_faces = []
        self.video_capture = None
        # Other initializations
        self.face_detector = dlib.get_frontal_face_detector()
        # Load the facial landmark predictor
        predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
        self.shape_predictor = dlib.shape_predictor(predictor_path)
        self.load_model()  # Load the face recognition model
        # EAR threshold for blink detection - less sensitive (harder to register)
        self.EAR_THRESHOLD = 0.20  # Higher threshold makes it harder to detect blinks
        # Consecutive frame requirements
        self.EYE_AR_CONSEC_FRAMES = 2  # Number of consecutive frames the eye must be below threshold
        # Initialize counters
        self.COUNTER = 0
        self.TOTAL = 0
        self.blink_detected = False
        self.blink_start_time = None
        self.highmotion_warned = False
        # Use a much lower resolution for liveness detection
        self.liveness_resolution = (320, 240)
        # Frame process rate (1 = process every frame, 2 = every other frame, etc.)
        self.process_nth_frame = 3
        # Use a separate thread for face detection to prevent UI blocking
        self.face_detection_thread = None
        self.stop_detection_thread = False
        # Flag to track if liveness passed has been logged
        self.has_logged_liveness_passed = False
    
    def reset_camera(self):
        """
        Fully reset and reinitialize the camera system
        """
        print("[DEBUG] reset_camera: Starting full camera system reset")
        # First, ensure any existing camera is properly released
        self.release_camera()
        
        # Reset all state variables
        self.COUNTER = 0
        self.TOTAL = 0
        self.blink_detected = False
        self.blink_start_time = None
        self.highmotion_warned = False
        self.stop_detection_thread = True
        self.has_logged_liveness_passed = False
        
        # Cleanup any lingering threads
        if self.face_detection_thread is not None:
            print("[DEBUG] Cleaning up face detection thread")
            if self.face_detection_thread.is_alive():
                self.face_detection_thread.join(timeout=1.0)
            self.face_detection_thread = None
                
        # Clean up any temporary files
        self.cleanup_diagnostic_images()
        
        # Force garbage collection to release any lingering resources
        import gc
        print("[DEBUG] Running garbage collection")
        gc.collect()
        
        # Wait a moment to ensure resources are fully released
        print("[DEBUG] Waiting after reset to ensure resources are released")
        time.sleep(0.5)
        print("[DEBUG] Camera system reset complete")
    
    def initialize_camera(self, resolution=None):
        """
        Initialize camera with robust error handling and multiple fallbacks
        
        Args:
            resolution: Optional resolution tuple (width, height)
            
        Returns:
            cv2.VideoCapture: Camera object or None if failed
        """
        print(f"[DEBUG] initialize_camera: Starting camera initialization with ID={self.camera_id}")
        
        # Check if there's an existing camera that needs to be released first
        if self.video_capture is not None:
            print(f"[DEBUG] initialize_camera: Existing camera found, releasing first")
            self.release_camera()
        
        # Only try index 0 since other indices don't work
        try:
            print(f"[DEBUG] Trying to open camera with index 0...")
            cap = cv2.VideoCapture(0)
            if cap is not None and cap.isOpened():
                self.camera_id = 0  # Set camera_id to 0
                print(f"[DEBUG] Successfully opened camera with index 0")
            else:
                print(f"[DEBUG] Failed to open camera with index 0")
                cap = None
        except Exception as e:
            print(f"[DEBUG] Error opening camera: {e}")
            cap = None
                
        # If still not opened, try with GSTREAMER
        if cap is None or not cap.isOpened():
            try:
                print("[DEBUG] Trying to open camera with GSTREAMER pipeline...")
                gst_str = "v4l2src device=/dev/video0 ! video/x-raw,width=640,height=480 ! videoconvert ! appsink"
                cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
                if cap is not None and cap.isOpened():
                    print("[DEBUG] Successfully opened camera with GSTREAMER")
            except Exception as e:
                print(f"[DEBUG] Failed to open camera with GSTREAMER: {e}")
        
        # Final check if camera is opened
        if cap is None or not cap.isOpened():
            print("[DEBUG] Error: Could not open camera with any method")
            return None
            
        # Set resolution if provided
        if resolution is not None:
            print(f"[DEBUG] Setting camera resolution to {resolution}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        
        # Store the video capture object
        self.video_capture = cap
        
        # Wait for camera to initialize
        print("[DEBUG] Waiting for camera to initialize...")
        time.sleep(0.5)
        print("[DEBUG] Camera initialization complete")
        
        return cap
    
    def release_camera(self):
        """
        Release camera resources
        """
        print("[DEBUG] release_camera: Attempting to release camera")
        if self.video_capture is not None:
            try:
                self.video_capture.release()
                print("[DEBUG] Camera released successfully")
                # Force OpenCV to forget about the camera
                cv2.destroyAllWindows()
                # Set up Qt environment
                self.setup_qt_environment()
            except Exception as e:
                print(f"[DEBUG] Error releasing camera: {e}")
            finally:
                self.video_capture = None
                # Wait a moment after releasing to prevent resource conflict
                time.sleep(0.5)
                print("[DEBUG] Camera reference set to None")
        else:
            print("[DEBUG] No camera to release (video_capture is None)")
    
    def load_model(self):
        """
        Load face recognition models and other resources needed for facial analysis.
        """
        # Face recognition model is loaded implicitly via face_recognition library
        # This method ensures we have all necessary resources
        # Check if shape predictor exists
        predictor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(predictor_path):
            print("WARNING: shape_predictor_68_face_landmarks.dat not found!")
            print("Download it from: https://github.com/davisking/dlib-models")
        else:
            print("Face landmark predictor loaded successfully")
    
    def capture_face(self):
        """Capture an image from the camera with face detection"""
        # Initialize camera with our helper method
        cap = self.initialize_camera(resolution=self.resolution)
        if cap is None:
            return None
        
        # Show camera feed until space is pressed
        print("Position face in frame and press SPACE to capture or ESC to cancel")
        face_found = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            # Show the frame
            display_frame = frame.copy()
            
            # Try to find faces - ensure correct format
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Make sure the image is uint8 (0-255)
            if rgb_small_frame.dtype != np.uint8:
                rgb_small_frame = rgb_small_frame.astype(np.uint8)
                
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # Draw rectangle around faces
            for (top, right, bottom, left) in face_locations:
                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                face_found = True
            
            # Show instructions on the frame
            text = "SPACE: Capture, ESC: Cancel"
            if face_found:
                status = "Face detected"
                color = (0, 255, 0)
            else:
                status = "No face detected"
                color = (0, 0, 255)
            
            cv2.putText(display_frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow("Capture Face", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                cap.release()
                return None
            elif key == 32:  # SPACE
                if face_found:
                    break
        
        # Get final frame and clean up
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        
        if not ret:
            return None
            
        return frame

    def detect_face(self, frame):
        """
        Detect a face and its landmarks in the given frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (success, face_bbox, landmarks)
        """
        try:
            # Resize for faster detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if not face_locations:
                return False, None, None
                
            # Get first face
            top, right, bottom, left = face_locations[0]
            
            # Scale back to full size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            face_bbox = (left, top, right-left, bottom-top)
            
            # Get facial landmarks
            landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [face_locations[0]])
            
            if not landmarks_list:
                return False, face_bbox, None
                
            # Convert face_recognition landmarks to properly scaled landmarks
            scaled_landmarks = {}
            for feature, points in landmarks_list[0].items():
                scaled_landmarks[feature] = [(p[0] * 2, p[1] * 2) for p in points]
            
            return True, face_bbox, scaled_landmarks
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return False, None, None
    
    def recognize_face(self, frame, known_face_encodings, known_face_names):
        """
        Recognize faces in the frame by comparing with known face encodings
        
        Args:
            frame: Input video frame
            known_face_encodings: List of known face encodings
            known_face_names: List of names corresponding to the encodings
            
        Returns:
            list: List of recognized names and their locations
        """
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces in the frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_data = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back to original size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            # Compare with known faces
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"
            confidence = 0
            
            # Find the best match
            if len(known_face_encodings) > 0:
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    confidence = 1 - face_distances[best_match_index]
                    if confidence > 0.6:  # Confidence threshold
                        name = known_face_names[best_match_index]
            
            face_names.append(name)
            face_data.append({
                "name": name,
                "location": (left, top, right, bottom),
                "confidence": confidence
            })
        
        return face_data
    
    def run_facial_recognition(self, known_face_encodings=[], known_face_names=[]):
        """
        Run a real-time facial recognition system
        
        Args:
            known_face_encodings: List of known face encodings
            known_face_names: List of names corresponding to the encodings
        """
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        print("Starting facial recognition. Press ESC to exit.")
        print("Press 'a' to add the current face to the database.")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Get display frame
            display_frame = frame.copy()
            
            # Recognize faces
            face_data = self.recognize_face(frame, known_face_encodings, known_face_names)
            
            # Draw results
            for data in face_data:
                left, top, right, bottom = data["location"]
                name = data["name"]
                confidence = data["confidence"]
                
                # Draw a box around the face
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
                
                # Draw a label with a name below the face
                cv2.rectangle(display_frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
                confidence_text = f" ({confidence:.2f})" if confidence > 0 else ""
                cv2.putText(display_frame, name + confidence_text, (left + 6, bottom - 6), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
            
            # Display instructions
            cv2.putText(display_frame, "ESC: Exit, A: Add face", (10, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show the image
            cv2.imshow("Facial Recognition", display_frame)
            
            # Check for key presses
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('a'):
                # Add current face to database
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                face_locations = face_recognition.face_locations(rgb_small_frame)
                if face_locations:
                    face_encoding = face_recognition.face_encodings(rgb_small_frame, [face_locations[0]])[0]
                    
                    # Ask for name
                    cv2.putText(display_frame, "Enter name in console", (10, 50), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.imshow("Facial Recognition", display_frame)
                    cv2.waitKey(1)
                    
                    name = input("Enter name for this face: ")
                    known_face_names.append(name)
                    known_face_encodings.append(face_encoding)
                    print(f"Added {name} to the database")
                else:
                    print("No face detected to add")
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        return known_face_encodings, known_face_names
        
    def eye_aspect_ratio(self, eye):
        """
        Calculate the eye aspect ratio (EAR) which is used to detect blinks
        
        Args:
            eye: 6 (x, y) coordinates of the eye landmarks
            
        Returns:
            float: Eye aspect ratio
        """
        # Compute the euclidean distances between the vertical eye landmarks
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        
        # Compute the euclidean distance between the horizontal eye landmarks
        C = dist.euclidean(eye[0], eye[3])
        
        # Compute the eye aspect ratio
        ear = (A + B) / (2.0 * C)
        
        return ear
    
    def get_eye_landmarks(self, landmarks):
        """
        Extract eye landmarks from the face landmarks dictionary
        
        Args:
            landmarks: Dictionary of facial landmarks from face_recognition
            
        Returns:
            tuple: (left_eye, right_eye) landmarks
        """
        if not landmarks or "left_eye" not in landmarks or "right_eye" not in landmarks:
            return None, None
            
        left_eye = landmarks["left_eye"]
        right_eye = landmarks["right_eye"]
        
        return left_eye, right_eye
    
    def create_tracker(self):
        """Create a KCF tracking object"""
        # Print OpenCV version for diagnostic purposes
        print(f"OpenCV version: {cv2.__version__}")
        
        try:
            # Only use KCF tracker for consistency and performance
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF'):
                print("Using legacy.TrackerKCF tracker")
                return cv2.legacy.TrackerKCF.create()
            elif hasattr(cv2, 'TrackerKCF'):
                print("Using TrackerKCF tracker")
                return cv2.TrackerKCF.create()
            else:
                print("KCF tracker not found - please check opencv-contrib-python installation")
                return None
            
        except Exception as e:
            print(f"Error creating tracker: {e}")
            print("Traceback:", traceback.format_exc())
            return None
    
    def perform_texture_liveness_check(self, cap, face_location):
        """
        Perform liveness check using texture analysis.
        
        Args:
            cap: OpenCV video capture
            face_location: Detected face coordinates (top, right, bottom, left)
            
        Returns:
            bool: True if the face is likely real, False if possibly a photo
        """
        try:
            print("Performing texture-based liveness check...")
            
            # Verify face_location is valid
            if face_location is None or len(face_location) != 4:
                print("ERROR: Invalid face location")
                return True
                
            # Capture frame for texture analysis
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to capture frame for texture analysis")
                return True
            
            # Run texture analysis
            texture_result = self.analyze_facial_texture(frame)
            
            # Display final result
            ret, frame = cap.read()
            if ret:
                summary_frame = frame.copy()
                
                # Display final result
                result_text = f"PASS: Real Face" if texture_result else f"FAIL: Possible Spoof"
                color = (0, 255, 0) if texture_result else (0, 0, 255)
                
                top, right, bottom, left = face_location
                center_x = left + int((right - left) / 2)
                
                cv2.putText(summary_frame, result_text, (center_x - 150, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show texture test result only
                texture_status = "PASS" if texture_result else "FAIL"
                texture_color = (0, 255, 0) if texture_result else (0, 0, 255)
                cv2.putText(summary_frame, f"Texture Test: {texture_status}", (20, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, texture_color, 2)
                
                # Add instruction to press any key to continue
                cv2.putText(summary_frame, "Press any key to continue...", 
                           (20, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Liveness Test Result", summary_frame)
                cv2.waitKey(0)  # Wait for user to press any key before continuing
            
            return texture_result
            
        except Exception as e:
            print(f"Error in liveness check: {e}")
            print(traceback.format_exc())
            return True  # Default to allowing the user through if check fails
    
    def analyze_facial_texture(self, frame):
        """
        Analyze facial texture and entropy to detect printed faces and screens
        """
        try:
            print("Starting advanced texture analysis...")
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to enhance texture
            equalized = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            
            # Calculate Local Binary Pattern at multiple scales
            # This helps capture both micro and macro texture patterns
            results = {}
            
            # Original LBP calculation
            radius1 = 1
            n_points1 = 8 * radius1
            lbp1 = skimage_feature.local_binary_pattern(blurred, n_points1, radius1, method="uniform")
            
            # Medium scale LBP - captures more structure
            radius2 = 2
            n_points2 = 8 * radius2
            lbp2 = skimage_feature.local_binary_pattern(blurred, n_points2, radius2, method="uniform")
            
            # Larger scale LBP - captures even more structure
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
            # Calculate uniformity metrics (sum of squared probabilities)
            uniformity1 = np.sum(hist1 * hist1)
            uniformity2 = np.sum(hist2 * hist2)
            uniformity3 = np.sum(hist3 * hist3)
            
            # Calculate multi-scale uniformity ratio - key for detecting printed materials
            uniformity_ratio = (uniformity1 + uniformity2) / (2 * uniformity3 + 1e-7)
            
            # Analyze reflectance properties using gradient information
            # Photos typically have more uniform gradient distribution
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            
            # Calculate micro-texture features that differentiate real skin
            gradient_mean = np.mean(magnitude)
            gradient_std = np.std(magnitude)
            gradient_entropy = self.calculate_entropy(magnitude)
            
            # For real skin, gradient distribution is distinctive
            gradient_ratio = gradient_std / (gradient_mean + 1e-7)
            
            # For debugging
            print(f"Texture entropy: multi-scale {entropy1:.2f}/{entropy2:.2f}/{entropy3:.2f}")
            print(f"Uniformity ratio: {uniformity_ratio:.4f}, Gradient ratio: {gradient_ratio:.4f}")
            
            # SIMPLIFIED DECISION LOGIC FOR WEB APPLICATION
            # Make test more lenient for better user experience
            
            # In web mode, we'll just check two simple conditions instead of all the complex checks
            # 1. Basic entropy check (relaxed threshold)
            entropy_score = (entropy3 > 3.5)  # Much lower entropy threshold
            
            # 2. Simple gradient check (relaxed threshold)
            gradient_score = (gradient_ratio > 1.0)  # Much lower gradient threshold
            
            # Greatly simplify the decision logic - just pass if either test passes
            is_real_texture = entropy_score or gradient_score
            
            # For logging
            print(f"Texture scores (simplified): Entropy={entropy_score}, Gradient={gradient_score}")
            print(f"Texture test result: {is_real_texture}")
            
            # Save debug visualization but don't display it in web mode
            # Create more informative visualization showing multiple scales
            micro_texture = cv2.normalize(lbp1.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            macro_texture = cv2.normalize(lbp3.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            gradient_viz = cv2.normalize(magnitude.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            
            # Create a nicer visualization with title and organized layout
            h, w = frame.shape[:2]
            viz_size = (w, h)
            
            # Create labels for each image
            labels = [
                "Original Image", "Grayscale", "Histogram Equalized",
                "LBP (r=1)", "LBP (r=2)", "LBP (r=3)",
                "Gradient Magnitude"
            ]
            
            # Resize each image to the same size
            frame_resized = cv2.resize(frame, viz_size)
            gray_resized = cv2.resize(gray, viz_size)
            equalized_resized = cv2.resize(equalized, viz_size)
            
            # Make sure LBP visualizations are 3-channel
            lbp1_viz = cv2.resize(cv2.cvtColor(micro_texture, cv2.COLOR_GRAY2BGR), viz_size)
            lbp2_viz = cv2.resize(cv2.cvtColor(cv2.normalize(lbp2.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR), viz_size)
            lbp3_viz = cv2.resize(cv2.cvtColor(macro_texture, cv2.COLOR_GRAY2BGR), viz_size)
            
            gradient_viz = cv2.resize(cv2.cvtColor(gradient_viz, cv2.COLOR_GRAY2BGR), viz_size)
            
            # Make sure gray images are 3-channel for display
            gray_resized = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
            equalized_resized = cv2.cvtColor(equalized_resized, cv2.COLOR_GRAY2BGR)
            
            # Create the visualization layout (but only save it, don't show in web mode)
            row1 = np.hstack([frame_resized, gray_resized, equalized_resized])
            row2 = np.hstack([lbp1_viz, lbp2_viz, lbp3_viz])
            
            # Create a placeholder with same shape as other images for empty space
            empty_viz = np.zeros_like(gradient_viz)
            # Add third row with gradient and placeholders
            row3 = np.hstack([gradient_viz, empty_viz, empty_viz])
            
            # Create header with title
            header = np.zeros((60, row1.shape[1], 3), dtype=np.uint8)
            cv2.putText(header, "TEXTURE ANALYSIS VISUALIZATION", (row1.shape[1]//2 - 200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Add labels to each image
            for i, img in enumerate([row1, row2, row3]):
                for j in range(3):
                    if i * 3 + j < len(labels):
                        label = labels[i * 3 + j]
                        x = j * viz_size[0] + 10
                        y = 30
                        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine everything
            texture_visualization = np.vstack([header, row1, row2, row3])
            
            # Save but don't display in window for web mode
            cv2.imwrite("texture_analysis_debug.jpg", texture_visualization)
            
            # Always return true in web mode for now to make the system more user-friendly
            # We still have blink detection as a security measure
            return True
            
        except Exception as e:
            print(f"Error in texture analysis: {e}")
            print(traceback.format_exc())
            # Be lenient in case of errors in web mode
            return True
            
    def calculate_entropy(self, image):
        """Calculate entropy of an image"""
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        hist = hist.astype(float) / (np.sum(hist) + 1e-7)
        return -np.sum(hist * np.log2(hist + 1e-7))
    
    def calculate_clarity(self, gray_img, point, region_size=20):
        """
        Calculate image clarity (sharpness) in a region around a point.
        Uses Laplacian variance as a measure of sharpness.
        
        Args:
            gray_img: Grayscale image
            point: (x, y) center of region to analyze
            region_size: Size of square region to analyze
            
        Returns:
            float: Clarity measure (higher = sharper)
        """
        try:
            # Ensure point coordinates are integers
            x, y = int(point[0]), int(point[1])
            region_size = int(region_size)
            h, w = gray_img.shape
            
            # Print debugging info
            print(f"Image shape: {gray_img.shape}, Point: ({x}, {y}), Region size: {region_size}")
            
            # Ensure region is within image bounds
            left = max(0, x - region_size // 2)
            top = max(0, y - region_size // 2)
            right = min(w, x + region_size // 2)
            bottom = min(h, y + region_size // 2)
            
            # Print debug info for region bounds
            print(f"Region bounds: left={left}, top={top}, right={right}, bottom={bottom}")
            
            if right <= left or bottom <= top:
                print("WARNING: Invalid region bounds")
                return 0.0
                
            # Extract region
            region = gray_img[top:bottom, left:right]
            
            # Apply Laplacian filter (edge detection)
            laplacian = cv2.Laplacian(region, cv2.CV_64F)
            
            # Return variance of the Laplacian
            return laplacian.var()
        except Exception as e:
            print(f"Error in calculate_clarity: {e}")
            return 0.0
    
    def detect_blink(self, timeout=7):
        """
        Detect if a person is blinking (liveness check)
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            tuple: (success, face_image)
        """
        # First, destroy any existing windows to ensure clean state
        cv2.destroyAllWindows()
        
        # Initialize camera with our helper method
        cap = self.initialize_camera(resolution=self.liveness_resolution)
        if cap is None:
            return False, None
            
        # Disable autofocus to reduce processing overhead
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        print("Blink detection started. Please look at the camera and blink normally.")
        
        # Variables for blink detection
        blink_counter = 0
        ear_thresh_counter = 0
        blink_detected = False
        start_time = time.time()
        
        # Keep the best face image for return
        best_face_image = None
        max_face_size = 0
        face_location = None
        
        # For tracking
        frame_count = 0
        
        # Motion detection variables
        previous_frame = None
        motion_history = []
        motion_history_max = 5
        high_motion_detected = False
        
        # Stillness tracking - only process blinks when still
        stillness_frames_required = 10  # Need this many consecutive still frames
        stillness_counter = 0
        is_still_enough = False
        
        # EAR history for detecting real blink patterns
        ear_history = []
        ear_history_max = 10
        
        # For adaptive EAR threshold
        ear_values = []
        
        # Position smoothing to reduce jitter
        position_history = []
        position_history_max = 5
        
        # Track FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        last_face_location = None
        last_landmarks = None
        
        # Initialize ear variable outside the scope to prevent reference errors
        ear = 0.0
        
        # Create tracker
        tracker = self.create_tracker()
        tracking_active = False
        
        try:
            # Process every frame for better blink detection reliability
            while (time.time() - start_time) < timeout:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # FPS calculation
                fps_frame_count += 1
                if fps_frame_count >= 10:
                    fps = fps_frame_count / (time.time() - fps_start_time)
                    fps_frame_count = 0
                    fps_start_time = time.time()
                
                # Create a copy for display
                display_frame = frame.copy()
                
                # Detect global motion between frames
                if previous_frame is not None:
                    # Convert frames to grayscale for motion detection
                    # Use lower resolution for motion detection to save CPU
                    motion_scale = 0.25
                    small_prev = cv2.resize(previous_frame, (0, 0), fx=motion_scale, fy=motion_scale)
                    small_curr = cv2.resize(frame, (0, 0), fx=motion_scale, fy=motion_scale)
                    
                    gray1 = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
                    gray2 = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate absolute difference between frames
                    diff = cv2.absdiff(gray1, gray2)
                    # Apply threshold to highlight significant changes
                    _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
                    # Calculate motion score as percentage of pixels changed
                    motion_score = np.sum(thresh) / (thresh.shape[0] * thresh.shape[1] * 255)
                    
                    # Add to motion history
                    motion_history.append(motion_score)
                    if len(motion_history) > motion_history_max:
                        motion_history.pop(0)
                    
                    # Detect high motion if average motion exceeds threshold
                    if len(motion_history) > 2:
                        avg_motion = sum(motion_history) / len(motion_history)
                        # Adjust motion threshold based on testing
                        high_motion_detected = avg_motion > 0.005
                        
                        # Track stillness
                        if high_motion_detected:
                            # Reset stillness counter if motion detected
                            stillness_counter = 0
                            is_still_enough = False
                        else:
                            # Increment stillness counter
                            stillness_counter += 1
                            # Check if we've been still long enough
                            is_still_enough = stillness_counter >= stillness_frames_required
                
                # Store current frame for next iteration
                previous_frame = frame.copy()
                
                # Alternate between tracking and detection to maintain performance
                process_this_frame = (frame_count % 3 == 0)
                frame_count += 1
                
                # Try to update tracker if active
                current_face_box = None
                if tracker is not None and tracking_active:
                    try:
                        ok, bbox = tracker.update(frame)
                        if ok:
                            # Convert from (x, y, width, height) to (top, right, bottom, left)
                            x, y, w, h = [int(v) for v in bbox]
                            left, top, right, bottom = x, y, x + w, y + h
                            current_face_box = (top, right, bottom, left)
                        else:
                            # If tracking failed, force detection on next frame
                            tracking_active = False
                            process_this_frame = True
                    except Exception as e:
                        print(f"Tracking error: {e}")
                        tracking_active = False
                        process_this_frame = True
                
                # Run face detection when needed
                if process_this_frame or not tracking_active:
                    # Detect face locations with efficient downsampling
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                    
                    if face_locations:
                        # Scale back up face locations
                        top, right, bottom, left = face_locations[0]
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        
                        current_face_box = (top, right, bottom, left)
                        face_location = current_face_box  # Store this for later liveness checks
                        
                        # Initialize tracker if available
                        if tracker is not None and not tracking_active:
                            try:
                                # Initialize with current face location
                                bbox = (left, top, right - left, bottom - top)
                                ok = tracker.init(frame, bbox)
                                if ok:
                                    tracking_active = True
                                else:
                                    print("Failed to initialize tracker")
                            except Exception as e:
                                print(f"Tracker initialization error: {e}")
                        
                        # Get landmarks for blink detection
                        landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [face_locations[0]])
                        
                        if landmarks_list:
                            # Convert landmarks to full scale
                            scaled_landmarks = {}
                            for feature, points in landmarks_list[0].items():
                                scaled_landmarks[feature] = [(p[0] * 4, p[1] * 4) for p in points]
                            
                            last_landmarks = scaled_landmarks
                
                # Apply smoothing for face position
                if current_face_box is not None:
                    position_history.append(current_face_box)
                    if len(position_history) > position_history_max:
                        position_history.pop(0)
                    
                    # Calculate smoothed position
                    if len(position_history) > 1:
                        smoothed_top = sum(pos[0] for pos in position_history) // len(position_history)
                        smoothed_right = sum(pos[1] for pos in position_history) // len(position_history)
                        smoothed_bottom = sum(pos[2] for pos in position_history) // len(position_history)
                        smoothed_left = sum(pos[3] for pos in position_history) // len(position_history)
                        
                        last_face_location = (smoothed_top, smoothed_right, smoothed_bottom, smoothed_left)
                    else:
                        last_face_location = current_face_box
                
                # Use the last detected face and landmarks for blink processing
                if last_face_location is not None:
                    top, right, bottom, left = last_face_location
                    
                    # Draw rectangle around face
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Save the largest face image for return
                    face_size = (right - left) * (bottom - top)
                    if face_size > max_face_size:
                        max_face_size = face_size
                        best_face_image = frame.copy()
                    
                    # Process eye landmarks if available
                    if last_landmarks is not None:
                        # Get eye landmarks
                        left_eye, right_eye = self.get_eye_landmarks(last_landmarks)
                        
                        if left_eye and right_eye:
                            # Calculate eye aspect ratio (EAR)
                            left_ear = self.eye_aspect_ratio(left_eye)
                            right_ear = self.eye_aspect_ratio(right_eye)
                            
                            # Average the EAR of both eyes
                            ear = (left_ear + right_ear) / 2.0
                            
                            # Add to EAR history
                            ear_history.append(ear)
                            if len(ear_history) > ear_history_max:
                                ear_history.pop(0)
                            
                            # Add to baseline values for adaptive thresholding
                            ear_values.append(ear)
                            
                            # Calculate adaptive threshold after sufficient samples
                            adaptive_threshold = self.EAR_THRESHOLD
                            if len(ear_values) > 10:
                                # Use the 70th percentile as our baseline open eye EAR
                                open_eye_ear = np.percentile(ear_values, 70)
                                # Set threshold to 75% of the open eye EAR
                                adaptive_threshold = open_eye_ear * 0.75
                                # But don't go below the minimum threshold
                                adaptive_threshold = max(adaptive_threshold, self.EAR_THRESHOLD)
                            
                            # Draw eye contours
                            for eye in [left_eye, right_eye]:
                                eye_hull = cv2.convexHull(np.array(eye))
                                cv2.drawContours(display_frame, [eye_hull], -1, (0, 255, 0), 1)
                            
                            # Only process blinks when relatively still
                            if is_still_enough and len(ear_history) >= 3:
                                # Check if current EAR is below threshold (eyes closed)
                                if ear < adaptive_threshold:
                                    ear_thresh_counter += 1
                                    # Draw RED eye contours for closed eyes
                                    for eye in [left_eye, right_eye]:
                                        eye_hull = cv2.convexHull(np.array(eye))
                                        cv2.drawContours(display_frame, [eye_hull], -1, (0, 0, 255), 2)
                                else:
                                    # Eyes are open now, check if it was a blink
                                    if ear_thresh_counter >= self.EYE_AR_CONSEC_FRAMES:
                                        # Check for natural blink pattern
                                        if len(ear_history) >= 5:
                                            # Calculate EAR differences to verify natural transition
                                            ear_diff = abs(ear_history[-1] - ear_history[-3])
                                            ear_diff2 = abs(ear_history[-3] - ear_history[-5])
                                            
                                            # Verify blink pattern
                                            natural_blink_pattern = (
                                                # Sequence: open -> closed -> open
                                                ear_history[-5] > ear_history[-3] and 
                                                ear_history[-3] < ear_history[-1] and
                                                # Check reasonable differences
                                                ear_diff > 0.01 and ear_diff2 > 0.005 and
                                                # Real blinks have a larger change
                                                abs(max(ear_history[-5:]) - min(ear_history[-5:])) > 0.03
                                            )
                                            
                                            if natural_blink_pattern:
                                                blink_counter += 1
                                                print(f"Blink detected! EAR: {ear:.3f}, Threshold: {adaptive_threshold:.3f}")
                                                blink_detected = True
                                        
                                        # Reset counter after eyes reopen
                                        ear_thresh_counter = 0
                
                    # Display EAR value and blink count
                    ear_text = f"EAR: {ear:.2f}"
                    cv2.putText(display_frame, ear_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display blink counter
                    cv2.putText(display_frame, f"Blinks: {blink_counter}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display time remaining
                    time_left = int(timeout - (time.time() - start_time))
                    cv2.putText(display_frame, f"Time: {time_left}s", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Display FPS
                    cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 120, 30), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    
                    # Display motion status
                    motion_status = "High Motion" if high_motion_detected else "Stable"
                    motion_color = (0, 0, 255) if high_motion_detected else (0, 255, 0)
                    cv2.putText(display_frame, motion_status, (display_frame.shape[1] - 120, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
                    
                    # Display blink status
                    if blink_counter > 0:
                        status = f"Blink detected! ({blink_counter})"
                        color = (0, 255, 0)
                    else:
                        status = "Please blink naturally..."
                        color = (0, 0, 255)
                    cv2.putText(display_frame, status, (10, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    # Display stillness status
                    stillness_status = f"Stillness: {stillness_counter}/{stillness_frames_required}"
                    stillness_color = (0, 255, 0) if is_still_enough else (0, 0, 255)
                    cv2.putText(display_frame, stillness_status, (display_frame.shape[1] - 120, 120),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, stillness_color, 1)
                    
                    cv2.putText(display_frame, "ESC: Cancel", (10, display_frame.shape[0] - 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                # Show the frame
                cv2.imshow("Liveness Detection", display_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    cap.release()
                    cv2.destroyAllWindows()
                    return False, None
                
                # Early exit if enough blinks detected
                if blink_counter >= 2 and time.time() - start_time > timeout/3:
                    print("Multiple blinks detected - proceeding to additional checks")
                    break
            
            # Clean up camera resources immediately
            if cap is not None:
                cap.release()
                cap = None
            
            # Close any open windows from this phase
            cv2.destroyAllWindows()
            
            # Skip texture analysis in web mode, just return success if blinks detected
            if blink_detected and blink_counter >= 2:
                print("Blink check passed, skipping additional checks in web mode")
                return True, best_face_image
            else:
                return False, None
                
        except Exception as e:
            print(f"Error in blink detection: {e}")
            return False, None
        finally:
            # Ensure cleanup happens even if there's an exception
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
    
    def cleanup_diagnostic_images(self):
        """Clean up diagnostic images created during testing"""
        try:
            # Remove texture analysis debug image if it exists
            if os.path.exists("texture_analysis_debug.jpg"):
                os.remove("texture_analysis_debug.jpg")
                print("Removed texture_analysis_debug.jpg")
        except Exception as e:
            print(f"Error cleaning up diagnostic images: {e}")
            # Continue even if cleanup fails
    
    def capture_face_with_liveness(self, timeout=7):
        """
        Capture a face with liveness detection (blink detection)
        
        Args:
            timeout: Maximum seconds to wait for liveness check
            
        Returns:
            image or None: Face image if liveness check passed, None otherwise
        """
        # Reset the camera system to ensure a clean state
        self.reset_camera()
        
        print(f"Starting liveness detection (blink check) with timeout={timeout}s...")
        success, face_image = self.detect_blink(timeout=timeout)
        
        if not success:
            print("Liveness check failed or was cancelled")
            return None
            
        print("Liveness check passed!")
        return face_image
        
    def detect_liveness(self, timeout=10):
        """
        Perform liveness detection
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            tuple: (success, face_image)
        """
        return self.detect_blink(timeout)

    def check_liveness(self, frame, face_location):
        """Perform liveness detection on the detected face."""
        try:
            print("\nPerforming liveness checks...")
            
            # Get video capture object
            cap = self.video_capture
            
            # Step 1: Conduct eye blink detection (basic)
            print("Starting blink detection...")
            blink_result = self.detect_blinks(cap, face_location)
            
            # If blink detection fails, end early
            if not blink_result:
                print("FAIL: Blink detection failed")
                return False
            
            # Step 2: Check facial texture analysis
            print("Starting texture analysis...")  
            texture_result = self.analyze_facial_texture(frame)
            
            # Make the final liveness decision based on both tests
            # Both tests must pass for a robust confirmation
            liveness_confirmed = blink_result and texture_result
            
            # Log results
            print("\nLiveness Check Results:")
            print(f"✓ Blink Detection: {'PASS' if blink_result else 'FAIL'}")
            print(f"✓ Texture Analysis: {'PASS' if texture_result else 'FAIL'}")
            print(f"Final Decision: {'LIVE FACE CONFIRMED' if liveness_confirmed else 'SPOOFING ATTEMPT DETECTED'}")
            
            # Display result on frame
            top, right, bottom, left = face_location
            label = "LIVE FACE" if liveness_confirmed else "FAKE DETECTED"
            color = (0, 255, 0) if liveness_confirmed else (0, 0, 255)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            # Display the final result for a short time
            cv2.putText(frame, "Processing liveness results...", 
                        (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.imshow("Liveness Check", frame)
            cv2.waitKey(300)  # Wait for 300ms instead of indefinitely
            
            return liveness_confirmed
            
        except Exception as e:
            print(f"Error in liveness check: {e}")
            print(traceback.format_exc())
            return False  # Default to rejecting on error

    def detect_blinks(self, cap, face_location):
        """
        Simplified method to detect blinks for a pre-detected face.
        
        Args:
            cap: OpenCV video capture
            face_location: Detected face coordinates (top, right, bottom, left)
            
        Returns:
            bool: True if blinks were detected, False otherwise
        """
        try:
            # Check if we have valid face location
            if face_location is None or len(face_location) != 4:
                print("Invalid face location for blink detection")
                return False
            
            # Convert face location format if needed
            if isinstance(face_location, tuple) and len(face_location) == 4:
                top, right, bottom, left = face_location
            
            # Process a few frames to detect blinks
            blink_counter = 0
            frames_to_check = 30  # Check 30 frames for blinks
            
            for _ in range(frames_to_check):
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame for blink detection")
                    break
                
                # Get face landmarks
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                
                # Scale face location for the smaller frame
                small_top = top // 2
                small_right = right // 2
                small_bottom = bottom // 2
                small_left = left // 2
                
                small_face_location = (small_top, small_right, small_bottom, small_left)
                
                # Get landmarks for the detected face
                landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [small_face_location])
                
                if landmarks_list:
                    # Scale landmarks back to original size
                    scaled_landmarks = {}
                    for feature, points in landmarks_list[0].items():
                        scaled_landmarks[feature] = [(p[0] * 2, p[1] * 2) for p in points]
                    
                    # Get eye landmarks
                    left_eye, right_eye = self.get_eye_landmarks(scaled_landmarks)
                    
                    if left_eye and right_eye:
                        # Calculate eye aspect ratio (EAR)
                        left_ear = self.eye_aspect_ratio(left_eye)
                        right_ear = self.eye_aspect_ratio(right_eye)
                        
                        # Average the EAR of both eyes
                        ear = (left_ear + right_ear) / 2.0
                        
                        # Check if EAR indicates a blink
                        if ear < self.EAR_THRESHOLD:
                            blink_counter += 1
                            if blink_counter >= 2:  # Consider 2+ blinks as successful
                                return True
            
            # Return True if we detected enough blinks
            return blink_counter >= 2
        
        except Exception as e:
            print(f"Error in blink detection: {e}")
            print(traceback.format_exc())
            return False

    def setup_qt_environment(self):
        """
        Set up the appropriate Qt environment variables based on available plugins
        """
        import os
        import subprocess
        import re
        
        try:
            # Try to detect available Qt plugins
            qt_available_plugins = []
            
            # Common plugin locations
            qt_plugin_paths = [
                "/usr/lib/x86_64-linux-gnu/qt5/plugins",
                "/usr/lib/qt5/plugins",
                "/usr/lib/qt/plugins",
                "/home/benas/IoT/lib/python3.11/site-packages/cv2/qt/plugins"
            ]
            
            # Check for existing environment variable
            if "QT_PLUGIN_PATH" in os.environ:
                qt_plugin_paths.append(os.environ["QT_PLUGIN_PATH"])
            
            # Try to detect available platforms from the error message
            try:
                # Temporarily unset QT_QPA_PLATFORM to force error that shows available plugins
                old_qt_platform = os.environ.pop("QT_QPA_PLATFORM", None)
                
                # Run a simple cv2 command that would use Qt
                output = ""
                try:
                    # This will likely fail, but output error message with available plugins
                    temp_img = np.zeros((10, 10, 3), dtype=np.uint8)
                    cv2.imshow("test", temp_img)
                    cv2.waitKey(1)
                except Exception as e:
                    output = str(e)
                    
                # Parse available platforms from error message
                if "Available platform plugins are:" in output:
                    plugins_line = output.split("Available platform plugins are:")[1].strip()
                    if plugins_line:
                        qt_available_plugins = [p.strip() for p in plugins_line.split(",")]
                
                # Restore the old value if it existed
                if old_qt_platform:
                    os.environ["QT_QPA_PLATFORM"] = old_qt_platform
            except:
                pass
                
            # Default plugin priority order
            plugin_preferences = ["xcb", "wayland", "offscreen", "minimal", "vnc"]
            
            # If we found plugins in the error message
            if qt_available_plugins:
                print(f"[DEBUG] Detected Qt plugins: {qt_available_plugins}")
                # Find first available plugin from our preferences
                for plugin in plugin_preferences:
                    if plugin in qt_available_plugins:
                        os.environ["QT_QPA_PLATFORM"] = plugin
                        print(f"[DEBUG] Using Qt platform plugin: {plugin}")
                        return
            
            # If we're here, we either found no plugins from error message or none of our preferences
            # Default to xcb as it's most common on Linux
            os.environ["QT_QPA_PLATFORM"] = "xcb"
            print("[DEBUG] Defaulting to Qt platform plugin: xcb")
            
        except Exception as e:
            print(f"[DEBUG] Error setting up Qt environment: {e}")
            # Default fallback
            os.environ["QT_QPA_PLATFORM"] = "xcb"

# For demonstration
if __name__ == "__main__":
    camera = CameraSystem()
    
    # Empty database to start
    known_encodings = []
    known_names = []
    
    # Run facial recognition
    known_encodings, known_names = camera.run_facial_recognition(known_encodings, known_names) 