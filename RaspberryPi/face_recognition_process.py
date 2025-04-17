#!/usr/bin/env python3
"""
Standalone script for facial recognition and liveness detection.
This script is called by the Flask app as a separate process to avoid resource conflicts.
"""
import os
import sys
import time
import json
import argparse
import traceback
import logging
import cv2
import numpy as np
import face_recognition
import base64
import requests
from datetime import datetime
import threading

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/face_recognition_process.log')
    ]
)

logger = logging.getLogger("FaceRecognitionProcess")

# Import skimage for texture analysis
try:
    from skimage import feature as skimage_feature
except ImportError:
    logger.warning("skimage not available, using simplified liveness detection")
    skimage_feature = None


class LivenessDetector:
    """Standalone liveness detector for face anti-spoofing."""
    
    def __init__(self):
        """Initialize the liveness detector."""
        self.logger = logging.getLogger("LivenessDetector")
    
    def detect_liveness(self, face_image):
        """
        Perform liveness detection on a face image.
        
        Args:
            face_image: Image containing a face
            
        Returns:
            tuple: (is_real, score, debug_info)
        """
        try:
            # Analyze facial texture for liveness detection
            if skimage_feature is None:
                # Fallback if skimage is not available
                self.logger.warning("skimage not available, using basic detection")
                return True, 1.0, {"error": "skimage not available"}
            
            is_real, metrics = self.analyze_facial_texture(face_image)
            
            # Calculate overall liveness score
            score = metrics.get("texture_score", 0.0)
            
            return is_real, score, metrics
        
        except Exception as e:
            self.logger.error(f"Error in liveness detection: {e}")
            self.logger.error(traceback.format_exc())
            # Be lenient in case of errors
            return True, 0.0, {"error": str(e)}
    
    def analyze_facial_texture(self, frame):
        """
        Analyze facial texture and entropy to detect printed faces and screens.
        Uses multi-scale Local Binary Patterns (LBP) to analyze texture patterns.
        
        Args:
            frame: Image containing a face
            
        Returns:
            tuple: (is_real, metrics_dict)
        """
        try:
            self.logger.info("Starting facial texture analysis")
            
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
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
            self.logger.info(f"Texture entropy (multi-scale): {entropy1:.2f}/{entropy2:.2f}/{entropy3:.2f}")
            self.logger.info(f"Uniformity ratio: {uniformity_ratio:.4f}, Gradient ratio: {gradient_ratio:.4f}")
            
            # Decision logic
            # 1. Basic entropy check
            entropy_score = (entropy3 > 3.5)  # Real faces have higher entropy
            
            # 2. Simple gradient check
            gradient_score = (gradient_ratio > 1.0)  # Real faces have more complex gradient patterns
            
            # Pass if either test passes (more lenient for web usage)
            is_real_texture = entropy_score or gradient_score
            
            # Calculate overall texture score (0.0 to 1.0)
            texture_score = (entropy3 / 5.0) * 0.6 + (gradient_ratio / 2.0) * 0.4
            texture_score = max(0.0, min(1.0, texture_score))  # Clamp to [0, 1]
            
            self.logger.info(f"Texture analysis results: Entropy={entropy_score}, Gradient={gradient_score}")
            self.logger.info(f"Texture test overall result: {'PASS' if is_real_texture else 'FAIL'}")
            
            # Return results and metrics
            metrics = {
                "entropy1": float(entropy1),
                "entropy2": float(entropy2),
                "entropy3": float(entropy3),
                "uniformity_ratio": float(uniformity_ratio),
                "gradient_ratio": float(gradient_ratio),
                "entropy_score": bool(entropy_score),
                "gradient_score": bool(gradient_score),
                "texture_score": float(texture_score)
            }
            
            return is_real_texture, metrics
            
        except Exception as e:
            self.logger.error(f"Error in texture analysis: {e}")
            self.logger.error(traceback.format_exc())
            # Be lenient in case of errors
            return True, {"error": str(e)}
    
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


class WebCamera:
    """Simplified camera wrapper for face recognition process."""
    
    def __init__(self, camera_id=0, resolution=(640, 480)):
        """Initialize the camera."""
        self.camera_id = camera_id
        self.width, self.height = resolution
        self.cap = None
        self.is_running = False
        self.logger = logging.getLogger("WebCamera")
        
        # Initialize the camera
        self.start()
        
        # Take a few initial frames to "warm up" the camera and activate the light
        self._warm_up_camera()
    
    def _warm_up_camera(self):
        """Warm up the camera by taking a few frames to ensure it's fully active"""
        if not self.is_running:
            return
            
        self.logger.info("Warming up camera...")
        for _ in range(10):  # Take 10 frames to warm up
            ret, frame = self.cap.read()
            time.sleep(0.05)  # Small delay between frames
        self.logger.info("Camera warm-up completed")
    
    def start(self):
        """Start the camera."""
        if self.is_running:
            return
            
        self.logger.info(f"Starting camera (ID: {self.camera_id})")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            self.logger.error("Failed to open camera")
            raise RuntimeError("Failed to open camera")
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Additional camera settings to ensure consistent performance
        self.cap.set(cv2.CAP_PROP_FPS, 30)             # Set target FPS
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)       # Minimize buffer to reduce latency
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)        # Enable autofocus if available
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.7)     # Increase brightness (0.5 -> 0.7)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 0.65)      # Increase contrast slightly (0.5 -> 0.65)
        self.cap.set(cv2.CAP_PROP_SATURATION, 0.6)     # Add saturation setting
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75) # Adjust auto exposure
        
        self.is_running = True
        self.logger.info("Camera started successfully")
    
    def get_frame(self):
        """Get a frame from the camera."""
        if not self.is_running:
            self.start()
            
        # Read multiple frames to ensure we get the latest (clear buffered frames)
        for _ in range(3):
            ret, frame = self.cap.read()
            
        # Final frame read for actual use
        ret, frame = self.cap.read()
        if not ret:
            self.logger.warning("Failed to capture frame")
            return None
            
        return frame
    
    def release(self):
        """Release the camera resources."""
        if self.cap and self.is_running:
            self.logger.info("Releasing camera resources")
            self.cap.release()
            cv2.destroyAllWindows()  # Close any OpenCV windows
            self.is_running = False
            self.logger.info("Camera resources released")


def setup_camera():
    """Setup and initialize camera with multiple attempts to ensure it's active"""
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            logger.info(f"Camera setup attempt {attempt+1}/{max_attempts}")
            
            # Force release of any existing camera resources
            try:
                # Release any previously opened camera
                cap = cv2.VideoCapture(0)
                if cap.isOpened():
                    cap.release()
                    logger.info("Released existing camera")
                    time.sleep(0.5)  # Give some time for the camera to reset
                
                # On Linux, try to reset the camera at system level
                if os.name == 'posix':
                    try:
                        os.system('sudo modprobe -r uvcvideo')
                        time.sleep(1)
                        os.system('sudo modprobe uvcvideo')
                        time.sleep(1)
                        logger.info("Reset camera driver")
                    except Exception as e:
                        logger.warning(f"Failed to reset camera driver: {e}")
            except Exception as e:
                logger.warning(f"Error during camera cleanup: {e}")
            
            # Create a new camera instance
            camera = WebCamera()
            
            # Take a test frame to ensure the camera is working
            test_frame = camera.get_frame()
            if test_frame is None or test_frame.size == 0:
                raise RuntimeError("Camera returned invalid frame")
                
            logger.info("Camera setup successful!")
            return camera
            
        except Exception as e:
            logger.error(f"Error setting up camera (attempt {attempt+1}): {e}")
            time.sleep(1)  # Wait before retrying
    
    # If we got here, we couldn't set up the camera after all attempts
    raise RuntimeError("Failed to set up camera after multiple attempts")

def perform_liveness_check(frame, face_locations):
    """Perform liveness detection check"""
    try:
        detector = LivenessDetector()
        results = []
        
        for face_location in face_locations:
            # Convert from face_recognition format (top, right, bottom, left) to cv2 format (x,y,w,h)
            top, right, bottom, left = face_location
            
            # Perform liveness detection
            face_roi = frame[top:bottom, left:right]
            is_real, score, debug_info = detector.detect_liveness(face_roi)
            
            results.append({
                "is_real": is_real,
                "score": score,
                "face_location": face_location,
                "debug_info": debug_info
            })
            
        return results
    except Exception as e:
        logger.error(f"Error in liveness detection: {e}")
        return []

def load_known_face_encodings(backend_url):
    """Load known face encodings from backend"""
    try:
        response = requests.get(f"{backend_url}/get-all-encodings")
        if response.status_code == 200:
            data = response.json()
            
            # Convert the data format to what we need
            known_face_encodings = []
            known_face_names = []
            known_face_ids = []
            
            for face_data in data:
                # Convert the encoding string back to numpy array
                encoding = np.array(face_data['encoding'])
                known_face_encodings.append(encoding)
                known_face_names.append(face_data['name'])
                known_face_ids.append(face_data['id'])
            
            return known_face_encodings, known_face_names, known_face_ids
        else:
            logger.error(f"Failed to get encodings: {response.status_code}")
            return [], [], []
    except Exception as e:
        logger.error(f"Error loading face encodings: {e}")
        return [], [], []

def calculate_ear(eye_landmarks):
    """
    Calculate Eye Aspect Ratio (EAR) using facial landmarks
    
    Args:
        eye_landmarks: List of (x, y) coordinates for the eye
        
    Returns:
        float: Eye Aspect Ratio value
    """
    # Compute the euclidean distances between the two sets of vertical eye landmarks
    A = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
    B = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
    
    # Compute the euclidean distance between the horizontal eye landmarks
    C = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
    
    # Calculate the EAR
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blink(landmarks, frame, ear_threshold=0.25):
    """
    Detect blinks using facial landmarks and calculate EAR values
    
    Args:
        landmarks: Facial landmarks from face_recognition
        frame: Image containing the face
        ear_threshold: Threshold for detecting a blink
        
    Returns:
        dict: Eye measurements and blink detection data
    """
    if not landmarks:
        return None
    
    # Get the facial landmarks
    facial_landmarks = landmarks[0]
    
    # The facial landmarks for the left and right eyes
    # These indices correspond to the 68-point facial landmark detector in dlib
    # that's used by the face_recognition library
    
    # Left eye indices in face_recognition library format
    left_eye = [
        facial_landmarks['left_eye'][0],  # left corner
        facial_landmarks['left_eye'][1],  # top left
        facial_landmarks['left_eye'][2],  # top right
        facial_landmarks['left_eye'][3],  # right corner
        facial_landmarks['left_eye'][4],  # bottom right
        facial_landmarks['left_eye'][5],  # bottom left
    ]
    
    # Right eye indices in face_recognition library format
    right_eye = [
        facial_landmarks['right_eye'][0],  # left corner
        facial_landmarks['right_eye'][1],  # top left
        facial_landmarks['right_eye'][2],  # top right
        facial_landmarks['right_eye'][3],  # right corner
        facial_landmarks['right_eye'][4],  # bottom right
        facial_landmarks['right_eye'][5],  # bottom left
    ]
    
    # Calculate the EAR for each eye
    left_ear = calculate_ear(left_eye)
    right_ear = calculate_ear(right_eye)
    
    # Average the EAR value for both eyes
    avg_ear = (left_ear + right_ear) / 2.0
    
    # Check if EAR is below the blink threshold
    blink_detected = avg_ear < ear_threshold
    
    # For visualization - draw eye contours in the debug image
    height, width = frame.shape[:2]
    for eye in [left_eye, right_eye]:
        pts = np.array(eye, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (0, 255, 255), 1)
    
    # Return both EAR values and landmarks for visualization
    return {
        'left_ear': float(left_ear),
        'right_ear': float(right_ear),
        'avg_ear': float(avg_ear),
        'blink_detected': bool(blink_detected),
        'left_eye_landmarks': [[float(point[0]), float(point[1])] for point in left_eye],
        'right_eye_landmarks': [[float(point[0]), float(point[1])] for point in right_eye]
    }

def save_debug_frame(frame, face_locations, liveness_results, match_names=None, upload_folder=None):
    """
    Save current debug frame with annotations for visualization
    
    Args:
        frame: The current video frame
        face_locations: List of face location tuples (top, right, bottom, left)
        liveness_results: Results from liveness detection
        match_names: Optional list of matched names for identified faces
        upload_folder: Where to save the debug frame
    
    Returns:
        None
    """
    try:
        if frame is None:
            logger.error("Cannot save debug frame - frame is None")
            return
            
        # Create a copy for visualization
        debug_frame = frame.copy()
        
        # Debug data to save as JSON - ensure all values are JSON serializable
        debug_data = {
            'face_locations': [[int(val) for val in loc] for loc in face_locations],
            'liveness_results': [],
            'match_names': match_names or []
        }
        
        # Convert liveness_results to JSON serializable format
        for result in liveness_results:
            # Convert each result to a serializable dict
            serializable_result = {}
            for key, value in result.items():
                if key == 'face_location':
                    serializable_result[key] = [int(val) for val in value]
                elif key == 'debug_info':
                    # Convert debug_info to serializable format
                    serializable_debug = {}
                    for k, v in value.items():
                        if isinstance(v, (bool, np.bool_)):
                            serializable_debug[k] = bool(v)  # Convert numpy bool to Python bool
                        elif isinstance(v, (int, float, np.number)):
                            serializable_debug[k] = float(v) if isinstance(v, float) else int(v)
                        elif isinstance(v, str):
                            serializable_debug[k] = v
                        else:
                            serializable_debug[k] = str(v)  # Fallback to string for unknown types
                    serializable_result[key] = serializable_debug
                elif isinstance(value, (bool, np.bool_)):
                    serializable_result[key] = bool(value)  # Convert numpy bool to Python bool
                elif isinstance(value, (int, float, np.number)):
                    serializable_result[key] = float(value) if isinstance(value, float) else int(value)
                elif isinstance(value, str):
                    serializable_result[key] = value
                else:
                    serializable_result[key] = str(value)  # Fallback to string for unknown types
            
            debug_data['liveness_results'].append(serializable_result)
        
        ear_values = None
        
        # Draw face boxes and liveness information
        for i, (top, right, bottom, left) in enumerate(face_locations):
            # Draw face bounding box
            face_color = (0, 255, 0)  # Default: green (live face)
            
            # Get liveness result for this face (if available)
            if i < len(liveness_results):
                is_live = liveness_results[i].get("is_real", False)
                if not is_live:
                    face_color = (0, 0, 255)  # Red (fake face)
            
            # Draw face bounding box
            cv2.rectangle(debug_frame, (left, top), (right, bottom), face_color, 2)
            
            # Show liveness score if available
            if i < len(liveness_results):
                is_live = liveness_results[i].get("is_real", False)
                status = "Live" if is_live else "Fake"
                cv2.putText(debug_frame, f"{status}: {liveness_results[i].get('score', 0):.2f}", 
                           (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, face_color, 2)
            
            # Show matched name if available
            if match_names and i < len(match_names) and match_names[i]:
                name = match_names[i]
                cv2.putText(debug_frame, f"{name}", 
                           (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                           0.5, face_color, 2)
            
            # Calculate facial landmarks for EAR
            face_landmarks = []
            try:
                # Extract the face from the frame
                face_img = frame[top:bottom, left:right]
                if face_img.size > 0:  # Make sure the face image is valid
                    # Get landmarks for this face
                    landmarks = face_recognition.face_landmarks(face_img)
                    if landmarks:
                        # Convert relative coordinates to absolute
                        for face_landmark in landmarks:
                            for feature, points in face_landmark.items():
                                for i, (x, y) in enumerate(points):
                                    face_landmark[feature][i] = (x + left, y + top)
                            face_landmarks.append(face_landmark)
                            
                        # Calculate EAR
                        ear_values = detect_blink(face_landmarks, debug_frame)
                        if ear_values:
                            # Ensure the ear_values are JSON serializable
                            serializable_ear = {}
                            for key, value in ear_values.items():
                                if isinstance(value, bool):
                                    serializable_ear[key] = bool(value)
                                elif isinstance(value, (int, float)):
                                    serializable_ear[key] = value
                                elif isinstance(value, list):
                                    # Handle lists of coordinates
                                    serializable_ear[key] = [[float(x), float(y)] for x, y in value]
                                else:
                                    serializable_ear[key] = str(value)
                                    
                            debug_data['ear_values'] = serializable_ear
            except Exception as e:
                logger.error(f"Error calculating EAR: {e}")
        
        # Save the frame with visualizations
        if upload_folder:
            debug_path = os.path.join(upload_folder, 'debug_frame.jpg')
            cv2.imwrite(debug_path, debug_frame)
            logger.info(f"Saved debug frame to {debug_path}")
            
            # Save the debug data as JSON
            debug_data_path = os.path.join(upload_folder, 'debug_data.json')
            with open(debug_data_path, 'w') as f:
                json.dump(debug_data, f)
            logger.info(f"Saved debug data to {debug_data_path}")
                
    except Exception as e:
        logger.error(f"Error saving debug frame: {e}")
        logger.error(traceback.format_exc())

def run_face_recognition(output_file, backend_url, max_attempts=100, confidence_threshold=60, upload_folder=None):
    """
    Main function to run face recognition with liveness detection
    
    Args:
        output_file: Where to save the recognition results
        backend_url: URL for the backend API
        max_attempts: Maximum number of frames to process
        confidence_threshold: Minimum confidence score (0-100) to consider a match valid
        upload_folder: Folder to save debug frames and visualization
        
    Returns:
        dict: Results of the face recognition process
    """
    logger.info("Starting face recognition process")
    
    # Set up camera
    camera = None
    try:
        camera = setup_camera()
        logger.info("Camera setup completed")
        
        # Load known face encodings
        logger.info(f"Loading known face encodings from backend: {backend_url}")
        known_faces, known_names, known_ids = load_known_face_encodings(backend_url)
        
        if not known_faces:
            logger.warning("No known faces loaded from backend")
        else:
            logger.info(f"Loaded {len(known_faces)} known faces")
        
        matches = []  # List to store matches
        attempts = 0  # Counter for attempts
        positive_matches = {}  # Track consistent matches
        best_detections = {}  # Track best detection scores per person
        
        # Variables for liveness check
        liveness_checks_passed = 0
        required_liveness_checks = 3
        frame_count = 0
        
        # Initialize our robust blink detector
        blink_detector = BlinkDetector(ear_threshold=0.25, consecutive_frames=2)
        last_debug_data = None  # Store the last frame's debug data
        
        while attempts < max_attempts:
            attempts += 1
            frame_count += 1
            
            # Get a frame from the camera
            frame = camera.get_frame()
            if frame is None:
                logger.error("Failed to get frame from camera")
                continue
            
            # Convert frame to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            
            if not face_locations:
                logger.info(f"No faces detected in frame {attempts}")
                # Save an empty debug frame to show the current view
                save_debug_frame(frame, [], [], upload_folder=upload_folder)
                continue
            
            # Extract face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Get facial landmarks for blink detection
            face_landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
            
            # Process EAR for blink detection
            current_ear_values = None
            if len(face_landmarks_list) > 0:
                current_ear_values = detect_blink(face_landmarks_list, frame)
                if current_ear_values:
                    # Update the blink detector with the current EAR value
                    avg_ear = current_ear_values.get('avg_ear', 0)
                    blink_status = blink_detector.update(avg_ear)
                    
                    # Update the EAR values with blink status from our detector
                    current_ear_values['blink_detected'] = blink_status['blink_count'] > 0
                    current_ear_values['blink_count'] = blink_status['blink_count']
                    current_ear_values['eye_state'] = blink_status['current_state']
                    
                    # Get text status for display
                    status_info = blink_detector.get_status_text()
                    current_ear_values['status_text'] = status_info['text']
                    
                    # Store for debug frame
                    last_debug_data = {
                        'face_locations': face_locations,
                        'ear_values': current_ear_values,
                        'blink_status': blink_status
                    }
            
            # Perform liveness check on each face
            liveness_results = perform_liveness_check(frame, face_locations)
            
            # Update the debug frame to show real-time processing
            frame_matches = []  # For this frame only
            
            # Process each face in the current frame
            for i, face_encoding in enumerate(face_encodings):
                if i >= len(face_locations):
                    continue  # Skip if index mismatch
                
                # Check if face passes liveness check
                is_live = False
                liveness_score = 0
                if i < len(liveness_results):
                    # Fix: unpack as dictionary instead of tuple
                    result = liveness_results[i]
                    is_live = result.get("is_real", False)
                    liveness_score = result.get("score", 0)
                
                # Consider face live if it passes texture analysis OR if blinking is detected
                natural_blinking = blink_detector.is_natural_blinking()
                is_live = is_live or natural_blinking
                
                # Only process matches for faces that pass liveness check
                face_name = "Unknown"
                confidence = 0
                
                if is_live:
                    # Compare with known faces
                    if known_faces:
                        # Calculate face similarity using face_distance
                        face_distances = face_recognition.face_distance(known_faces, face_encoding)
                        if len(face_distances) > 0:
                            # Find the best match
                            best_match_idx = np.argmin(face_distances)
                            best_match_distance = face_distances[best_match_idx]
                            
                            # Convert distance to similarity score (0-100%)
                            confidence = (1 - best_match_distance) * 100
                            
                            # Check if confidence meets threshold
                            if confidence >= confidence_threshold:
                                face_name = known_names[best_match_idx]
                                logger.info(f"Match found: {face_name}, confidence: {confidence:.2f}%")
                                
                                # Add confidence to results
                                frame_matches.append({
                                    "name": face_name,
                                    "confidence": confidence,
                                    "is_live": is_live
                                })
                
                # Track best detection for each person
                if face_name != "Unknown" and confidence >= confidence_threshold:
                    if face_name not in best_detections or confidence > best_detections[face_name]["confidence"]:
                        best_detections[face_name] = {
                            "confidence": confidence,
                            "encoding": face_encoding.tolist(),
                            "frame": attempts
                        }
                    
                    # Track consistent matches
                    if face_name not in positive_matches:
                        positive_matches[face_name] = 1
                    else:
                        positive_matches[face_name] += 1
                    
                    # Check if we have a reliable match (5+ consistent detections)
                    if positive_matches[face_name] >= 5:
                        matches.append({
                            "name": face_name,
                            "confidence": confidence,
                            "is_live": is_live,
                            "liveness_score": liveness_score
                        })
                        logger.info(f"Reliable match found after {attempts} attempts: {face_name}")
                        
                        # Return early if we have a good match with natural blinking
                        if natural_blinking:
                            logger.info("Face recognized with natural blinking pattern - early return")
                            camera.release()
                            
                            # Save results to output file
                            results = {
                                "success": True,
                                "face_detected": True,
                                "is_live": True,
                                "blink_detected": True,
                                "blink_count": blink_detector.blink_count,
                                "natural_blinking": natural_blinking,
                                "results": matches,
                                "attempts": attempts
                            }
                            
                            # Add the best face encoding if available
                            if face_name in best_detections:
                                results["face_encodings"] = [best_detections[face_name]["encoding"]]
                            
                            with open(output_file, 'w') as f:
                                json.dump(results, f)
                            
                            return results
            
            # Update the debug visualization
            all_match_names = [match.get("name", "Unknown") if i < len(frame_matches) else "Unknown" 
                               for i, _ in enumerate(face_locations)]
            
            # Add blink status text to the frame
            if current_ear_values and blink_detector.blink_count > 0:
                status_info = blink_detector.get_status_text()
                y_pos = 30  # Starting position
                cv2.putText(frame, f"Blinks: {blink_detector.blink_count}", (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_info['color'], 2)
                y_pos += 30
                cv2.putText(frame, status_info['text'], (10, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_info['color'], 2)
            
            # Save the debug frame with all annotations
            save_debug_frame(frame, face_locations, liveness_results, match_names=all_match_names, upload_folder=upload_folder)
            
            # Check if this frame passed liveness check
            if any(result.get("is_real", False) for result in liveness_results):
                liveness_checks_passed += 1
                
                # If we've passed enough liveness checks or have natural blinking, we can consider the detection valid
                if liveness_checks_passed >= required_liveness_checks or natural_blinking:
                    break
            
            # Slow down processing to avoid overloading the system
            time.sleep(0.1)
        
        # Process is complete - release camera
        logger.info(f"Face recognition process completed after {attempts} attempts")
        camera.release()
        
        # Log final blink detection stats
        logger.info(f"Blink count: {blink_detector.blink_count}")
        logger.info(f"Natural blinking detected: {blink_detector.is_natural_blinking()}")
        
        # Determine final result
        face_detected = len(face_locations) > 0
        is_live = liveness_checks_passed >= required_liveness_checks or blink_detector.blink_count > 0
        natural_blinking = blink_detector.is_natural_blinking()
        
        # If no matches but we detected faces and liveness, save the face encoding
        # for potential registration
        if not matches and face_detected and is_live and len(face_encodings) > 0:
            best_face_encoding = face_encodings[0].tolist()
        
        # Write results to output file
            results = {
                "success": True,
                "face_detected": face_detected,
                "is_live": is_live,
                "blink_detected": blink_detector.blink_count > 0,
                "blink_count": blink_detector.blink_count,
                "natural_blinking": natural_blinking,
                "results": [],  # No matches
                "face_encodings": [best_face_encoding]  # Save the encoding for registration
            }
        else:
            # Write results to output file
            results = {
                "success": True,
                "face_detected": face_detected,
                "is_live": is_live,
                "blink_detected": blink_detector.blink_count > 0,
                "blink_count": blink_detector.blink_count,
                "natural_blinking": natural_blinking,
                "results": matches,
                "attempts": attempts
            }
            
            # Add the best face encoding if available
            best_encoding = None
            if matches and matches[0]["name"] in best_detections:
                best_encoding = best_detections[matches[0]["name"]]["encoding"]
                results["face_encodings"] = [best_encoding]
        
        with open(output_file, 'w') as f:
            json.dump(results, f)
        
        return results
            
    except Exception as e:
        logger.error(f"Error during face recognition: {e}")
        logger.error(traceback.format_exc())
        
        # Make sure we always have some result
        results = {
                "success": False,
                "error": str(e),
            "face_detected": False,
            "is_live": False,
            "blink_detected": False,
            "blink_count": 0,
            "natural_blinking": False,
            "results": []
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f)
        
        return results
        
    finally:
        # Clean up resources
        if camera:
            logger.info("Cleaning up camera resources")
            camera.release()
        
        logger.info("Face recognition process finished")

def main():
    """Main entry point for the face recognition process"""
    parser = argparse.ArgumentParser(description='Face Recognition Process')
    parser.add_argument('--output', required=True, help='Output file path for recognition results')
    parser.add_argument('--backend-url', required=False, help='Backend URL for authentication')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds')
    parser.add_argument('--skip-liveness', action='store_true', help='Skip liveness detection (for testing)')
    parser.add_argument('--debug-frames', action='store_true', help='Save debug frames for web display')
    parser.add_argument('--frames-dir', default='/tmp/face_recognition_frames', help='Directory to save debug frames')
    parser.add_argument('--upload-folder', help='Flask upload folder to save debug frame for web display')
    
    args = parser.parse_args()
    
    # Setup frames directory if debug mode is enabled
    frames_dir = None
    if args.debug_frames:
        frames_dir = args.frames_dir
        os.makedirs(frames_dir, exist_ok=True)
        # Clean up old frames
        for old_file in os.listdir(frames_dir):
            try:
                os.remove(os.path.join(frames_dir, old_file))
            except:
                pass
        logger.info(f"Debug mode enabled, frames will be saved to {frames_dir}")
        
    # Ensure upload folder exists if provided
    upload_folder = None
    if args.upload_folder:
        upload_folder = args.upload_folder
        os.makedirs(upload_folder, exist_ok=True)
        logger.info(f"Web debug enabled, frames will be saved to {upload_folder}")
    
    # Run the face recognition process    
    logger.info(f"Starting face recognition process. Output will be saved to {args.output}")
    
    # Run face recognition directly
    success = run_face_recognition(
        args.output,
        args.backend_url,
        max_attempts=int(args.timeout * 2),  # Set max attempts based on timeout
        upload_folder=upload_folder
    )
    
    return 0 if success else 1

class BlinkDetector:
    """Robust blink detector using state machine approach to track blink patterns"""
    
    def __init__(self, ear_threshold=0.25, consecutive_frames=2):
        """
        Initialize the blink detector
        
        Args:
            ear_threshold: Threshold below which an eye is considered closed
            consecutive_frames: Number of consecutive frames required to confirm state change
        """
        self.EAR_THRESHOLD = ear_threshold
        self.CONSECUTIVE_FRAMES = consecutive_frames
        
        # Initialize state machine
        self.current_state = "OPEN"  # OPEN, CLOSING, CLOSED, OPENING
        self.consecutive_below_threshold = 0
        self.consecutive_above_threshold = 0
        self.blink_count = 0
        self.ear_history = []
        self.max_history = 30  # Store last 30 frames of EAR values
        
        # For tracking blink speed and patterns
        self.last_blink_time = None
        self.blink_intervals = []
        
        self.logger = logging.getLogger("BlinkDetector")
    
    def update(self, ear_value):
        """
        Update the blink detector with a new EAR value
        
        Args:
            ear_value: Current frame's Eye Aspect Ratio
            
        Returns:
            dict: Current blink detection state
        """
        # Add to history
        self.ear_history.append(ear_value)
        if len(self.ear_history) > self.max_history:
            self.ear_history = self.ear_history[-self.max_history:]
        
        # Update state machine
        if self.current_state == "OPEN":
            if ear_value < self.EAR_THRESHOLD:
                self.consecutive_below_threshold += 1
                if self.consecutive_below_threshold >= self.CONSECUTIVE_FRAMES:
                    self.current_state = "CLOSING"
                    self.consecutive_below_threshold = 0
                    self.logger.info("Eye state: CLOSING")
            else:
                self.consecutive_below_threshold = 0
                
        elif self.current_state == "CLOSING":
            if ear_value < self.EAR_THRESHOLD:
                self.consecutive_below_threshold += 1
                if self.consecutive_below_threshold >= self.CONSECUTIVE_FRAMES:
                    self.current_state = "CLOSED"
                    self.consecutive_below_threshold = 0
                    self.logger.info("Eye state: CLOSED")
            else:
                self.consecutive_below_threshold = 0
                self.consecutive_above_threshold += 1
                if self.consecutive_above_threshold >= self.CONSECUTIVE_FRAMES:
                    # False alarm, eye didn't actually close
                    self.current_state = "OPEN"
                    self.consecutive_above_threshold = 0
                    self.logger.info("Eye state: Back to OPEN (false alarm)")
                
        elif self.current_state == "CLOSED":
            if ear_value >= self.EAR_THRESHOLD:
                self.consecutive_above_threshold += 1
                if self.consecutive_above_threshold >= self.CONSECUTIVE_FRAMES:
                    self.current_state = "OPENING"
                    self.consecutive_above_threshold = 0
                    self.logger.info("Eye state: OPENING")
            else:
                self.consecutive_above_threshold = 0
                
        elif self.current_state == "OPENING":
            if ear_value >= self.EAR_THRESHOLD:
                self.consecutive_above_threshold += 1
                if self.consecutive_above_threshold >= self.CONSECUTIVE_FRAMES:
                    # Complete blink detected
                    self.current_state = "OPEN"
                    self.consecutive_above_threshold = 0
                    self.blink_count += 1
                    
                    # Record blink timing
                    current_time = time.time()
                    if self.last_blink_time is not None:
                        interval = current_time - self.last_blink_time
                        self.blink_intervals.append(interval)
                        # Keep only recent intervals
                        if len(self.blink_intervals) > 5:
                            self.blink_intervals = self.blink_intervals[-5:]
                    
                    self.last_blink_time = current_time
                    self.logger.info(f"BLINK DETECTED! Count: {self.blink_count}")
            else:
                self.consecutive_above_threshold = 0
                self.consecutive_below_threshold += 1
                if self.consecutive_below_threshold >= self.CONSECUTIVE_FRAMES:
                    # Eye closed again without completing the blink
                    self.current_state = "CLOSED"
                    self.consecutive_below_threshold = 0
                    self.logger.info("Eye state: Back to CLOSED")
        
        # Calculate blink rate (blinks per minute)
        blink_rate = 0
        if self.blink_intervals:
            # Calculate average interval
            avg_interval = sum(self.blink_intervals) / len(self.blink_intervals)
            if avg_interval > 0:
                blink_rate = 60.0 / avg_interval
        
        # Prepare result
        is_blinking = self.current_state in ["CLOSING", "CLOSED", "OPENING"]
        
        return {
            "current_state": self.current_state,
            "is_blinking": is_blinking,
            "blink_count": self.blink_count,
            "blink_rate": blink_rate,
            "current_ear": ear_value
        }
    
    def is_natural_blinking(self):
        """
        Check if the observed blinking pattern appears natural
        
        Returns:
            bool: True if blinking pattern seems natural
        """
        # Need at least 2 blinks to analyze pattern
        if self.blink_count < 2:
            return False
            
        if not self.blink_intervals:
            return False
            
        # Check for consistent but not too regular intervals
        # Natural blinking is typically 2-5 seconds apart but varies
        avg_interval = sum(self.blink_intervals) / len(self.blink_intervals)
        
        # Too fast or too slow is suspicious
        if avg_interval < 1.0 or avg_interval > 8.0:
            return False
            
        # Check for variation in blink intervals (standard deviation)
        if len(self.blink_intervals) >= 3:
            std_dev = np.std(self.blink_intervals)
            variation = std_dev / avg_interval
            
            # Too regular is suspicious (robots blink at exact intervals)
            if variation < 0.1:
                return False
                
        # If we have multiple blinks with reasonable timing, it's probably natural
        return self.blink_count >= 2
    
    def get_status_text(self):
        """
        Get status text for display
        
        Returns:
            dict: Status text information
        """
        status = ""
        color = (255, 255, 255)  # Default white
        
        if self.blink_count == 0:
            status = "No blinks detected"
            color = (0, 0, 255)  # Red
        elif self.blink_count == 1:
            status = "Blinked once"
            color = (0, 255, 255)  # Yellow
        elif self.is_natural_blinking():
            status = f"Natural blinking ({self.blink_count})"
            color = (0, 255, 0)  # Green
        else:
            status = f"Blinked {self.blink_count} times"
            color = (255, 255, 0)  # Light blue
            
        return {
            "text": status,
            "color": color,
            "blink_count": self.blink_count,
            "is_natural": self.is_natural_blinking()
        }

if __name__ == "__main__":
    sys.exit(main()) 