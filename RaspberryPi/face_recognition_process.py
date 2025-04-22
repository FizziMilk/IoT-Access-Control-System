#!/usr/bin/env python3
"""
Standalone script for facial recognition and liveness detection.
"""
import os
import sys
import time
import json
import logging
import cv2
import numpy as np
import face_recognition
import base64
import requests
from datetime import datetime
from camera_config import CAMERA_INDEX, MIN_FACE_WIDTH, MIN_FACE_HEIGHT
import pickle

# Configure logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/tmp/face_recognition_process.log')
    ]
)

logger = logging.getLogger("FaceRecognitionProcess")

class LivenessDetector:
    """Enhanced liveness detection for face recognition with anti-spoofing measures"""
    
    def __init__(self):
        # Default thresholds - can be tuned based on environment
        self.entropy_threshold = 3.5  # Reduced from 4.0
        self.gradient_threshold = 10.0  # Reduced from 15.0
        self.reflection_threshold = 0.05  # Critical for distinguishing phones - real faces have very low values (<0.001)
        self.color_threshold = 15.0  # Reduced from 18.0
        self.lbp_uniformity_threshold = 0.4  # Reduced from 0.6
        self.frequency_threshold = 20.0  # Reduced from 25.0
        self.gradient_variance_threshold = 15.0  # Reduced from 20.0
        self.micro_texture_threshold = 0.15  # Reduced from 0.2
        # Maximum gradient and gradient variance for real faces
        self.max_gradient_threshold = 35.0  # Real faces typically below this
        self.max_gradient_variance = 3000.0  # Real faces typically below this
    
    def check_liveness(self, frame):
        """
        Enhanced check if a face appears to be from a live person with anti-spoofing
        
        Args:
            frame: OpenCV BGR image containing a face
            
        Returns:
            dict: Liveness metrics and result
        """
        # Validate input frame
        if frame is None or frame.size == 0:
            return {"is_live": False, "error": "Invalid frame"}
        
        # Ensure frame is large enough for analysis
        if frame.shape[0] < 32 or frame.shape[1] < 32:
            return {"is_live": False, "error": "Frame too small"}
            
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate LBP for texture analysis
        lbp_image = self._calculate_lbp(gray)
        
        # Calculate texture entropy from LBP
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=[0, 256])
        hist = hist / float(hist.sum() + 1e-10)  # Avoid division by zero
        texture_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calculate gradients with Sobel operator
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(gradient_magnitude)
        
        # ENHANCEMENT 1: Calculate gradient variance (printed photos have more uniform gradients)
        gradient_variance = np.var(gradient_magnitude)
        
        # ENHANCEMENT 2: Calculate gradient direction (printed photos have more uniform direction)
        gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        direction_hist, _ = np.histogram(gradient_direction.ravel(), bins=18, range=[-180, 180])
        direction_hist = direction_hist / float(direction_hist.sum() + 1e-10)
        direction_entropy = -np.sum(direction_hist * np.log2(direction_hist + 1e-10))
        
        # Detect reflections (phone screens have much higher reflection values)
        _, bright_spots = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_spot_count = cv2.countNonZero(bright_spots)
        bright_spot_ratio = bright_spot_count / (gray.shape[0] * gray.shape[1])
        
        # Analyze color variance
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:,:,1])  # Saturation channel
        
        # Convert any arrays to scalars to ensure safe boolean operations
        texture_entropy_scalar = float(texture_entropy.item() if hasattr(texture_entropy, 'item') else texture_entropy)
        mean_gradient_scalar = float(mean_gradient.item() if hasattr(mean_gradient, 'item') else mean_gradient)
        gradient_variance_scalar = float(gradient_variance.item() if hasattr(gradient_variance, 'item') else gradient_variance)
        direction_entropy_scalar = float(direction_entropy.item() if hasattr(direction_entropy, 'item') else direction_entropy)
        bright_spot_ratio_scalar = float(bright_spot_ratio.item() if hasattr(bright_spot_ratio, 'item') else bright_spot_ratio)
        color_std_scalar = float(color_std.item() if hasattr(color_std, 'item') else color_std)
        
        # Evaluate basic liveness checks
        entropy_ok = texture_entropy_scalar > self.entropy_threshold
        gradient_ok = mean_gradient_scalar > self.gradient_threshold
        reflection_ok = bright_spot_ratio_scalar < self.reflection_threshold  # Critical check
        color_ok = color_std_scalar > self.color_threshold
        
        # Enhanced checks - phone screens show unusually high values
        gradient_normal = mean_gradient_scalar < self.max_gradient_threshold
        variance_normal = gradient_variance_scalar < self.max_gradient_variance
        
        # Calculate confidence score for basic checks (0-1)
        basic_checks = [
            entropy_ok, gradient_ok, color_ok
        ]
        
        # CRITICAL: Make reflection check mandatory and add upper bounds checks
        is_live = (reflection_ok and                # Must pass reflection check
                  sum(1 for c in basic_checks if c) >= 2 and  # Pass at least 2 other basic checks
                  (gradient_normal or variance_normal))       # Gradient not abnormally high
        
        # Calculate overall confidence
        confidence_score = (sum(1 for c in basic_checks if c) / len(basic_checks) * 0.4 + 
                           int(reflection_ok) * 0.4 +          # Reflection gets highest weight
                           (int(gradient_normal) + int(variance_normal)) / 2 * 0.2)
        
        # Log metrics
        logger.info(f"Liveness: entropy={texture_entropy_scalar:.2f}, gradient={mean_gradient_scalar:.2f}, "
                   f"grad_var={gradient_variance_scalar:.2f}, dir_entropy={direction_entropy_scalar:.2f}, "
                   f"reflection={bright_spot_ratio_scalar:.4f}, color={color_std_scalar:.2f}, "
                   f"grad_normal={gradient_normal}, var_normal={variance_normal}, "
                   f"confidence={confidence_score:.2f}, live={is_live}")
        
        return {
            "texture_entropy": texture_entropy_scalar,
            "mean_gradient": mean_gradient_scalar,
            "gradient_variance": gradient_variance_scalar,
            "direction_entropy": direction_entropy_scalar,
            "bright_spot_ratio": bright_spot_ratio_scalar,
            "color_std": color_std_scalar,
            "confidence_score": confidence_score,
            "is_live": bool(is_live)
        }
    
    def _calculate_lbp(self, img, radius=1, num_points=8):
        """
        Calculate Local Binary Pattern for texture analysis
        
        Args:
            img: Grayscale image
            radius: Radius around the central pixel
            num_points: Number of points around the center pixel
            
        Returns:
            numpy.ndarray: LBP image
        """
        # Use float32 to avoid uint8 overflow
        lbp = np.zeros_like(img, dtype=np.uint16)
        for i in range(radius, img.shape[0] - radius):
            for j in range(radius, img.shape[1] - radius):
                center = img[i, j]
                binary_pattern = 0
                
                # Sample points around center pixel
                for k in range(num_points):
                    angle = 2 * np.pi * k / num_points
                    x = j + radius * np.cos(angle)
                    y = i - radius * np.sin(angle)
                    
                    # Get nearest pixel
                    x_floor, y_floor = int(np.floor(x)), int(np.floor(y))
                    
                    if 0 <= x_floor < img.shape[1] and 0 <= y_floor < img.shape[0]:
                        value = img[y_floor, x_floor]
                        if value >= center:
                            # Use modulo to prevent overflow if num_points is large
                            if k < 16:  # Ensure we don't exceed 16 bits
                                binary_pattern |= (1 << k)
                
                lbp[i, j] = binary_pattern
        
        return lbp
        
    def check_face_liveness(self, frame, face_location=None):
        """
        Check liveness for a specific face in the frame
        
        Args:
            frame: OpenCV BGR image
            face_location: Optional face location tuple (top, right, bottom, left)
            
        Returns:
            dict: Liveness result
        """
        if face_location is None:
            return self.check_liveness(frame)
            
        # Extract face region
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]
        
        # Ensure face region is valid
        if face_img.size == 0:
            logger.warning("Invalid face region for liveness check")
            return {"is_live": False, "error": "Invalid face region"}
        
        return self.check_liveness(face_img)
        
    def check_multiple_faces(self, frame, face_locations):
        """
        Perform liveness detection for all faces in frame
        
        Args:
            frame: OpenCV BGR image
            face_locations: List of face location tuples
            
        Returns:
            list: Liveness check results for each face
        """
        results = []
        
        for face_loc in face_locations:
            try:
                result = self.check_face_liveness(frame, face_loc)
                
                # Ensure all values are scalar for safe boolean operations
                sanitized_result = {}
                for key, value in result.items():
                    if isinstance(value, np.ndarray):
                        if value.size == 1:
                            sanitized_result[key] = value.item()
                        else:
                            sanitized_result[key] = float(value.mean())
                    else:
                        sanitized_result[key] = value
                
                # Make sure is_live is a bool
                if "is_live" in sanitized_result:
                    sanitized_result["is_live"] = bool(sanitized_result["is_live"])
                
                results.append(sanitized_result)
            except Exception as e:
                logger.error(f"Error in liveness check: {e}")
                results.append({"is_live": False, "error": str(e)})
        
        return results


class WebRecognition:
    """Face recognition system for web applications"""
    
    def __init__(self):
        """Initialize the recognition system"""
        logger.info("Initializing WebRecognition")
        self.known_face_encodings = []
        self.known_face_names = []
        self.detection_threshold = 0.6  # Lower values are more strict
        self.liveness_detector = LivenessDetector()
        
    def load_encodings(self, encodings, names):
        """Load face encodings and corresponding names"""
        if len(encodings) != len(names):
            logger.error(f"Mismatch between encodings ({len(encodings)}) and names ({len(names)})")
            return False
            
        # Convert all encodings to numpy arrays
        self.known_face_encodings = [np.array(e) if isinstance(e, list) else e for e in encodings]
        self.known_face_names = names
        logger.info(f"Loaded {len(encodings)} encodings with names")
        return True
        
    def identify_face(self, frame=None, face_location=None, face_encoding=None):
        """
        Identify a face in the given frame or using provided encoding
        
        Args:
            frame: Optional OpenCV BGR image
            face_location: Optional face location tuple (top, right, bottom, left)
            face_encoding: Optional face encoding directly provided
            
        Returns:
            dict or None: Match information if found, None otherwise
        """
        # If face_encoding is provided directly, use it
        if face_encoding is not None:
            # Convert to numpy array if it's a list
            if isinstance(face_encoding, list):
                face_encoding = np.array(face_encoding)
        elif frame is not None:
            # Convert to RGB (face_recognition uses RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # If face location not provided, detect faces
            if face_location is None:
                face_locations = face_recognition.face_locations(rgb_frame)
                if not face_locations:
                    return None
                face_location = face_locations[0]  # Use first detected face
            else:
                face_locations = [face_location]
                
            # Get face encodings
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            if not face_encodings:
                logger.warning("Could not encode detected face")
                return None
                
            face_encoding = face_encodings[0]
        else:
            # Need either frame or face_encoding
            logger.error("Either frame or face_encoding must be provided")
            return None
        
        # No known faces to compare against
        if not self.known_face_encodings:
            logger.warning("No known face encodings to match against")
            return {
                "encoding": face_encoding,
                "location": face_location if 'face_location' in locals() else None,
                "match": None
            }
        
        # Ensure all encodings are numpy arrays
        if isinstance(face_encoding, list):
            face_encoding = np.array(face_encoding)
        
        # Group encodings by person (based on name)
        # This allows us to compare against all encodings for each person
        person_encodings = {}
        for i, name in enumerate(self.known_face_names):
            if name not in person_encodings:
                person_encodings[name] = []
            person_encodings[name].append(self.known_face_encodings[i])
        
        # For each person, calculate the average distance across all their encodings
        person_distances = {}
        for name, encodings in person_encodings.items():
            distances = []
            for encoding in encodings:
                # Make sure encoding is a numpy array
                if isinstance(encoding, list):
                    encoding = np.array(encoding)
                # Calculate face distance for each encoding
                try:
                    distance = face_recognition.face_distance([encoding], face_encoding)[0]
                    # Make sure we have a scalar value, not an array
                    if isinstance(distance, np.ndarray):
                        distance = float(distance.item()) if distance.size == 1 else float(distance.mean())
                    distances.append(distance)
                except Exception as e:
                    logger.error(f"Error calculating face distance: {e}")
                    continue
            
            # Calculate the average distance for this person
            # Use a weighted average that emphasizes the best matching encoding
            if distances:
                # Sort distances from best (lowest) to worst
                sorted_distances = sorted(distances)
                # Take weighted average - give more weight to better matches
                # This rewards having at least one very good match
                weights = [max(0.5, 1.0 - 0.1*i) for i in range(len(sorted_distances))]
                total_weight = sum(weights)
                avg_distance = sum(d * w for d, w in zip(sorted_distances, weights)) / total_weight
                # Also track the best single match for this person
                best_distance = min(distances)
                person_distances[name] = {
                    'avg_distance': avg_distance,
                    'best_distance': best_distance,
                    'num_encodings': len(encodings)
                }
        
        # Find the best matching person
        if person_distances:
            # Sort by average distance
            sorted_people = sorted(person_distances.items(), key=lambda x: x[1]['avg_distance'])
            best_match_name, best_match_info = sorted_people[0]
            best_match_distance = best_match_info['avg_distance']
            best_single_distance = best_match_info['best_distance']
            
            result = {
                "encoding": face_encoding,
                "location": face_location if 'face_location' in locals() else None,
                "match": None
            }
            
            # Check if the match is close enough
            # Use .all() to ensure all elements in the array meet the threshold criteria
            if isinstance(best_match_distance, np.ndarray):
                is_close_enough = (best_match_distance <= self.detection_threshold).all()
            else:
                is_close_enough = best_match_distance <= self.detection_threshold
                
            if is_close_enough:
                match_name = best_match_name
                # Convert distance to confidence (0 distance = 100% confidence, 1 distance = 0% confidence)
                if isinstance(best_match_distance, np.ndarray):
                    # If it's an array, take the mean for confidence
                    match_confidence = 1.0 - np.mean(best_match_distance)
                    match_confidence_scalar = float(match_confidence.item() if isinstance(match_confidence, np.ndarray) and match_confidence.size == 1 else match_confidence)
                    best_match_distance_scalar = float(np.mean(best_match_distance))
                else:
                    match_confidence_scalar = 1.0 - best_match_distance
                    best_match_distance_scalar = float(best_match_distance)
                
                logger.info(f"Face matched with {match_name} (confidence: {match_confidence_scalar:.2f}, distance: {best_match_distance_scalar:.2f})")
                
                result["match"] = {
                    "name": match_name,
                    "confidence": match_confidence_scalar,
                    "distance": best_match_distance_scalar
                }
            else:
                logger.info(f"Best match too far ({best_match_distance:.2f} > {self.detection_threshold:.2f})")
            
            return result
        else:
            # Fallback to original logic - should never reach here if there are known encodings
            logger.warning("No person encodings available despite having known face encodings")
            try:
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                # Ensure face_distances is a list of scalar values
                if isinstance(face_distances, np.ndarray):
                    face_distances = [float(d) if not isinstance(d, np.ndarray) else float(d.mean()) for d in face_distances]
            except Exception as e:
                logger.error(f"Error in fallback face distance calculation: {e}")
                face_distances = []
            
            result = {
                "encoding": face_encoding,
                "location": face_location if 'face_location' in locals() else None,
                "match": None
            }
            
            if len(face_distances) > 0:
                # Find best match
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]
                
                # Check if the match is close enough (lower distance means better match)
                # face_distance is a measure of difference, with 0 being a perfect match
                # and values closer to 1 being poor matches
                if isinstance(best_match_distance, np.ndarray):
                    is_close_enough = (best_match_distance <= self.detection_threshold).all()
                else:
                    is_close_enough = best_match_distance <= self.detection_threshold
                    
                if is_close_enough:
                    match_name = self.known_face_names[best_match_index]
                    # Convert distance to confidence (0 distance = 100% confidence, 1 distance = 0% confidence)
                    if isinstance(best_match_distance, np.ndarray):
                        # If it's an array, take the mean for confidence
                        match_confidence = 1.0 - np.mean(best_match_distance)
                        match_confidence_scalar = float(match_confidence.item() if isinstance(match_confidence, np.ndarray) and match_confidence.size == 1 else match_confidence)
                        best_match_distance_scalar = float(np.mean(best_match_distance))
                    else:
                        match_confidence_scalar = 1.0 - best_match_distance
                        best_match_distance_scalar = float(best_match_distance)
                    
                    logger.info(f"Face matched with {match_name} (confidence: {match_confidence_scalar:.2f}, distance: {best_match_distance_scalar:.2f})")
                    
                    result["match"] = {
                        "name": match_name,
                        "confidence": match_confidence_scalar,
                        "distance": best_match_distance_scalar
                    }
                else:
                    logger.info(f"Best match too far ({best_match_distance:.2f} > {self.detection_threshold:.2f})")
            
            return result
    
    def check_liveness(self, frame, face_location=None):
        """Check if a face is live"""
        result = self.liveness_detector.check_face_liveness(frame, face_location)
        
        # Ensure all values in the result are scalar, not NumPy arrays
        sanitized_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                if value.size == 1:
                    sanitized_result[key] = value.item()
                else:
                    sanitized_result[key] = float(value.mean())
            else:
                sanitized_result[key] = value
                
        # Ensure is_live is a boolean
        if "is_live" in sanitized_result:
            sanitized_result["is_live"] = bool(sanitized_result["is_live"])
            
        return sanitized_result


def save_debug_frame(frame, filename, faces=None, liveness_results=None, matches=None):
    """
    Save a debug frame with detection visualization
    
    Args:
        frame: OpenCV BGR image
        filename: Output filename
        faces: List of face locations (top, right, bottom, left)
        liveness_results: Liveness check results
        matches: Match information
    """
    # Make a copy to avoid modifying the original
    debug_frame = frame.copy()
    
    # Draw face boxes if provided
    if faces:
        for i, face_loc in enumerate(faces):
            top, right, bottom, left = face_loc
            
            # Default color is red (unknown)
            color = (0, 0, 255)
            
            # If we have match info
            match_text = "Unknown"
            if matches and i < len(matches) and matches[i]:
                match = matches[i].get("match")
                if match:
                    # Green for match
                    color = (0, 255, 0)
                    confidence = match.get("confidence", 0) * 100
                    match_text = f"{match.get('name')} ({confidence:.1f}%)"
            
            # If we have liveness info
            is_live = "?"
            if liveness_results and i < len(liveness_results):
                result = liveness_results[i]
                # Handle is_live which might be a NumPy array
                is_live_value = result.get("is_live", False)
                if isinstance(is_live_value, np.ndarray):
                    # Use .any() for array boolean evaluation
                    is_live_value = is_live_value.any()
                        
                if is_live_value:
                    is_live = "Live"
                else:
                    is_live = "Not Live"
                    # Change color to blue for detected but not live
                    if color == (0, 0, 255):
                        color = (255, 0, 0)
            
            # Draw rectangle around face
            cv2.rectangle(debug_frame, (left, top), (right, bottom), color, 2)
            
            # Draw labels
            cv2.putText(debug_frame, match_text, (left, top - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.putText(debug_frame, is_live, (left, bottom + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Remove timestamp overlay
    # timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # cv2.putText(debug_frame, timestamp, (10, 30), 
    #            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the image
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(filename, debug_frame)
    logger.info(f"Saved debug frame to {filename}")


def load_known_faces(backend_url):
    """
    Load known face encodings from backend
    
    Args:
        backend_url: Backend API URL
        
    Returns:
        tuple: (encodings, names) lists
    """
    try:
        if not backend_url:
            logger.warning("No backend URL provided, skipping face loading")
            return [], []
            
        # Create a session for API calls
        session = requests.Session()
        
        # Request face data from backend
        response = session.get(f"{backend_url}/get-face-data")
        
        if response.status_code != 200:
            logger.warning(f"Backend returned non-200 status: {response.status_code}")
            return [], []
            
        data = response.json()
        
        if data.get("status") != "success":
            logger.warning(f"Backend returned error status: {data.get('status')}")
            return [], []
            
        face_data = data.get("face_data", [])
        logger.info(f"Retrieved {len(face_data)} faces from backend")
        
        # Process face data
        encodings = []
        names = []
        users = []
        
        for entry in face_data:
            phone_number = entry.get("phone_number")
            face_encoding_b64 = entry.get("face_encoding")
            
            if not phone_number or not face_encoding_b64:
                logger.warning("Skipping entry with missing data")
                continue
                
            # Decode the face encoding
            try:
                # First try the direct JSON decode approach
                try:
                    # Decode base64 to JSON string then to list
                    decoded_data = base64.b64decode(face_encoding_b64)
                    encoding_json = decoded_data.decode('utf-8')
                    encoding_list = json.loads(encoding_json)
                    encoding = np.array(encoding_list)
                except Exception:
                    # If that fails, try the alternative approach where encoding is already JSON
                    try:
                        encoding_list = json.loads(face_encoding_b64)
                        encoding = np.array(encoding_list)
                    except Exception:
                        # Last resort - try directly decoding as a pickle
                        encoding = pickle.loads(base64.b64decode(face_encoding_b64))
                
                if isinstance(encoding, np.ndarray) and encoding.size == 128:  # Standard face encoding length
                    logger.info(f"Decoded face encoding for {phone_number} (shape: {encoding.shape})")
                    encodings.append(encoding)
                    # Try to get user information from the database
                    try:
                        user_response = requests.get(f"{backend_url}/users?phone_number={phone_number}")
                        if user_response.status_code == 200:
                            user_data = user_response.json()
                            if user_data and isinstance(user_data, list) and len(user_data) > 0:
                                user = user_data[0]
                                user_name = user.get("name", "Unknown")
                                if user_name and user_name.strip():
                                    names.append(user_name)
                                    users.append(phone_number)
                                    logger.info(f"Found user name {user_name} for {phone_number}")
                                    continue
                    except Exception as e:
                        logger.error(f"Error fetching user name: {e}")
                    
                    # Fallback to using phone number as name
                    names.append(phone_number)
                    users.append(phone_number)
                else:
                    logger.warning(f"Invalid encoding shape for {phone_number}: {encoding.shape if hasattr(encoding, 'shape') else type(encoding)}")
            except Exception as e:
                logger.error(f"Error decoding face encoding for {phone_number}: {e}")
                continue
        
        logger.info(f"Loaded {len(encodings)} face encodings with names: {names}")
        return encodings, names
            
    except Exception as e:
        logger.error(f"Error loading faces from backend: {e}", exc_info=True)
        return [], []


def run_face_recognition(camera_index=0, backend_url=None, skip_liveness=False, output_file=None, debug_dir=None):
    """
    Main function to run face recognition process
    """
    try:
        # Initialize variables
        result = {
            "success": False,
            "face_detected": False,
            "face_recognized": False,
            "liveness_check_passed": False,
            "face_too_small": False,  # New flag to indicate if face is too small
            "distance_feedback": None  # New field for distance feedback
        }
        camera = None
        
        # Use camera index from argument or default from config
        if camera_index == 0:
            camera_index = CAMERA_INDEX
        
        # Initialize WebRecognition class
        recognition = WebRecognition()
        
        # Try to load known faces from backend
        if backend_url:
            encodings, names = load_known_faces(backend_url)
            if encodings and names:
                recognition.load_encodings(encodings, names)
            else:
                logger.warning("No face encodings loaded from backend")
        
        # Initialize camera with multiple attempts
        for attempt in range(1, 6):  # Try up to 5 times
            logger.info(f"Opening camera at index {camera_index} (attempt {attempt}/5)")
            try:
                camera = cv2.VideoCapture(camera_index)
                if camera.isOpened():
                    logger.info(f"Successfully opened camera at index {camera_index}")
                    break
            except Exception as e:
                logger.error(f"Error opening camera on attempt {attempt}: {e}")
            
            # Wait before retrying
            time.sleep(1)
        
        # If camera still not opened after all attempts, return error
        if not camera or not camera.isOpened():
            logger.error("Failed to open camera after multiple attempts")
            result["error"] = "Failed to open camera"
            return result
        
        # Create debug directory if specified
        if debug_dir and not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        
        # Read frame from camera
        ret, frame = camera.read()
        if not ret or frame is None:
            logger.error("Failed to capture frame from camera")
            result["error"] = "Failed to capture frame"
            return result
        
        # Save initial debug frame
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if debug_dir:
            save_debug_frame(frame, f"{debug_dir}/frame_initial_{timestamp}.jpg")
        
        # Detect faces in the frame
        # Use a higher upsample value to detect smaller/more distant faces
        face_locations = face_recognition.face_locations(frame, model="hog", number_of_times_to_upsample=2)
        
        if not face_locations:
            logger.warning("No faces detected in frame")
            if debug_dir:
                # Add text to the frame indicating no face detected
                cv2.putText(
                    frame, 
                    "No face detected - Please move closer to the camera", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
                save_debug_frame(frame, f"{debug_dir}/frame_nofaces_{timestamp}.jpg")
            result["error"] = "No faces detected"
            return result
        
        # Check if the detected face is large enough
        face_location = face_locations[0]  # Take the first face
        top, right, bottom, left = face_location
        face_width = right - left
        face_height = bottom - top
        
        logger.info(f"Detected face size: {face_width}x{face_height} pixels")
        
        # If face is too small, return feedback
        if face_width < MIN_FACE_WIDTH or face_height < MIN_FACE_HEIGHT:
            logger.warning(f"Detected face is too small for reliable recognition ({face_width}x{face_height})")
            result["face_detected"] = True
            result["face_too_small"] = True
            
            if face_width < MIN_FACE_WIDTH * 0.5 or face_height < MIN_FACE_HEIGHT * 0.5:
                distance_feedback = "much_too_far"
            else:
                distance_feedback = "too_far"
                
            result["distance_feedback"] = distance_feedback
            
            if debug_dir:
                # Add text to the frame indicating face is too small
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                cv2.putText(
                    frame, 
                    "Face too small - Please move closer", 
                    (left, top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.7, 
                    (0, 0, 255), 
                    2
                )
                save_debug_frame(frame, f"{debug_dir}/frame_face_too_small_{timestamp}.jpg", 
                                faces=face_locations)
            return result
        
        # Extract face encodings
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        if not face_encodings:
            logger.warning("Failed to extract face encodings")
            if debug_dir:
                save_debug_frame(frame, f"{debug_dir}/frame_nofaces_{timestamp}.jpg")
            result["error"] = "Failed to extract face encodings"
            return result
        
        # Process detected faces
        result["face_detected"] = True
        
        # Save debug frame with detected faces
        if debug_dir:
            save_debug_frame(frame, f"{debug_dir}/frame_faces_{timestamp}.jpg", 
                            faces=face_locations)
        
        # Perform liveness check if not skipped
        liveness_results = None
        if not skip_liveness:
            liveness_results = recognition.liveness_detector.check_multiple_faces(frame, face_locations)
            
            # Check if any face passes liveness
            all_fake = all(not result.get("is_live", False) for result in liveness_results)
            
            if all_fake:
                logger.warning("All detected faces failed liveness check")
                
                # Save debug frame with liveness failures
                if debug_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_debug_frame(frame, f"{debug_dir}/frame_liveness_fail_{timestamp}.jpg", 
                                    faces=face_locations, liveness_results=liveness_results)
                
                result["liveness_check_passed"] = False
                result["is_live"] = False
                result["liveness_results"] = liveness_results
            else:
                result["liveness_check_passed"] = True
                result["is_live"] = any(r.get("is_live", False) for r in liveness_results)
                result["liveness_results"] = liveness_results
        
        # Identify faces
        matches = []
        for face_encoding in face_encodings:
            # Use the identify_face method from WebRecognition
            match_result = recognition.identify_face(face_encoding=face_encoding)
            if match_result and "match" in match_result:
                matches.append({"match": match_result["match"]})
            else:
                matches.append({"match": None})
        
        # Save final debug frame with all information
        if debug_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_debug_frame(frame, f"{debug_dir}/frame_final_{timestamp}.jpg", 
                           faces=face_locations, liveness_results=liveness_results, 
                           matches=matches)
        
        # Make sure NumPy arrays are converted to lists for JSON serialization
        result["face_encodings"] = [e.tolist() for e in face_encodings]
        result["face_locations"] = face_locations
        
        # Add match if found
        for match in matches:
            if match.get("match"):
                result["match"] = match.get("match")
                result["face_recognized"] = True
                break
        
        # Set success flag
        result["success"] = True
        
        return result
        
    except Exception as e:
        logger.error(f"Error during face recognition: {e}")
        result = {"success": False, "error": str(e)}
        return result
    
    finally:
        # Release camera
        if camera is not None:
            try:
                # Try to read any remaining frames to clear buffer
                for _ in range(3):
                    camera.read()
                camera.release()
                logger.info("Camera released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
                
        # Clean up any OpenCV windows
        try:
            cv2.destroyAllWindows()
            cv2.waitKey(1)  # Give a moment for windows to close
            logger.info("Destroyed all OpenCV windows")
        except Exception as e:
            logger.error(f"Error destroying windows: {e}")
                
        # Sleep briefly to ensure camera resources are freed
        time.sleep(0.5)


def main():
    """Main entry point for standalone execution"""
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Face Recognition Process")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--backend", type=str, help="Backend URL")
    parser.add_argument("--output", type=str, help="Output file for results")
    parser.add_argument("--skip-liveness", action="store_true", help="Skip liveness detection")
    parser.add_argument("--debug-dir", type=str, help="Directory to save debug frames")
    
    args = parser.parse_args()
    
    # Set a maximum processing time for face recognition 
    max_processing_time = 8  # Reduced from 10 seconds for faster timeout
    
    # Log the start time
    start_time = time.time()
    
    # Run face recognition
    results = run_face_recognition(
        camera_index=args.camera,
        backend_url=args.backend,
        skip_liveness=args.skip_liveness,
        debug_dir=args.debug_dir
    )
    
    # Log the end time
    end_time = time.time()
    logger.info(f"Face recognition completed in {end_time - start_time:.2f} seconds")
    
    # Write results to file if specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f)
        logger.info(f"Results written to {args.output}")
    else:
        # Print results to stdout
        print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main() 