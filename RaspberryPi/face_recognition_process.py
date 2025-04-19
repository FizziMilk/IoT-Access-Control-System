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
import pickle

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

class LivenessDetector:
    """Simple liveness detection for face recognition"""
    
    def __init__(self):
        # Default thresholds - can be tuned based on environment
        self.entropy_threshold = 4.0
        self.gradient_threshold = 15.0
        self.reflection_threshold = 0.05
        self.color_threshold = 18.0
    
    def check_liveness(self, frame):
        """
        Check if a face appears to be from a live person
        
        Args:
            frame: OpenCV BGR image containing a face
            
        Returns:
            dict: Liveness metrics and result
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate LBP for texture analysis
        lbp_image = self._calculate_lbp(gray)
        
        # Calculate texture entropy
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=[0, 256])
        hist = hist / float(hist.sum() + 1e-10)  # Avoid division by zero
        texture_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calculate gradients
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(gradient_magnitude)
        
        # Detect reflections
        _, bright_spots = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_spot_count = cv2.countNonZero(bright_spots)
        bright_spot_ratio = bright_spot_count / (gray.shape[0] * gray.shape[1])
        
        # Analyze color variance
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:,:,1])  # Saturation channel
        
        # Evaluate liveness
        entropy_ok = texture_entropy > self.entropy_threshold
        gradient_ok = mean_gradient > self.gradient_threshold
        reflection_ok = bright_spot_ratio < self.reflection_threshold
        color_ok = color_std > self.color_threshold
        
        # Combined result
        is_live = entropy_ok and gradient_ok and reflection_ok and color_ok
        
        # Log metrics
        logger.info(f"Liveness: entropy={texture_entropy:.2f}, gradient={mean_gradient:.2f}, "
                    f"reflection={bright_spot_ratio:.4f}, color={color_std:.2f}, live={is_live}")
        
        return {
            "texture_entropy": float(texture_entropy),
            "mean_gradient": float(mean_gradient),
            "bright_spot_ratio": float(bright_spot_ratio),
            "color_std": float(color_std),
            "is_live": is_live
        }
    
    def _calculate_lbp(self, img, radius=1, num_points=8):
        """Calculate Local Binary Pattern for texture analysis"""
        lbp = np.zeros_like(img)
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
                            binary_pattern |= (1 << k)
                
                lbp[i, j] = binary_pattern
        
        return lbp


class WebRecognition:
    """
    Face recognition handler for web applications.
    """
    
    def __init__(self, backend_url=None):
        """
        Initialize the recognition system.
        
        Args:
            backend_url: URL of the backend API for retrieving known faces
        """
        self.backend_url = backend_url
        self.known_face_encodings = []
        self.known_face_names = []
        self.liveness_detector = LivenessDetector()
        self.load_encodings()
        
    def load_encodings(self):
        """
        Load face encodings and names.
        
        Raises:
            Exception: If there is a mismatch in the lengths of encodings and names
        """
        try:
            # Try to load from the backend first
            if self.backend_url:
                self.load_known_faces()
            
            # If no encoding from the backend, try to load from file
            if not self.known_face_encodings:
                # Check if the pickle file exists
                if os.path.exists("known_face_encodings.pkl") and os.path.exists("known_face_names.pkl"):
                    logging.info("Loading face encodings from local files")
                    with open("known_face_encodings.pkl", "rb") as f:
                        self.known_face_encodings = pickle.load(f)
                    with open("known_face_names.pkl", "rb") as f:
                        self.known_face_names = pickle.load(f)
            
            # Ensure all encodings are numpy arrays
            self.known_face_encodings = [np.array(encoding) if not isinstance(encoding, np.ndarray) else encoding 
                                        for encoding in self.known_face_encodings]
            
            # Verify that the number of encodings and names match
            if len(self.known_face_encodings) != len(self.known_face_names):
                raise Exception(f"Mismatch in encodings and names: {len(self.known_face_encodings)} encodings, {len(self.known_face_names)} names")
                
            logging.info(f"Loaded {len(self.known_face_encodings)} face encodings")
            
        except Exception as e:
            logging.error(f"Error loading encodings: {str(e)}")
            # Initialize empty lists if loading fails
            self.known_face_encodings = []
            self.known_face_names = []
            
    def identify_face(self, frame=None, face_location=None, face_encoding=None):
        """
        Identify a face using either a provided frame or encoding.
        
        Args:
            frame: Optional image frame containing a face
            face_location: Optional location of face in the frame
            face_encoding: Optional pre-computed face encoding
            
        Returns:
            Dictionary with name and confidence of the match, or None if no match
        """
        # Ensure we have a face encoding to work with
        if face_encoding is None and frame is not None and face_location is not None:
            face_encoding = face_recognition.face_encodings(frame, [face_location])[0]
        elif face_encoding is None:
            logging.error("Cannot identify face: need either face_encoding or both frame and face_location")
            return None
            
        # Convert face_encoding to numpy array if it's not already
        if not isinstance(face_encoding, np.ndarray):
            face_encoding = np.array(face_encoding)
            
        # Check if we have any known face encodings to compare against
        if not self.known_face_encodings:
            logging.warning("No known face encodings available for comparison")
            return None
            
        try:
            # Calculate face distances
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            # Find the index of the closest match
            best_match_index = np.argmin(face_distances)
            best_match_distance = face_distances[best_match_index]
            
            # Convert distance to confidence (0-1 scale, higher is better)
            # Using formula: confidence = 1 - min(1, face_distance / 0.6)
            confidence = max(0, min(1, 1 - (best_match_distance / 0.6)))
            
            # Only consider it a match if confidence is above threshold
            if confidence >= 0.5:  # 50% confidence threshold
                return {
                    "name": self.known_face_names[best_match_index],
                    "confidence": float(confidence),  # Convert numpy float to Python float for JSON serialization
                    "distance": float(best_match_distance)
                }
            else:
                logging.info(f"Best match below threshold: {self.known_face_names[best_match_index]} with confidence {confidence:.2f}")
                return {
                    "name": "Unknown",
                    "confidence": 0.0,
                    "distance": float(best_match_distance)
                }
                
        except Exception as e:
            logging.error(f"Error identifying face: {str(e)}")
            return {
                "name": "Unknown",
                "confidence": 0.0,
                "distance": 1.0
            }
            
    def check_liveness(self, frame, face_location):
        """
        Check if a detected face is live.
        
        Args:
            frame: Image frame containing a face
            face_location: Location of face in the frame as (top, right, bottom, left)
            
        Returns:
            Boolean indicating whether the face is live
        """
        return self.liveness_detector.check_liveness(frame, face_location)
    
    def load_known_faces(self):
        """
        Load known face encodings from a backend API.
        
        Returns:
            Boolean indicating success or failure
        """
        if not self.backend_url:
            logging.warning("No backend URL provided for loading known faces")
            return False
            
        try:
            # Construct the endpoint URL for face encodings
            url = f"{self.backend_url.rstrip('/')}/api/face-encodings"
            logging.info(f"Fetching face encodings from: {url}")
            
            # Make the request
            response = requests.get(url, timeout=10)
            
            # Check if the request was successful
            if response.status_code != 200:
                logging.error(f"Failed to fetch face encodings: HTTP {response.status_code}")
                return False
                
            # Parse the response data
            data = response.json()
            
            if not data or not isinstance(data, list):
                logging.error(f"Invalid response data: {data}")
                return False
                
            # Clear existing encodings
            self.known_face_encodings = []
            self.known_face_names = []
            
            # Process each encoding
            for item in data:
                if not isinstance(item, dict) or "encoding" not in item or "name" not in item:
                    logging.warning(f"Skipping invalid encoding entry: {item}")
                    continue
                    
                # Convert encoding to numpy array
                try:
                    encoding = np.array(item["encoding"])
                    name = item["name"]
                    
                    self.known_face_encodings.append(encoding)
                    self.known_face_names.append(name)
                except Exception as e:
                    logging.warning(f"Error processing encoding for {item.get('name')}: {str(e)}")
                    
            logging.info(f"Loaded {len(self.known_face_encodings)} face encodings from backend")
            return len(self.known_face_encodings) > 0
            
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error fetching face encodings: {str(e)}")
            return False
        except Exception as e:
            logging.error(f"Error loading known faces: {str(e)}")
            return False


def save_debug_frame(frame, debug_dir, face_locations=None, liveness_results=None, match_results=None, prefix="debug"):
    """
    Save a debug frame with visualizations of detected faces, liveness, and matches.
    
    Args:
        frame: The image frame to save
        debug_dir: Directory to save debug images
        face_locations: List of face locations as (top, right, bottom, left)
        liveness_results: List of liveness check results for each face
        match_results: List of match results for each face
        prefix: Prefix for the debug image filename
    """
    if frame is None or debug_dir is None:
        return None
        
    # Create debug directory if it doesn't exist
    if not os.path.exists(debug_dir):
        os.makedirs(debug_dir)
        
    # Create a copy of the frame for drawing
    debug_frame = frame.copy()
    
    # Draw face boxes and labels if available
    if face_locations is not None:
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            
            # Default values
            is_live = False
            match_name = "Unknown"
            confidence = 0.0
            
            # Get liveness result if available
            if liveness_results is not None and i < len(liveness_results):
                is_live = liveness_results[i]
                
            # Get match result if available
            if match_results is not None and i < len(match_results):
                match_result = match_results[i]
                if match_result is not None and "name" in match_result:
                    match_name = match_result["name"]
                    confidence = match_result.get("confidence", 0.0)
            
            # Choose color based on liveness
            color = (0, 255, 0) if is_live else (0, 0, 255)  # Green for live, Red for not live
            
            # Draw the box
            cv2.rectangle(debug_frame, (left, top), (right, bottom), color, 2)
            
            # Create label: Name (Confidence)
            if match_name != "Unknown" and confidence > 0:
                label = f"{match_name} ({confidence:.2f})"
            else:
                label = match_name
                
            # Add liveness indicator to label
            label = f"{label} - {'Live' if is_live else 'Not Live'}"
            
            # Draw label above the face box
            cv2.putText(debug_frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Add timestamp and debug info to the image
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Create info string based on available data
    info_text = f"Timestamp: {timestamp}"
    if face_locations is None:
        info_text += " | No faces detected"
    else:
        info_text += f" | Faces: {len(face_locations)}"
        if liveness_results is not None:
            live_count = sum(1 for result in liveness_results if result)
            info_text += f" | Live: {live_count}/{len(liveness_results)}"
        if match_results is not None:
            match_count = sum(1 for result in match_results if result is not None)
            info_text += f" | Matches: {match_count}/{len(match_results)}"
    
    # Add debug type
    info_text += f" | Type: {prefix}"
    
    # Draw background rectangle for info text
    cv2.rectangle(debug_frame, (0, 0), (debug_frame.shape[1], 30), (0, 0, 0), -1)
    
    # Draw info text
    cv2.putText(debug_frame, info_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Generate timestamped filename
    timestamp_file = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    debug_file = os.path.join(debug_dir, f"frame_{prefix}_{timestamp_file}.jpg")
    
    # Save the debug frame
    cv2.imwrite(debug_file, debug_frame)
    logging.info(f"Saved debug frame to {debug_file}")
    
    return os.path.basename(debug_file)


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
        
        for entry in face_data:
            phone_number = entry.get("phone_number")
            face_encoding_b64 = entry.get("face_encoding")
            
            if not phone_number or not face_encoding_b64:
                logger.warning("Skipping entry with missing data")
                continue
                
            # Decode the face encoding
            try:
                # Decode base64 to JSON string then to list
                decoded_data = base64.b64decode(face_encoding_b64)
                encoding_json = decoded_data.decode('utf-8')
                encoding_list = json.loads(encoding_json)
                encoding = np.array(encoding_list)
                
                if len(encoding) == 128:  # Standard face encoding length
                    encodings.append(encoding)
                    names.append(phone_number)
                    logger.info(f"Decoded face encoding for {phone_number}")
            except Exception as e:
                logger.error(f"Error decoding face encoding for {phone_number}: {e}")
        
        logger.info(f"Loaded {len(encodings)} face encodings")
        return encodings, names
            
    except Exception as e:
        logger.error(f"Error loading faces from backend: {e}")
        return [], []


def perform_liveness_check(frame, face_locations):
    """
    Perform liveness detection for all faces in frame
    
    Args:
        frame: OpenCV BGR image
        face_locations: List of face location tuples
        
    Returns:
        list: Liveness check results for each face
    """
    detector = LivenessDetector()
    results = []
    
    for face_loc in face_locations:
        top, right, bottom, left = face_loc
        face_img = frame[top:bottom, left:right]
        
        # Ensure face region is valid
        if face_img.size > 0:
            result = detector.check_liveness(face_img)
            results.append(result)
        else:
            results.append({"is_live": False, "error": "Invalid face region"})
    
    return results


def run_face_recognition(camera_index=0, backend_url=None, output_file=None, skip_liveness=False, debug_dir=None):
    """
    Run the face recognition process.
    
    Args:
        camera_index: Index of the camera to use
        backend_url: URL of the backend API
        output_file: File to write the results to
        skip_liveness: Whether to skip liveness checks
        debug_dir: Directory to save debug images
        
    Returns:
        Dictionary with recognition results
    """
    camera = None
    start_time = time.time()
    metrics = {
        "total_time": 0,
        "camera_init_time": 0,
        "face_detection_time": 0,
        "liveness_check_time": 0,
        "recognition_time": 0
    }
    debug_frame_path = None
    
    try:
        logging.info("Starting face recognition process")
        
        # Initialize camera
        camera_start = time.time()
        camera = WebCamera(camera_index)
        
        # Wait for camera to initialize properly (added delay)
        time.sleep(2)
        
        # Check if camera is working
        for _ in range(5):  # Try up to 5 times
            frame = camera.get_frame()
            if frame is not None and not frame.size == 0:
                break
            logging.warning("Camera not ready, retrying...")
            time.sleep(1)
        
        if frame is None or frame.size == 0:
            logging.error("Could not get frame from camera")
            return {
                "success": False,
                "message": "No camera available or cannot capture frame",
                "metrics": metrics,
                "debug_frame": debug_frame_path
            }
            
        metrics["camera_init_time"] = time.time() - camera_start
        logging.info(f"Camera initialized in {metrics['camera_init_time']:.2f} seconds")
        
        # Initialize recognition
        recognition = WebRecognition(backend_url)
        
        # Detect faces
        face_detection_start = time.time()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        metrics["face_detection_time"] = time.time() - face_detection_start
        
        if not face_locations:
            logging.warning("No faces detected in frame")
            
            # Save debug frame showing no faces detected
            if debug_dir:
                debug_frame_path = save_debug_frame(frame, debug_dir, prefix="no_faces")
                
            return {
                "success": False,
                "message": "No faces detected",
                "metrics": metrics,
                "debug_frame": debug_frame_path
            }
            
        logging.info(f"Detected {len(face_locations)} faces in {metrics['face_detection_time']:.2f} seconds")
        
        # Check liveness for each face
        liveness_results = []
        liveness_start = time.time()
        
        if not skip_liveness:
            for face_location in face_locations:
                is_live = recognition.check_liveness(frame, face_location)
                liveness_results.append(is_live)
                
            metrics["liveness_check_time"] = time.time() - liveness_start
            logging.info(f"Liveness checks completed in {metrics['liveness_check_time']:.2f} seconds")
            
            # Check if any face passed liveness test
            if not any(liveness_results):
                logging.warning("No live faces detected")
                
                # Save debug frame showing liveness failures
                if debug_dir:
                    debug_frame_path = save_debug_frame(frame, debug_dir, face_locations, liveness_results, None, prefix="no_live_faces")
                    
                return {
                    "success": False,
                    "message": "No live faces detected",
                    "metrics": metrics,
                    "debug_frame": debug_frame_path
                }
        else:
            # Skip liveness check, assume all faces are live
            liveness_results = [True] * len(face_locations)
            logging.info("Skipping liveness checks")
            
        # Identify each live face
        recognition_start = time.time()
        match_results = []
        best_match = None
        best_confidence = -1
        
        for i, (face_location, is_live) in enumerate(zip(face_locations, liveness_results)):
            if not is_live and not skip_liveness:
                match_results.append(None)
                continue
                
            # Get face encoding
            face_encoding = face_recognition.face_encodings(rgb_frame, [face_location])[0]
            
            # Identify face
            match = recognition.identify_face(face_encoding=face_encoding)
            match_results.append(match)
            
            if match and match.get("confidence", 0) > best_confidence:
                best_match = match
                best_confidence = match.get("confidence", 0)
                
        metrics["recognition_time"] = time.time() - recognition_start
        logging.info(f"Face recognition completed in {metrics['recognition_time']:.2f} seconds")
        
        # Save final debug frame with all information
        if debug_dir:
            debug_frame_path = save_debug_frame(frame, debug_dir, face_locations, liveness_results, match_results, prefix="final")
            
        # Prepare final result
        metrics["total_time"] = time.time() - start_time
        
        result = {
            "success": True,
            "message": "Face recognition completed successfully",
            "metrics": metrics,
            "debug_frame": debug_frame_path,
            "matches": [m for m in match_results if m is not None],
            "best_match": best_match
        }
        
        # Write results to file if specified
        if output_file:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
                
        return result
        
    except Exception as e:
        logging.error(f"Error in face recognition process: {str(e)}")
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "metrics": metrics,
            "debug_frame": debug_frame_path
        }
        
    finally:
        # Clean up resources
        if camera:
            camera.release()
            
        # Close any open cv2 windows
        cv2.destroyAllWindows()


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
    
    # Run face recognition
    results = run_face_recognition(
        camera_index=args.camera,
        backend_url=args.backend,
        skip_liveness=args.skip_liveness,
        debug_dir=args.debug_dir
    )
    
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