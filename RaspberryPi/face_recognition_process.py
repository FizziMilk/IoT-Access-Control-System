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
    """Face recognition system for web applications"""
    
    def __init__(self, backend_url=None):
        """Initialize the recognition system"""
        logger.info("Initializing WebRecognition")
        self.known_face_encodings = []
        self.known_face_names = []
        self.detection_threshold = 0.6  # Lower values are more strict
        self.liveness_detector = LivenessDetector()
        self.backend_url = backend_url
        
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
        
    def identify_face(self, frame=None, face_encoding=None, face_location=None):
        """
        Identify a face in the provided frame or using a provided encoding.
        Returns a match object with name, distance, and confidence.
        """
        logger.info(f"Attempting to identify face with {len(self.known_face_encodings)} known faces")
        
        # If we don't have any known faces to compare against, return no match
        if len(self.known_face_encodings) == 0:
            logger.warning("No known faces to compare against")
            return {"match": None, "distance": 1.0, "confidence": 0}
        
        if face_encoding is None and frame is not None:
            # Detect faces if location not provided
            if face_location is None:
                face_locations = face_recognition.face_locations(frame)
                if not face_locations:
                    logger.warning("No faces detected during identification")
                    return {"match": None, "distance": 1.0, "confidence": 0}
                face_location = face_locations[0]  # Use the first face
            
            # Get face encoding
            face_encoding = face_recognition.face_encodings(frame, [face_location])
            if not face_encoding:
                logger.warning("Failed to extract face encoding")
                return {"match": None, "distance": 1.0, "confidence": 0}
            face_encoding = face_encoding[0]
        
        # Ensure face_encoding is a numpy array
        if not isinstance(face_encoding, np.ndarray):
            try:
                face_encoding = np.array(face_encoding)
                logger.debug("Converted face encoding to numpy array")
            except Exception as e:
                logger.error(f"Failed to convert face encoding to numpy array: {e}")
                return {"match": None, "distance": 1.0, "confidence": 0}
                
        # Calculate face distance to all known faces
        try:
            # Verify all known face encodings are numpy arrays
            self.known_face_encodings = [
                np.array(encoding) if not isinstance(encoding, np.ndarray) else encoding
                for encoding in self.known_face_encodings
            ]
            
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            logger.debug(f"Face distances: {face_distances}")
            
            # Find the best match (lowest distance)
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]
                
                # Convert distance to confidence (0-100)
                # The lower the distance, the higher the confidence
                confidence = (1 - best_match_distance) * 100
                
                # Check if the match is good enough (threshold)
                # A lower distance means a better match
                if best_match_distance < 0.6:  # Threshold for good match
                    match = {
                        "name": self.known_face_names[best_match_index],
                        "distance": float(best_match_distance),
                        "confidence": float(confidence)
                    }
                    logger.info(f"Face identified as {match['name']} with confidence {confidence:.2f}%")
                    return {"match": match, "distance": float(best_match_distance), "confidence": float(confidence)}
                else:
                    logger.info(f"Face not recognized (best distance: {best_match_distance:.4f}, threshold: 0.6)")
                    return {"match": None, "distance": float(best_match_distance), "confidence": float(confidence)}
            else:
                logger.warning("No face distances computed")
                return {"match": None, "distance": 1.0, "confidence": 0}
                
        except Exception as e:
            logger.error(f"Error during face identification: {e}")
            return {"match": None, "distance": 1.0, "confidence": 0}
    
    def check_liveness(self, frame, face_location=None):
        """Check if a face is live"""
        # If no face location provided, get the whole frame
        if face_location is None:
            return self.liveness_detector.check_liveness(frame)
        
        # Extract face region
        top, right, bottom, left = face_location
        face_img = frame[top:bottom, left:right]
        
        # Ensure face region is valid
        if face_img.size == 0:
            logger.warning("Invalid face region for liveness check")
            return {"is_live": False, "error": "Invalid face region"}
        
        return self.liveness_detector.check_liveness(face_img)


def save_debug_frame(frame, output_path, faces=None, liveness_results=None, matches=None):
    """
    Save a debug frame with visual annotations for face detection, liveness and matching.
    
    Args:
        frame: The original camera frame
        output_path: Path to save the debug image
        faces: List of face locations (top, right, bottom, left)
        liveness_results: List of liveness check results
        matches: List of match results
    """
    try:
        # Make a copy of the frame to avoid modifying the original
        debug_frame = frame.copy()
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(debug_frame, timestamp, (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw faces if provided
        if faces:
            for i, face_location in enumerate(faces):
                # Handle both tuple format and dictionary format
                if isinstance(face_location, tuple):
                    top, right, bottom, left = face_location
                elif isinstance(face_location, dict):
                    top = face_location.get('top', 0)
                    right = face_location.get('right', 0)
                    bottom = face_location.get('bottom', 0)
                    left = face_location.get('left', 0)
                else:
                    logger.warning(f"Unknown face_location format: {type(face_location)}")
                    continue
                
                # Default color for face box (yellow)
                color = (0, 255, 255)  # Yellow = default
                
                # Check if we have liveness results for this face
                is_live = False
                if liveness_results and i < len(liveness_results):
                    liveness_result = liveness_results[i]
                    if isinstance(liveness_result, dict) and liveness_result.get('is_live', False):
                        color = (0, 255, 0)  # Green = live face
                        is_live = True
                    else:
                        color = (0, 0, 255)  # Red = fake face
                
                # Draw face box
                cv2.rectangle(debug_frame, (left, top), (right, bottom), color, 2)
                
                # Add liveness text
                liveness_text = "LIVE" if is_live else "FAKE"
                cv2.putText(debug_frame, liveness_text, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Add match information if available
                if matches and i < len(matches):
                    match_info = matches[i]
                    name = "Unknown"
                    confidence = 0
                    
                    # Better handling of different match formats
                    logger.debug(f"Match info format: {type(match_info)} - {match_info}")
                    
                    # Format 1: {"match": {"name": "...", "confidence": ...}, ...}
                    if isinstance(match_info, dict) and "match" in match_info:
                        match_data = match_info.get("match")
                        if match_data and isinstance(match_data, dict):
                            name = match_data.get("name", "Unknown")
                            confidence = match_data.get("confidence", 0)
                    
                    # Format 2: Direct data in the object
                    elif isinstance(match_info, dict) and "name" in match_info:
                        name = match_info.get("name", "Unknown")
                        confidence = match_info.get("confidence", 0)
                    
                    # Format 3: Object with only confidence/distance but no match
                    elif isinstance(match_info, dict) and "confidence" in match_info:
                        confidence = match_info.get("confidence", 0)
                    
                    # Prepare text to display
                    if name != "Unknown" and confidence > 0:
                        match_text = f"{name} ({confidence:.1f}%)"
                    elif confidence > 0:
                        match_text = f"Unknown ({confidence:.1f}%)"
                    else:
                        match_text = "Unknown"
                    
                    # Draw match text under the face
                    text_y = bottom + 25
                    cv2.putText(debug_frame, match_text, (left, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        # Save the debug frame
        cv2.imwrite(output_path, debug_frame)
        logger.info(f"Saved debug frame to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving debug frame: {e}", exc_info=True)


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


def run_face_recognition(camera_index=0, backend_url=None, output_file=None, perform_liveness_check=True, debug_dir=None):
    """
    Run the face recognition process
    
    Args:
        camera_index: Index of the camera to use
        backend_url: URL of the backend API for face data
        output_file: Path to save JSON results
        perform_liveness_check: Whether to perform liveness check
        debug_dir: Directory to save debug frames
        
    Returns:
        dict with recognition results
    """
    camera = None
    result = {
        "success": False,
        "message": "",
        "results": [],
        "debug_frames": []
    }
    
    try:
        # Initialize the WebRecognition class
        recognition = WebRecognition(backend_url=backend_url)
        
        # Try to open the camera with multiple attempts
        logger.info(f"Attempting to open camera {camera_index}")
        max_attempts = 5  # Increased attempts
        for attempt in range(1, max_attempts + 1):
            try:
                # Release any existing camera first
                if camera and camera.isOpened():
                    camera.release()
                    time.sleep(0.5)  # Wait for release to complete
                
                # Open the camera
                camera = cv2.VideoCapture(camera_index)
                
                # Wait a bit and check if it's really open
                time.sleep(0.5)
                if not camera.isOpened():
                    logger.warning(f"Failed to open camera (attempt {attempt}/{max_attempts})")
                    if attempt < max_attempts:
                        time.sleep(1.5)  # Longer wait before retrying
                        continue
                    else:
                        result["message"] = f"Failed to open camera after {max_attempts} attempts"
                        return result
                else:
                    # Set camera properties for better performance
                    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffering delay
                    logger.info(f"Camera opened successfully on attempt {attempt}")
                    break
            except Exception as e:
                logger.error(f"Error opening camera (attempt {attempt}/{max_attempts}): {e}")
                if attempt < max_attempts:
                    time.sleep(1.5)  # Longer wait before retrying
                    continue
                else:
                    result["message"] = f"Error opening camera: {str(e)}"
                    return result
        
        # Enhanced camera warmup - discard first few frames with better checking
        logger.info("Starting enhanced camera warmup...")
        warmup_frames = 20  # Increased from 10 to 20
        valid_frames = 0
        max_warmup_attempts = 30  # Maximum number of attempts to get valid frames
        attempt = 0
        
        while valid_frames < warmup_frames and attempt < max_warmup_attempts:
            ret, frame = camera.read()
            attempt += 1
            
            if ret and frame is not None and frame.size > 0:
                valid_frames += 1
                logger.debug(f"Warmup frame {valid_frames}/{warmup_frames} captured")
            else:
                logger.debug(f"Invalid frame during warmup (attempt {attempt})")
            
            time.sleep(0.1)  # Small delay between frames
        
        if valid_frames < warmup_frames:
            logger.warning(f"Camera warmup incomplete: only {valid_frames}/{warmup_frames} valid frames captured")
        else:
            logger.info("Camera warmup completed successfully")
        
        # Capture frame with enhanced retry
        logger.info("Capturing frame for face recognition")
        max_frame_attempts = 8  # Increased from 5 to 8
        frame = None
        
        for frame_attempt in range(1, max_frame_attempts + 1):
            ret, frame = camera.read()
            if not ret or frame is None or frame.size == 0:
                logger.warning(f"Failed to capture valid frame (attempt {frame_attempt}/{max_frame_attempts})")
                if frame_attempt < max_frame_attempts:
                    time.sleep(0.3)  # Increased wait before retrying
                    continue
                else:
                    result["message"] = "Failed to capture a valid frame"
                    return result
            else:
                logger.info(f"Valid frame captured on attempt {frame_attempt}")
                break
                
        # Save initial debug frame
        if debug_dir:
            initial_debug_path = os.path.join(debug_dir, "initial_frame.jpg")
            save_debug_frame(frame, initial_debug_path)
            result["debug_frames"].append(initial_debug_path)
        
        # Step 1: Detect faces in the frame
        logger.info("Detecting faces in frame")
        face_locations = face_recognition.face_locations(frame)
        
        if not face_locations:
            logger.warning("No faces detected in frame")
            result["message"] = "No faces detected"
            
            # Save debug frame showing no faces detected
            if debug_dir:
                no_face_debug_path = os.path.join(debug_dir, "no_faces.jpg")
                save_debug_frame(frame, no_face_debug_path)
                result["debug_frames"].append(no_face_debug_path)
                
            return result
            
        logger.info(f"Detected {len(face_locations)} faces")
        
        # Save debug frame with face detections
        if debug_dir:
            faces_debug_path = os.path.join(debug_dir, "faces_detected.jpg")
            save_debug_frame(frame, faces_debug_path, faces=face_locations)
            result["debug_frames"].append(faces_debug_path)
        
        # Step 2: Perform liveness check if requested
        liveness_results = []
        if perform_liveness_check:
            logger.info("Performing liveness check")
            liveness_detector = LivenessDetector()
            
            for face_location in face_locations:
                liveness_result = liveness_detector.check_liveness(frame, face_location)
                liveness_results.append(liveness_result)
                
                if not liveness_result.get("is_live", False):
                    logger.warning(f"Liveness check failed: {liveness_result}")
            
            # Check if any face passed liveness
            if not any(result.get("is_live", False) for result in liveness_results):
                logger.warning("All faces failed liveness check")
                result["message"] = "Liveness check failed for all faces"
                result["results"] = [
                    {"face_location": loc, "liveness": lv, "match": None} 
                    for loc, lv in zip(face_locations, liveness_results)
                ]
                
                # Save debug frame with liveness results
                if debug_dir:
                    liveness_debug_path = os.path.join(debug_dir, "liveness_failed.jpg")
                    save_debug_frame(frame, liveness_debug_path, faces=face_locations, liveness_results=liveness_results)
                    result["debug_frames"].append(liveness_debug_path)
                    
                return result
                
            # Save debug frame with liveness results
            if debug_dir:
                liveness_debug_path = os.path.join(debug_dir, "liveness_check.jpg")
                save_debug_frame(frame, liveness_debug_path, faces=face_locations, liveness_results=liveness_results)
                result["debug_frames"].append(liveness_debug_path)
        
        # Step 3: Get face encodings
        logger.info("Extracting face encodings")
        face_encodings = face_recognition.face_encodings(frame, face_locations)
        
        if not face_encodings:
            logger.warning("Could not extract face encodings")
            result["message"] = "Failed to extract face encodings"
            return result
            
        # Step 4: Identify faces
        logger.info("Identifying faces against known encodings")
        match_results = []
        
        for i, face_encoding in enumerate(face_encodings):
            # Only process faces that passed liveness check if liveness was requested
            if perform_liveness_check and not liveness_results[i].get("is_live", True):
                logger.info(f"Skipping identification for face {i} due to failed liveness check")
                match_results.append({"match": None, "distance": 1.0, "confidence": 0})
                continue
                
            match = recognition.identify_face(face_encoding=face_encoding)
            match_results.append(match)
            
            if match.get("match"):
                logger.info(f"Face {i} identified as {match['match']['name']} with confidence {match['match']['confidence']:.2f}")
            else:
                logger.info(f"Face {i} not recognized")
        
        # Step 5: Prepare results
        face_data = []
        
        for i, (face_location, face_encoding) in enumerate(zip(face_locations, face_encodings)):
            face_result = {
                "face_location": {
                    "top": int(face_location[0]),
                    "right": int(face_location[1]),
                    "bottom": int(face_location[2]),
                    "left": int(face_location[3])
                }
            }
            
            if perform_liveness_check:
                face_result["liveness"] = liveness_results[i]
                
            face_result["match"] = match_results[i].get("match")
            face_result["confidence"] = match_results[i].get("confidence", 0)
            face_result["distance"] = match_results[i].get("distance", 1.0)
            face_data.append(face_result)
            
        result["success"] = True
        result["message"] = "Face recognition completed"
        result["results"] = face_data
        
        # Save final debug frame
        if debug_dir:
            final_debug_path = os.path.join(debug_dir, "final_result.jpg")
            # Ensure matches are properly formatted for save_debug_frame
            save_debug_frame(frame, final_debug_path, faces=face_locations, 
                            liveness_results=liveness_results, matches=match_results)
            result["debug_frames"].append(final_debug_path)
            
        # Save results to output file if specified
        if output_file:
            try:
                with open(output_file, 'w') as f:
                    # Ensure results are JSON serializable
                    json_result = json.dumps(result, cls=NumpyEncoder)
                    f.write(json_result)
                logger.info(f"Results saved to {output_file}")
            except Exception as e:
                logger.error(f"Error saving results to file: {e}")
                
    except Exception as e:
        error_msg = f"Error during face recognition: {str(e)}"
        logger.error(error_msg, exc_info=True)
        result["message"] = error_msg
    finally:
        # Release camera resources
        if camera:
            try:
                camera.release()
                logger.info("Camera resources released")
            except Exception as e:
                logger.error(f"Error releasing camera: {e}")
        
        # Close all OpenCV windows if any
        try:
            cv2.destroyAllWindows()
        except Exception as e:
            logger.error(f"Error closing OpenCV windows: {e}")
        
    return result


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