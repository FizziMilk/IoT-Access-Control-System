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
            
        # Compare with known faces
        face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        
        result = {
            "encoding": face_encoding,
            "location": face_location if 'face_location' in locals() else None,
            "match": None
        }
        
        if len(face_distances) > 0:
            # Find best match
            best_match_index = np.argmin(face_distances)
            best_match_distance = face_distances[best_match_index]
            
            # Check if the match is close enough
            if best_match_distance <= self.detection_threshold:
                match_name = self.known_face_names[best_match_index]
                match_confidence = 1.0 - best_match_distance
                
                logger.info(f"Face matched with {match_name} (confidence: {match_confidence:.2f})")
                
                result["match"] = {
                    "name": match_name,
                    "confidence": float(match_confidence),
                    "distance": float(best_match_distance)
                }
            else:
                logger.info(f"Best match too far ({best_match_distance:.2f} > {self.detection_threshold:.2f})")
        
        return result
    
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
                if result.get("is_live", False):
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
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv2.putText(debug_frame, timestamp, (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
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


def run_face_recognition(camera_index=0, backend_url=None, skip_liveness=False, debug_dir=None):
    """
    Run the face recognition process
    
    Args:
        camera_index: Camera device index
        backend_url: URL for backend API
        skip_liveness: Whether to skip liveness detection
        debug_dir: Directory to save debug frames
        
    Returns:
        dict: Recognition results
    """
    camera = None
    
    try:
        # Initialize camera with retries
        max_camera_retries = 5
        retry_count = 0
        retry_delay = 2.0  # seconds
        
        while retry_count < max_camera_retries:
            logger.info(f"Opening camera at index {camera_index} (attempt {retry_count + 1}/{max_camera_retries})")
            camera = cv2.VideoCapture(camera_index)
            
            # Give camera time to initialize
            time.sleep(1.0)
            
            # Set camera properties
            if camera.isOpened():
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                # Test if camera works by reading a frame
                ret, test_frame = camera.read()
                if ret and test_frame is not None:
                    logger.info(f"Successfully opened camera at index {camera_index}")
                    break
                else:
                    logger.warning(f"Camera opened but could not read frame, retrying...")
                    camera.release()
            
            # If we get here, camera failed to open or read frames
            logger.warning(f"Failed to open camera on attempt {retry_count + 1}, retrying in {retry_delay} seconds...")
            retry_count += 1
            
            # Release any partial camera resources
            if camera is not None:
                try:
                    camera.release()
                except:
                    pass
            
            # Wait before retry
            time.sleep(retry_delay)
        
        # If we exhausted all retries
        if retry_count >= max_camera_retries:
            logger.error(f"Failed to open camera after {max_camera_retries} attempts")
            return {"success": False, "error": "Failed to open camera after multiple attempts"}
        
        # At this point, we have a working camera
        
        # Initialize recognition system
        recognition = WebRecognition()
        
        # Load known faces
        encodings, names = load_known_faces(backend_url)
        if encodings:
            recognition.load_encodings(encodings, names)
        
        # Read multiple frames to stabilize camera
        for _ in range(5):
            ret, _ = camera.read()
            if not ret:
                logger.warning("Failed to read warmup frame")
            time.sleep(0.1)
        
        # Capture and process frame
        ret, frame = camera.read()
        
        if not ret or frame is None:
            logger.error("Failed to capture frame")
            return {"success": False, "error": "Failed to capture frame"}
        
        # Save initial frame if debugging
        if debug_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_debug_frame(frame, f"{debug_dir}/frame_initial_{timestamp}.jpg")
        
        # Convert to RGB for face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if not face_locations:
            logger.warning("No faces detected in frame")
            
            # Save debug frame with no faces
            if debug_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_debug_frame(frame, f"{debug_dir}/frame_nofaces_{timestamp}.jpg")
            
            return {"success": True, "face_detected": False}
        
        # Save frame with faces
        if debug_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_debug_frame(frame, f"{debug_dir}/frame_faces_{timestamp}.jpg", 
                            faces=face_locations)
        
        # Perform liveness check if not skipped
        liveness_results = None
        if not skip_liveness:
            liveness_results = perform_liveness_check(frame, face_locations)
            
            # Check if any face passes liveness
            all_fake = all(not result.get("is_live", False) for result in liveness_results)
            
            if all_fake:
                logger.warning("All detected faces failed liveness check")
                
                # Save debug frame with liveness failures
                if debug_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_debug_frame(frame, f"{debug_dir}/frame_liveness_fail_{timestamp}.jpg", 
                                    faces=face_locations, liveness_results=liveness_results)
                
                return {
                    "success": True,
                    "face_detected": True,
                    "is_live": False,
                    "liveness_results": liveness_results
                }
        
        # Get face encodings
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        # Identify faces
        matches = []
        for face_encoding in face_encodings:
            # Compare with known faces if we have any
            if recognition.known_face_encodings:
                face_distances = face_recognition.face_distance(
                    recognition.known_face_encodings, face_encoding)
                
                if len(face_distances) > 0:
                    # Find best match
                    best_match_index = np.argmin(face_distances)
                    best_match_distance = face_distances[best_match_index]
                    
                    # Check if match is close enough
                    if best_match_distance <= recognition.detection_threshold:
                        match_name = recognition.known_face_names[best_match_index]
                        match_confidence = 1.0 - best_match_distance
                        
                        matches.append({
                            "match": {
                                "name": match_name,
                                "confidence": float(match_confidence),
                                "distance": float(best_match_distance)
                            }
                        })
                        continue
            
            # No match found
            matches.append({"match": None})
        
        # Save final debug frame with all information
        if debug_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_debug_frame(frame, f"{debug_dir}/frame_final_{timestamp}.jpg", 
                           faces=face_locations, liveness_results=liveness_results, 
                           matches=matches)
        
        # Make sure NumPy arrays are converted to lists for JSON serialization
        result = {
            "success": True,
            "face_detected": True,
            "face_encodings": [e.tolist() for e in face_encodings],
            "face_locations": face_locations
        }
        
        # Add liveness results if available
        if liveness_results:
            result["is_live"] = any(r.get("is_live", False) for r in liveness_results)
            result["liveness_results"] = liveness_results
        
        # Add match if found
        for match in matches:
            if match.get("match"):
                result["match"] = match.get("match")
                break
        
        return result
        
    except Exception as e:
        logger.error(f"Error during face recognition: {e}")
        return {"success": False, "error": str(e)}
    
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