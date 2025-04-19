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

# Add WebRecognition class implementation
class WebRecognition:
    """
    Face recognition system optimized for web applications.
    Handles storing face encodings and recognizing faces.
    """
    
    def __init__(self):
        """Initialize the recognition system."""
        logger.info("Initializing WebRecognition")
        self.known_face_encodings = []
        self.known_face_names = []
        self.detection_threshold = 0.6  # Lower values are more strict
        
    def load_encodings(self, encodings, names):
        """
        Load face encodings and corresponding names.
        
        Args:
            encodings: List of face encodings (numpy arrays)
            names: List of names corresponding to each encoding
            
        Returns:
            bool: True if successful
        """
        if len(encodings) != len(names):
            logger.error(f"Mismatch between encodings ({len(encodings)}) and names ({len(names)})")
            return False
            
        self.known_face_encodings = encodings
        self.known_face_names = names
        logger.info(f"Loaded {len(encodings)} encodings with names")
        return True
        
    def identify_face(self, frame, face_location=None):
        """
        Identify a face in the given frame.
        
        Args:
            frame: OpenCV BGR image
            face_location: Optional face location tuple (top, right, bottom, left)
            
        Returns:
            dict or None: Match information if found, None otherwise
        """
        try:
            if not self.known_face_encodings:
                logger.warning("No known face encodings to match against")
                return None
                
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
            
            # Compare with known faces
            face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
            
            if len(face_distances) > 0:
                # Find best match
                best_match_index = np.argmin(face_distances)
                best_match_distance = face_distances[best_match_index]
                
                # Check if the match is close enough
                if best_match_distance <= self.detection_threshold:
                    match_name = self.known_face_names[best_match_index]
                    match_confidence = 1.0 - best_match_distance  # Convert distance to confidence
                    
                    logger.info(f"Face matched with {match_name} (confidence: {match_confidence:.2f})")
                    
                    return {
                        "name": match_name,
                        "confidence": float(match_confidence),
                        "distance": float(best_match_distance)
                    }
                else:
                    logger.info(f"Best match too far ({best_match_distance:.2f} > {self.detection_threshold:.2f})")
            
            return None
            
        except Exception as e:
            logger.error(f"Error identifying face: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def process_face_encoding(self, face_image):
        """
        Generate face encoding from image.
        
        Args:
            face_image: OpenCV BGR image
            
        Returns:
            numpy.ndarray or None: Face encoding if successful
        """
        try:
            # Convert to RGB for face_recognition
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_image)
            
            if not face_locations:
                logger.warning("No face detected in image for encoding")
                return None
                
            # Generate encoding
            face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
            
            if not face_encodings:
                logger.warning("Could not generate face encoding")
                return None
                
            return face_encodings[0]
            
        except Exception as e:
            logger.error(f"Error processing face encoding: {e}")
            logger.error(traceback.format_exc())
            return None

def setup_camera():
    """Initialize the camera and return the camera object"""
    # Try multiple camera indices
    camera_indices = [0, 1, 2]  # Try these indices in order
    
    for idx in camera_indices:
        try:
            logger.info(f"Attempting to open camera at index {idx}")
            
            # First check if camera is already in use by trying to release it
            try:
                temp_cam = cv2.VideoCapture(idx)
                if temp_cam.isOpened():
                    temp_cam.release()
                    time.sleep(0.5)  # Give OS time to fully release resources
            except Exception as e:
                logger.warning(f"Error checking camera {idx} status: {e}")
            
            # Now try to open it for real
            camera = cv2.VideoCapture(idx)
            
            # Give camera more time to initialize (increased from 0.3 to 1.0 seconds)
            time.sleep(1.0)
            
            # Check if camera opened successfully
            if camera.isOpened():
                # Set camera properties
                camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
                
                # Read a test frame to confirm camera is working
                ret, test_frame = camera.read()
                if not ret or test_frame is None:
                    logger.warning(f"Camera {idx} opened but failed to provide frame, trying next index")
                    camera.release()
                    continue
                
                logger.info(f"Successfully opened camera at index {idx}")
                return camera
            else:
                logger.warning(f"Failed to open camera {idx}, trying next index")
        except Exception as e:
            logger.error(f"Error initializing camera at index {idx}: {str(e)}")
    
    logger.error("Failed to open any camera")
    return None

def perform_liveness_check(frame):
    """
    Perform liveness detection to prevent spoofing with photos/videos
    
    Args:
        frame: OpenCV image to check
        
    Returns:
        dict: Liveness check results with metrics and decision
    """
    try:
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate Local Binary Pattern (LBP) for texture analysis
        def get_lbp_texture(img, radius=1, num_points=8):
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
                        
                        # Interpolate if needed
                        x_floor, x_ceil = int(np.floor(x)), int(np.ceil(x))
                        y_floor, y_ceil = int(np.floor(y)), int(np.ceil(y))
                        
                        if x_floor >= 0 and x_ceil < img.shape[1] and y_floor >= 0 and y_ceil < img.shape[0]:
                            # Simple interpolation
                            value = img[y_floor, x_floor]
                            
                            # Compare with center
                            if value >= center:
                                binary_pattern |= (1 << k)
                    
                    lbp[i, j] = binary_pattern
            
            return lbp
        
        # Calculate LBP texture
        lbp_image = get_lbp_texture(gray)
        
        # Calculate texture entropy (measure of randomness in texture)
        hist, _ = np.histogram(lbp_image.ravel(), bins=256, range=[0, 256])
        hist = hist / float(hist.sum())
        texture_entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        # Calculate gradients (real faces have more gradient variation)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        mean_gradient = np.mean(gradient_magnitude)
        
        # Detect face reflections (printed photos have more uniform reflections)
        # - Use specular reflection detection by looking at bright spots
        _, bright_spots = cv2.threshold(gray, 220, 255, cv2.THRESH_BINARY)
        bright_spot_count = cv2.countNonZero(bright_spots)
        bright_spot_ratio = bright_spot_count / (gray.shape[0] * gray.shape[1])
        
        # Analyze color variance in face region (screens/photos may have less variance)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        color_std = np.std(hsv[:,:,1])  # Saturation channel
        
        # Make liveness decision based on multiple features
        # These thresholds may need tuning based on testing
        results = {
            "texture_entropy": float(texture_entropy),
            "mean_gradient": float(mean_gradient),
            "bright_spot_ratio": float(bright_spot_ratio),
            "color_std": float(color_std),
            "is_live": False
        }
        
        # Combined decision logic
        entropy_ok = texture_entropy > 4.0  # Lower threshold from 4.5 to 4.0 for texture entropy
        gradient_ok = mean_gradient > 15.0  # Higher gradient variation in real faces
        reflection_ok = bright_spot_ratio < 0.05  # Not too many bright reflective spots
        color_ok = color_std > 18.0  # Lower threshold from 20.0 to 18.0 for color variation
        
        # Final liveness decision
        results["is_live"] = entropy_ok and gradient_ok and reflection_ok and color_ok
        
        logger.info(f"Liveness check - entropy: {texture_entropy:.2f}, gradient: {mean_gradient:.2f}, " +
                    f"reflection: {bright_spot_ratio:.4f}, color_std: {color_std:.2f}, " +
                    f"is_live: {results['is_live']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Error in liveness check: {str(e)}")
        logger.error(traceback.format_exc())
        return {"is_live": False, "error": str(e)}

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
                # First attempt: Try as base64 encoded JSON string
                decoded_data = base64.b64decode(face_encoding_b64)
                encoding_json = decoded_data.decode('utf-8')
                encoding_list = json.loads(encoding_json)
                encoding = np.array(encoding_list)
                
                if len(encoding) == 128:  # Typical face encoding length
                    encodings.append(encoding)
                    names.append(phone_number)
                    logger.info(f"Decoded face encoding for {phone_number}")
            except Exception as e:
                logger.error(f"Error decoding face encoding for {phone_number}: {e}")
                logger.error(traceback.format_exc())
        
        logger.info(f"Loaded {len(encodings)} face encodings")
        return encodings, names
            
    except Exception as e:
        logger.error(f"Error loading faces from backend: {e}")
        logger.error(traceback.format_exc())
        return [], []

def capture_and_analyze_frame(camera, recognition):
    """
    Capture a frame and perform face recognition with liveness check
    
    Args:
        camera: OpenCV camera object
        recognition: WebRecognition object
        
    Returns:
        dict: Recognition results
    """
    result = {
        'success': False,
        'face_detected': False,
        'face_locations': [],
        'face_encodings': [],
        'is_live': False,
        'liveness_metrics': {},
        'match': None,
        'error': None
    }
    
    try:
        # Capture frame
        ret, frame = camera.read()
        if not ret or frame is None:
            result['error'] = "Failed to capture frame"
            return result
            
        # Convert to RGB (face_recognition uses RGB)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_frame)
        
        if face_locations:
            result['face_detected'] = True
            result['face_locations'] = [[top, right, bottom, left] for top, right, bottom, left in face_locations]
            
            # Focus on the first detected face
            face_location = face_locations[0]
            
            # Extract face area for liveness check
            top, right, bottom, left = face_location
            face_image = frame[top:bottom, left:right]
            
            # Perform liveness check on face region
            if face_image.size > 0:  # Ensure face_image is not empty
                liveness_result = perform_liveness_check(face_image)
                result['is_live'] = liveness_result['is_live']
                result['liveness_metrics'] = {
                    k: v for k, v in liveness_result.items() if k != 'is_live'
                }
                
                # Only proceed with recognition if the face seems live
                if liveness_result['is_live']:
                    # Get face encodings
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                    result['face_encodings'] = [encoding.tolist() for encoding in face_encodings]
                    
                    # Try to identify the face if we have a recognition object and known faces
                    if recognition and recognition.known_face_encodings:
                        match = recognition.identify_face(frame, face_location)
                        if match:
                            result['match'] = match
            else:
                logger.warning("Empty face image extracted, skipping liveness check")
            
        result['success'] = True
        return result
    
    except Exception as e:
        logger.error(f"Error during face recognition: {str(e)}")
        logger.error(traceback.format_exc())
        result['error'] = f"Error: {str(e)}"
        return result

def main():
    """Main entry point for the face recognition process"""
    parser = argparse.ArgumentParser(description='Face Recognition Process')
    parser.add_argument('--output', required=True, help='Output file path for recognition results')
    parser.add_argument('--backend-url', required=False, help='Backend URL for authentication')
    parser.add_argument('--timeout', type=int, default=30, help='Timeout in seconds')
    parser.add_argument('--skip-liveness', action='store_true', help='Skip liveness detection (for testing)')
    
    args = parser.parse_args()
    
    camera = None
    result = {
        'success': False,
        'error': None,
        'face_detected': False,
        'is_live': False
    }
    
    try:
        logger.info(f"Starting face recognition process. Output will be saved to {args.output}")
        
        # Initialize recognition system
        recognition = WebRecognition()
        
        # Load known faces if backend URL provided
        if args.backend_url:
            logger.info(f"Loading known faces from backend: {args.backend_url}")
            encodings, names = load_known_faces(args.backend_url)
            if encodings:
                recognition.load_encodings(encodings, names)
        
        # Initialize camera
        camera = setup_camera()
        if not camera:
            result['error'] = "Failed to initialize camera"
            write_results(args.output, result)
            return 1
            
        # Give the camera time to warm up
        time.sleep(0.5)
        
        # Capture and analyze multiple frames for reliability
        max_attempts = 15
        start_time = time.time()
        
        for i in range(max_attempts):
            # Check timeout
            if time.time() - start_time > args.timeout:
                logger.warning(f"Recognition timed out after {args.timeout} seconds")
                result['error'] = f"Timeout after {args.timeout} seconds"
                break
                
            logger.info(f"Capturing frame {i+1}/{max_attempts}")
            frame_result = capture_and_analyze_frame(camera, recognition)
            
            if frame_result['success']:
                # If we found a face, update our result
                if frame_result['face_detected']:
                    result['face_detected'] = True
                    result['face_locations'] = frame_result['face_locations']
                    
                    # If we have encodings and liveness check passed (or skipped)
                    if frame_result['face_encodings'] and (frame_result['is_live'] or args.skip_liveness):
                        result['face_encodings'] = frame_result['face_encodings']
                        result['is_live'] = True if args.skip_liveness else frame_result['is_live']
                        result['liveness_metrics'] = frame_result.get('liveness_metrics', {})
                        
                        # If we found a match
                        if frame_result.get('match'):
                            result['match'] = frame_result['match']
                            logger.info(f"Face matched with {frame_result['match']['name']}")
                            
                        result['success'] = True
                        logger.info("Face detected and encoded successfully")
                        break
                    elif frame_result['face_encodings'] and not frame_result['is_live']:
                        # Face detected but failed liveness check
                        logger.warning("Face detected but failed liveness check")
                        result['is_live'] = False
                        result['liveness_metrics'] = frame_result.get('liveness_metrics', {})
                        
                        # Continue trying for a better frame
                        time.sleep(0.2)
                    else:
                        # Face detected but no encodings
                        time.sleep(0.2)
                else:
                    # No face detected in this frame
                    time.sleep(0.2)
            else:
                # Error in processing this frame
                logger.warning(f"Error in frame {i+1}: {frame_result.get('error')}")
                time.sleep(0.2)
        
        # If we went through all attempts without a successful recognition
        if not result.get('success'):
            # If we at least found a face
            if result['face_detected']:
                if not result.get('is_live'):
                    result['error'] = "Face failed liveness detection"
                elif not result.get('face_encodings'):
                    result['error'] = "Could not generate face encoding"
                else:
                    result['error'] = "Face detected but not recognized"
                    result['success'] = True  # Consider it a success even if not recognized
            else:
                result['error'] = "No face detected after multiple attempts"
    
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        result['error'] = f"Unhandled exception: {str(e)}"
    
    finally:
        # Clean up
        if camera:
            try:
                logger.info("Releasing camera resources")
                # Try to read any remaining frames to clear buffer
                for _ in range(5):
                    camera.read()
                camera.release()
                logger.info("Camera released successfully")
            except Exception as e:
                logger.error(f"Error releasing camera: {str(e)}")
                
        # Explicitly destroy all OpenCV windows
        try:
            cv2.destroyAllWindows()
            # Force window cleanup with multiple waitKey calls
            for _ in range(5):
                cv2.waitKey(1)
        except Exception as e:
            logger.error(f"Error destroying windows: {str(e)}")
        
        # Save results
        write_results(args.output, result)
            
    return 0 if result.get('success', False) else 1

def write_results(output_file, results):
    """Write results to the output file"""
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        
        # Write to output file
        with open(output_file, 'w') as f:
            json.dump(results, f)
        
        logger.info(f"Results written to {output_file}")
    except Exception as e:
        logger.error(f"Error writing results: {e}")
        logger.error(traceback.format_exc())

if __name__ == "__main__":
    sys.exit(main()) 