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
# Remove unused imports
# from Web_Recognition.web_face_service import WebFaceService
# from Web_Recognition.camera import WebCamera
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

# Add WebRecognition class implementation here to replace the missing import
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
    try:
        camera = cv2.VideoCapture(0)
        if not camera.isOpened():
            logger.error("Failed to open camera")
            return None
            
        # Set camera properties if needed
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        # Give camera time to adjust
        time.sleep(0.3)
        return camera
    except Exception as e:
        logger.error(f"Error setting up camera: {str(e)}")
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

def save_debug_frame(frame, prefix, directory, face_locations=None, is_live=None, match=None):
    """
    Save a debug frame with optional visualizations for detections.
    
    Args:
        frame: OpenCV image frame
        prefix: Prefix for the filename
        directory: Directory to save the frame
        face_locations: Optional list of face locations
        is_live: Optional boolean indicating liveness
        match: Optional match information
    """
    if not directory:
        return None
        
    try:
        # Create a copy to avoid modifying the original
        debug_frame = frame.copy()
        
        # Draw face locations if provided
        if face_locations:
            for top, right, bottom, left in face_locations:
                # Draw a box around the face
                cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Add status text
                status_text = ""
                status_color = (255, 255, 255)
                
                if is_live is not None:
                    if is_live:
                        status_text = "LIVE"
                        status_color = (0, 255, 0)  # Green
                    else:
                        status_text = "FAKE"
                        status_color = (0, 0, 255)  # Red
                
                if match:
                    status_text += f" - {match['name']} ({match['confidence']:.2f})"
                
                if status_text:
                    # Draw a filled rectangle for text background
                    text_size = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                    cv2.rectangle(debug_frame, 
                                (left, top - 25), 
                                (left + text_size[0] + 10, top), 
                                (0, 0, 0), -1)
                    # Add text
                    cv2.putText(debug_frame, status_text, (left + 5, top - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Generate a unique filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{prefix}_{timestamp}.jpg"
        filepath = os.path.join(directory, filename)
        
        # Save the frame
        cv2.imwrite(filepath, debug_frame)
        logger.info(f"Saved debug frame to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error saving debug frame: {e}")
        logger.error(traceback.format_exc())
        return None

def capture_and_analyze_frame(camera, recognition, debug_frames=False, frames_dir=None):
    """
    Capture a frame and perform face recognition with liveness check
    
    Args:
        camera: OpenCV camera object
        recognition: WebRecognition object
        debug_frames: Whether to save debug frames
        frames_dir: Directory to save debug frames
        
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
        'error': None,
        'debug_frame': None
    }
    
    try:
        # Capture frame
        ret, frame = camera.read()
        if not ret or frame is None:
            result['error'] = "Failed to capture frame"
            return result
            
        # Save initial frame if debug enabled
        if debug_frames and frames_dir:
            save_debug_frame(frame, "initial", frames_dir)
            
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
            
            # Save detected face if debug enabled
            if debug_frames and frames_dir:
                save_debug_frame(frame, "detected_face", frames_dir, face_locations)
            
            # Perform liveness check on face region
            if face_image.size > 0:  # Ensure face_image is not empty
                liveness_result = perform_liveness_check(face_image)
                result['is_live'] = liveness_result['is_live']
                result['liveness_metrics'] = {
                    k: v for k, v in liveness_result.items() if k != 'is_live'
                }
                
                # Save liveness result frame if debug enabled
                if debug_frames and frames_dir:
                    save_debug_frame(frame, "liveness_check", frames_dir, face_locations, result['is_live'])
                
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
                            
                            # Save match result frame if debug enabled
                            if debug_frames and frames_dir:
                                result['debug_frame'] = save_debug_frame(
                                    frame, "match_result", frames_dir, 
                                    face_locations, result['is_live'], match
                                )
            else:
                logger.warning("Empty face image extracted, skipping liveness check")
        else:
            # Save no-face frame if debug enabled
            if debug_frames and frames_dir:
                save_debug_frame(frame, "no_face", frames_dir)
            
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
    # Add the missing arguments
    parser.add_argument('--debug-frames', action='store_true', help='Save debug frames during processing')
    parser.add_argument('--frames-dir', help='Directory to store debug frames')
    parser.add_argument('--upload-folder', help='Upload folder path')
    
    args = parser.parse_args()
    
    camera = None
    result = {
        'success': False,
        'error': None,
        'face_detected': False,
        'is_live': False,
        'debug_frame': None  # Add field for debug frame path
    }
    
    # Create debug frames directory if enabled
    if args.debug_frames and args.frames_dir:
        os.makedirs(args.frames_dir, exist_ok=True)
    
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
            frame_result = capture_and_analyze_frame(camera, recognition, args.debug_frames, args.frames_dir)
            
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
                        
                        # Include debug frame path in result if available
                        if frame_result.get('debug_frame'):
                            result['debug_frame'] = frame_result['debug_frame']
                            
                        result['success'] = True
                        logger.info("Face detected and encoded successfully")
                        break
                    elif frame_result['face_encodings'] and not frame_result['is_live']:
                        # Face detected but failed liveness check
                        logger.warning("Face detected but failed liveness check")
                        result['is_live'] = False
                        result['liveness_metrics'] = frame_result.get('liveness_metrics', {})
                        result['debug_frame'] = frame_result.get('debug_frame')
                        
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
                
                # Create a final debug frame for no face detected
                if args.debug_frames and args.frames_dir and camera:
                    ret, frame = camera.read()
                    if ret:
                        result['debug_frame'] = save_debug_frame(
                            frame, "final_no_face", args.frames_dir
                        )
    
    except Exception as e:
        logger.error(f"Unhandled exception: {str(e)}")
        logger.error(traceback.format_exc())
        result['error'] = f"Unhandled exception: {str(e)}"
    
    finally:
        # Clean up
        if camera:
            camera.release()
        cv2.destroyAllWindows()
        
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

class WebCamera:
    """Simplified camera wrapper for face recognition process."""
    
    def __init__(self, camera_id=0, resolution=(640, 480)):
        """Initialize the camera."""
        self.camera_id = camera_id
        self.width, self.height = resolution
        self.cap = None
        self.is_running = False
        self.logger = logging.getLogger("WebCamera")
        
        # Try different camera indices to find a working one
        self.start()
    
    def start(self):
        """Start the camera. Try multiple indices if necessary."""
        if self.is_running:
            return
        
        # First try the specified camera ID
        self._try_camera(self.camera_id)
        
        # If that failed, try other common camera indices
        if not self.is_running:
            alt_indices = [1, 2, -1]  # -1 is a special value that tries to use any available camera
            for idx in alt_indices:
                if idx != self.camera_id:
                    self.logger.info(f"Trying alternate camera index: {idx}")
                    if self._try_camera(idx):
                        self.camera_id = idx
                        break
        
        # If we still couldn't open a camera, handle the failure more gracefully
        if not self.is_running:
            self.logger.error("Failed to open any camera")
            
            # Instead of raising an exception, create a fake camera with a placeholder frame
            self._create_fake_camera()
            
    def _try_camera(self, camera_id):
        """Attempt to open the camera with the given ID."""
        try:
            self.logger.info(f"Starting camera (ID: {camera_id})")
            
            # Try with different API backends
            for api_preference in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_V4L2]:
                try:
                    # Don't use V4L2 on Windows
                    if api_preference == cv2.CAP_V4L2 and os.name == 'nt':
                        continue
                        
                    self.logger.info(f"Trying camera {camera_id} with API: {api_preference}")
                    self.cap = cv2.VideoCapture(camera_id, api_preference)
                    
                    if self.cap.isOpened():
                        # Set resolution
                        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                        
                        # Set parameters to reduce latency
                        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        
                        # Check if camera works by reading a frame
                        ret, frame = self.cap.read()
                        if ret and frame is not None:
                            self.is_running = True
                            self.logger.info(f"Camera {camera_id} started successfully with API: {api_preference}")
                            return True
                        else:
                            self.logger.warning(f"Camera {camera_id} opened but failed to read frame, trying next API")
                            self.cap.release()
                except Exception as e:
                    self.logger.warning(f"Failed to open camera {camera_id} with API {api_preference}: {e}")
            
            self.logger.warning(f"Failed to open camera {camera_id} with any API")
            return False
        except Exception as e:
            self.logger.error(f"Error starting camera {camera_id}: {e}")
            return False
    
    def _create_fake_camera(self):
        """Create a fake camera that returns a placeholder frame."""
        self.logger.warning("Creating a fake camera with placeholder images")
        self.is_running = True
        self.is_fake = True
        
        # Create a simple placeholder frame
        self.placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        
        # Add a message to the frame
        cv2.putText(self.placeholder_frame, "Camera Not Available", (120, 240),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(self.placeholder_frame, "Using simulated camera", (120, 280),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def get_frame(self):
        """Get a frame from the camera."""
        if not self.is_running:
            self.start()
        
        # For fake camera, return the placeholder frame
        if hasattr(self, 'is_fake') and self.is_fake:
            # Add a timestamp to simulate a live camera
            timestamp = datetime.now().strftime("%H:%M:%S")
            frame = self.placeholder_frame.copy()
            cv2.putText(frame, timestamp, (10, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            return frame
            
        # For real camera, get a frame
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.logger.warning("Failed to capture frame")
                # Try to restart the camera if we failed to get a frame
                self.is_running = False
                self.start()
                
                # If restarting didn't help, return placeholder
                if hasattr(self, 'placeholder_frame'):
                    return self.placeholder_frame
                return None
                
            return frame
        
        return None
    
    def read(self):
        """Compatibility method to match OpenCV's camera.read() interface."""
        frame = self.get_frame()
        return (frame is not None), frame
    
    def release(self):
        """Release the camera resources."""
        if self.cap and self.is_running and not hasattr(self, 'is_fake'):
            self.logger.info("Releasing camera resources")
            self.cap.release()
            self.is_running = False

if __name__ == "__main__":
    sys.exit(main()) 