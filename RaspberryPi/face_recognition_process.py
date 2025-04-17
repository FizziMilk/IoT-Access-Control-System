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
        
        self.is_running = True
        self.logger.info("Camera started successfully")
    
    def get_frame(self):
        """Get a frame from the camera."""
        if not self.is_running:
            self.start()
            
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
            self.is_running = False


def setup_camera():
    """Setup and initialize camera"""
    try:
        camera = WebCamera()
        return camera
    except Exception as e:
        logger.error(f"Error setting up camera: {e}")
        raise

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

def save_debug_frame(frame, face_locations, liveness_results, match_names=None, upload_folder=None):
    """Save a debug frame with detection visualizations"""
    try:
        debug_frame = frame.copy()
        
        # Draw face locations and liveness results
        for i, face_location in enumerate(face_locations):
            top, right, bottom, left = face_location
            
            # Get liveness result if available
            is_real = False
            score = 0
            if i < len(liveness_results):
                is_real = liveness_results[i]["is_real"]
                score = liveness_results[i]["score"]
            
            # Draw face rectangle
            if is_real:
                color = (0, 255, 0)  # Green for real
            else:
                color = (0, 0, 255)  # Red for fake
            
            cv2.rectangle(debug_frame, (left, top), (right, bottom), color, 2)
            
            # Add liveness score text
            liveness_text = f"Real: {is_real}, Score: {score:.2f}"
            cv2.putText(debug_frame, liveness_text, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Add person name if matched
            if match_names and i < len(match_names) and match_names[i]:
                name_text = f"Name: {match_names[i]}"
                cv2.putText(debug_frame, name_text, (left, bottom + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # If no upload folder specified, use a default
        if not upload_folder:
            # Try to get the upload folder from command line arguments
            parser = argparse.ArgumentParser()
            parser.add_argument('--upload-folder', help='Flask upload folder for debug frames')
            args, _ = parser.parse_known_args()
            
            if args.upload_folder:
                upload_folder = args.upload_folder
            else:
                # Fall back to default location
                upload_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
        
        # Ensure upload folder exists
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder, exist_ok=True)
            logger.info(f"Created debug frame directory: {upload_folder}")
        
        # Use a safer approach to write the frame
        debug_frame_path = os.path.join(upload_folder, 'debug_frame.jpg')
        temp_frame_path = os.path.join(upload_folder, f'debug_frame_temp_{os.getpid()}.jpg')
        
        # First encode the image in memory
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]  
        _, buffer = cv2.imencode('.jpg', debug_frame, encode_params)
        
        # Write the buffer directly to a file
        with open(temp_frame_path, 'wb') as f:
            f.write(buffer)
            f.flush()
            os.fsync(f.fileno())  # Ensure file is written to disk
            
        # On Windows, we need to remove the destination file first
        if os.name == 'nt' and os.path.exists(debug_frame_path):
            try:
                os.remove(debug_frame_path)
            except Exception as e:
                logger.warning(f"Failed to remove existing debug frame on Windows: {e}")
                
        # Rename the temporary file to the final filename
        try:
            os.rename(temp_frame_path, debug_frame_path)
        except Exception as e:
            logger.warning(f"Failed to rename debug frame: {e}")
            # If rename fails (which can happen on Windows), try copy and delete
            import shutil
            try:
                shutil.copy2(temp_frame_path, debug_frame_path)
                os.remove(temp_frame_path)
            except Exception as copy_err:
                logger.error(f"Failed to copy debug frame: {copy_err}")
                
    except Exception as e:
        logger.error(f"Error saving debug frame: {e}")
        logger.error(traceback.format_exc())

def run_face_recognition(output_file, backend_url, max_attempts=100, confidence_threshold=60, upload_folder=None):
    """Run the face recognition process using frames captured by the web video feed"""
    try:
        # Frame sharing approach - use frames from the video feed instead of opening camera directly
        frame_share_dir = '/tmp/camera_frames'
        os.makedirs(frame_share_dir, exist_ok=True)
        frame_path = os.path.join(frame_share_dir, 'current_frame.jpg')
        
        # Wait for the first frame to be available
        retries = 0
        while not os.path.exists(frame_path) and retries < 10:
            logger.info(f"Waiting for shared camera frame ({retries}/10)...")
            time.sleep(0.5)
            retries += 1
                
        # Load known faces from backend
        known_face_encodings, known_face_names, known_face_ids = load_known_face_encodings(backend_url)
        
        if not known_face_encodings:
            logger.warning("No known faces available from backend")
        
        attempt_count = 0
        recognition_results = []
        last_liveness_check_time = time.time()
        liveness_check_interval = 0.3  # Check liveness more frequently for better accuracy
        
        # For tracking successful detections
        consecutive_real_faces = 0
        required_consecutive_detections = 3  # Require 3 consecutive real face detections for better accuracy
        best_match_count = 0
        best_match_index = -1
        best_confidence = 0
        
        # For debugging - create a frame with instructions
        if upload_folder:
            instruction_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(instruction_frame, "Face Recognition Active", (140, 200), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(instruction_frame, "Please look at the camera", (140, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            save_debug_frame(instruction_frame, [], [], upload_folder=upload_folder)
        
        while attempt_count < max_attempts:
            # Read frame from shared location
            try:
                # Check if the file exists and has content
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    # Read the frame safely
                    with open(frame_path, 'rb') as f:
                        img_data = f.read()
                    
                    # Decode the image
                    img_array = np.frombuffer(img_data, dtype=np.uint8)
                    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
                    
                    if frame is None or frame.size == 0:
                        logger.warning("Failed to decode shared frame")
                        time.sleep(0.1)
                        continue
                else:
                    logger.warning("Shared frame not available yet")
                    time.sleep(0.1)
                    continue
            except Exception as e:
                logger.warning(f"Error reading shared frame: {e}")
                time.sleep(0.1)
                continue
            
            # Always save a frame with current status to show progress
            status_frame = frame.copy()
            progress_percent = int((attempt_count/max_attempts)*100)
            cv2.putText(status_frame, f"Processing: {progress_percent}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Convert the frame from BGR to RGB (required by face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # No faces detected
            if not face_locations:
                # Save frame with "No face detected" message
                cv2.putText(status_frame, "No face detected", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                save_debug_frame(status_frame, [], [], upload_folder=upload_folder)
                time.sleep(0.05)
                attempt_count += 1
                consecutive_real_faces = 0  # Reset counter when no face is detected
                continue
            
            # Always show where faces are detected
            liveness_results = []
            
            # Get the current time for liveness check frequency
            current_time = time.time()
            
            # Only perform full liveness check periodically to improve framerate
            perform_full_check = (current_time - last_liveness_check_time >= liveness_check_interval)
            
            if perform_full_check:
                # Perform liveness detection
                liveness_results = perform_liveness_check(frame, face_locations)
                last_liveness_check_time = current_time
                
                # Check if any face passes liveness check
                real_faces = [result for result in liveness_results if result["is_real"]]
                
                # If no real faces, continue to next frame
                if not real_faces:
                    # Mark all faces as not real
                    cv2.putText(status_frame, "Possible spoofing detected", (10, 60),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    logger.info("No real faces detected, possible spoofing attempt")
                    consecutive_real_faces = 0  # Reset counter
                    save_debug_frame(status_frame, face_locations, liveness_results, upload_folder=upload_folder)
                    time.sleep(0.05)
                    attempt_count += 1
                    continue
                else:
                    # Increment consecutive real face counter
                    consecutive_real_faces += 1
                    
                    # Set status message
                    cv2.putText(status_frame, f"Real face detected ({consecutive_real_faces}/{required_consecutive_detections})", 
                               (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                # Get face encodings for recognized faces
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
                
                # Initialize match results
                match_names = [None] * len(face_locations)
                
                # Check each face against known faces
                for i, face_encoding in enumerate(face_encodings):
                    # Skip if this face failed liveness
                    if i >= len(liveness_results) or not liveness_results[i]["is_real"]:
                        continue
                    
                    if known_face_encodings:
                        # Compare face with known faces
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        
                        if True in matches:
                            # Find the best match
                            match_index = np.argmin(distances)
                            
                            # Convert distance to confidence percentage (smaller distance = higher confidence)
                            confidence = (1 - distances[match_index]) * 100
                            
                            if confidence >= confidence_threshold:
                                match_names[i] = known_face_names[match_index]
                                matched_id = known_face_ids[match_index]
                                
                                # Count matches for this person
                                if match_index == best_match_index:
                                    best_match_count += 1
                                    if confidence > best_confidence:
                                        best_confidence = confidence
                                else:
                                    # Track if this is a better match
                                    if best_match_count < 1 or confidence > best_confidence:
                                        best_match_index = match_index
                                        best_match_count = 1
                                        best_confidence = confidence
                                
                                # Add matching information to the frame
                                cv2.putText(status_frame, f"Match: {known_face_names[match_index]}", (10, 90),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                cv2.putText(status_frame, f"Confidence: {confidence:.1f}%", (10, 120),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                                
                                logger.info(f"Match found: {known_face_names[match_index]} with {confidence:.2f}% confidence")
                                
                                # If we have enough consecutive real faces and matches, exit early with success
                                if consecutive_real_faces >= required_consecutive_detections and best_match_count >= 2:
                                    logger.info(f"Found reliable match after {attempt_count} attempts")
                                    recognition_results.append({
                                        "id": matched_id,
                                        "name": known_face_names[match_index],
                                        "confidence": float(confidence),
                                        "timestamp": datetime.now().isoformat()
                                    })
                                    break
                
                # Save frame with detection results
                save_debug_frame(status_frame, face_locations, liveness_results, match_names, upload_folder=upload_folder)
            else:
                # Even when not doing full check, still show face locations
                placeholder_liveness = [{"is_real": True, "score": 0.5, "face_location": loc, "debug_info": {}} for loc in face_locations]
                save_debug_frame(status_frame, face_locations, placeholder_liveness, upload_folder=upload_folder)
            
            # If we have a strong match, exit the loop
            if consecutive_real_faces >= required_consecutive_detections and best_match_count >= 2:
                if best_match_index >= 0:
                    matched_id = known_face_ids[best_match_index]
                    matched_name = known_face_names[best_match_index]
                    
                    if not any(result["id"] == matched_id for result in recognition_results):
                        recognition_results.append({
                            "id": matched_id,
                            "name": matched_name,
                            "confidence": float(best_confidence),
                            "timestamp": datetime.now().isoformat()
                        })
                
                # Show success screen before exiting
                if upload_folder:
                    success_frame = frame.copy()
                    cv2.putText(success_frame, "Recognition Successful!", (120, 200),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if best_match_index >= 0:
                        cv2.putText(success_frame, f"Welcome, {known_face_names[best_match_index]}", (120, 240),
                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    save_debug_frame(success_frame, [], [], upload_folder=upload_folder)
                
                break
            
            time.sleep(0.05)  # Shorter sleep for better framerate
            attempt_count += 1
        
        # If we exited without a match, show a failure screen
        if len(recognition_results) == 0 and upload_folder:
            failure_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
            if consecutive_real_faces > 0:
                cv2.putText(failure_frame, "Face detected but not recognized", (100, 200),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            else:
                cv2.putText(failure_frame, "No valid face detected", (140, 200),
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(failure_frame, "Please try again", (180, 240),
                      cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            save_debug_frame(failure_frame, [], [], upload_folder=upload_folder)
        
        # Write results to output file
        with open(output_file, 'w') as f:
            json.dump({
                "success": len(recognition_results) > 0,
                "results": recognition_results,
                "attempts": attempt_count,
                "face_detected": consecutive_real_faces > 0,
                "is_live": consecutive_real_faces >= required_consecutive_detections
            }, f)
            
        return len(recognition_results) > 0
            
    except Exception as e:
        logger.error(f"Error in face recognition: {e}")
        logger.error(traceback.format_exc())
        
        # Create error frame if possible
        if 'upload_folder' in locals() and upload_folder:
            try:
                error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(error_frame, "Error during recognition", (140, 200),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(error_frame, str(e)[:40], (100, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                save_debug_frame(error_frame, [], [], upload_folder=upload_folder)
            except Exception as e2:
                logger.error(f"Error creating error frame: {e2}")
        
        # Write error to output file
        with open(output_file, 'w') as f:
            json.dump({
                "success": False,
                "error": str(e),
                "attempts": attempt_count if 'attempt_count' in locals() else 0
            }, f)
            
        return False

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

if __name__ == "__main__":
    sys.exit(main()) 