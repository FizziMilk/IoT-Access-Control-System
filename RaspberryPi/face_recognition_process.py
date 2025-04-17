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
from Web_Recognition.web_face_service import WebFaceService
from Web_Recognition.recognition import WebRecognition
from Web_Recognition.camera import WebCamera
from Web_Recognition.liveness_detection import LivenessDetector
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

def setup_camera():
    """Setup and initialize camera"""
    try:
        camera = WebCamera()
        camera.start()
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
            x, y, w, h = left, top, right-left, bottom-top
            
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

def save_debug_frame(frame, face_locations, liveness_results, match_names=None):
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
        
        # Save the frame
        upload_folder = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'uploads')
        if not os.path.exists(upload_folder):
            os.makedirs(upload_folder)
        
        debug_frame_path = os.path.join(upload_folder, 'debug_frame.jpg')
        cv2.imwrite(debug_frame_path, debug_frame)
    except Exception as e:
        logger.error(f"Error saving debug frame: {e}")

def run_face_recognition(output_file, backend_url, max_attempts=100, confidence_threshold=60):
    """Run the face recognition process"""
    try:
        camera = setup_camera()
        
        # Load known faces from backend
        known_face_encodings, known_face_names, known_face_ids = load_known_face_encodings(backend_url)
        
        if not known_face_encodings:
            logger.warning("No known faces available from backend")
        
        attempt_count = 0
        recognition_results = []
        
        while attempt_count < max_attempts:
            # Get frame from camera
            frame = camera.get_frame()
            
            if frame is None:
                logger.warning("No frame captured, trying again...")
                time.sleep(0.1)
                continue
            
            # Convert the frame from BGR to RGB (required by face_recognition)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            
            # No faces detected
            if not face_locations:
                logger.info("No faces detected in frame")
                save_debug_frame(frame, [], [])  # Save empty frame
                time.sleep(0.1)
                attempt_count += 1
                continue
            
            # Perform liveness detection
            liveness_results = perform_liveness_check(frame, face_locations)
            
            # Check if any face passes liveness check
            real_faces = [result for result in liveness_results if result["is_real"]]
            
            # If no real faces, continue to next frame
            if not real_faces:
                logger.info("No real faces detected, possible spoofing attempt")
                save_debug_frame(frame, face_locations, liveness_results)  # Save frame with fake faces
                time.sleep(0.1)
                attempt_count += 1
                continue
            
            # Get face encodings for recognized faces
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Initialize match results
            match_names = [None] * len(face_locations)
            matched_ids = []
            
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
                        best_match_index = np.argmin(distances)
                        
                        # Convert distance to confidence percentage (smaller distance = higher confidence)
                        confidence = (1 - distances[best_match_index]) * 100
                        
                        if confidence >= confidence_threshold:
                            match_names[i] = known_face_names[best_match_index]
                            matched_id = known_face_ids[best_match_index]
                            matched_ids.append(matched_id)
                            
                            logger.info(f"Match found: {known_face_names[best_match_index]} with {confidence:.2f}% confidence")
                            recognition_results.append({
                                "id": matched_id,
                                "name": known_face_names[best_match_index],
                                "confidence": float(confidence),
                                "timestamp": datetime.now().isoformat()
                            })
            
            # Save frame with detection results
            save_debug_frame(frame, face_locations, liveness_results, match_names)
            
            # If we found a match, exit the loop
            if matched_ids:
                break
            
            time.sleep(0.1)
            attempt_count += 1
        
        # Clean up
        camera.release()
        
        # Write results to output file
        with open(output_file, 'w') as f:
            json.dump({
                "success": len(recognition_results) > 0,
                "results": recognition_results,
                "attempts": attempt_count
            }, f)
        
        return True
    except Exception as e:
        logger.error(f"Error in face recognition process: {e}")
        logger.error(traceback.format_exc())
        
        # Write error to output file
        with open(output_file, 'w') as f:
            json.dump({
                "success": False,
                "error": str(e),
                "attempts": 0
            }, f)
        
        return False
    finally:
        # Make sure camera is released
        try:
            if 'camera' in locals():
                camera.release()
        except:
            pass

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
            encodings, names, ids = load_known_face_encodings(args.backend_url)
            if encodings:
                recognition.load_encodings(encodings, names)
                logger.info(f"Recognition running with {len(encodings)} known faces")
            else:
                logger.warning("No face encodings loaded from backend, running in detection-only mode")
        else:
            logger.warning("No backend URL provided, running in detection-only mode (no face matching)")
        
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
            frame_result = run_face_recognition(
                args.output,
                args.backend_url,
                max_attempts=max_attempts,
                confidence_threshold=60
            )
            
            if frame_result:
                # If we found a face, update our result
                result['face_detected'] = True
                result['face_locations'] = face_recognition.face_locations(cv2.cvtColor(camera.get_frame(), cv2.COLOR_BGR2RGB))
                
                # If we have encodings and liveness check passed (or skipped)
                if result['face_encodings'] and (result['is_live'] or args.skip_liveness):
                    result['face_encodings'] = result['face_encodings']
                    result['is_live'] = True if args.skip_liveness else result['is_live']
                    result['liveness_metrics'] = result.get('liveness_metrics', {})
                    
                    # If we found a match
                    if result.get('match'):
                        result['match'] = result['match']
                        logger.info(f"Face matched with {result['match']['name']}")
                        
                    result['success'] = True
                    logger.info("Face detected and encoded successfully")
                    break
                elif result['face_encodings'] and not result['is_live']:
                    # Face detected but failed liveness check
                    logger.warning("Face detected but failed liveness check")
                    result['is_live'] = False
                    result['liveness_metrics'] = result.get('liveness_metrics', {})
                    
                    # Continue trying for a better frame
                    time.sleep(0.2)
                else:
                    # Face detected but no encodings
                    time.sleep(0.2)
            else:
                # Error in processing this frame
                logger.warning(f"Error in frame {i+1}: {result.get('error')}")
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
        
        # Make sure all values are JSON serializable
        def convert_to_serializable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()  # Convert numpy scalars to Python types
            elif isinstance(obj, (bool, int, float, str, list, dict, tuple, type(None))):
                return obj
            else:
                return str(obj)  # Convert any other types to strings
        
        # Convert all values in the results dict
        serializable_results = {}
        for key, value in results.items():
            if isinstance(value, dict):
                # Handle nested dictionaries
                serializable_results[key] = {k: convert_to_serializable(v) for k, v in value.items()}
            else:
                serializable_results[key] = convert_to_serializable(value)
        
        # Write to output file using the serializable results
        with open(output_file, 'w') as f:
            json.dump(serializable_results, f)
        
        logger.info(f"Results written to {output_file}")
    except Exception as e:
        logger.error(f"Error writing results: {e}")
        logger.error(traceback.format_exc())
        
        # Emergency fallback - write a simple valid JSON with the error
        try:
            with open(output_file, 'w') as f:
                json.dump({"success": False, "error": f"JSON serialization error: {str(e)}"}, f)
            logger.info(f"Fallback results written to {output_file}")
        except Exception as e2:
            logger.error(f"Fallback writing also failed: {e2}")

if __name__ == "__main__":
    sys.exit(main()) 