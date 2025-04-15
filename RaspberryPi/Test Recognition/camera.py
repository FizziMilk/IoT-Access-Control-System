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
import math
from imutils import face_utils

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
LOGGER = logging.getLogger("CameraSystem")

class CameraSystem:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
        # EAR threshold for blink detection - less sensitive (harder to register)
        self.EAR_THRESHOLD = 0.20  # Higher threshold makes it harder to detect blinks
        # Number of consecutive frames the eye must be below threshold to count as a blink
        self.EAR_CONSEC_FRAMES = 1  # Detect blinks in a single frame
        # Use a much lower resolution for liveness detection
        self.liveness_resolution = (320, 240)
        # Frame process rate (1 = process every frame, 2 = every other frame, etc.)
        self.process_nth_frame = 3
        # Use a separate thread for face detection to prevent UI blocking
        self.face_detection_thread = None
        self.stop_detection_thread = False
        # Flag to track if liveness passed has been logged
        self.has_logged_liveness_passed = False
        # Autofocus liveness detection
        self.focus_check_passed = False
        self.use_focus_check = True  # Enable focus-based liveness detection
    
    def capture_face(self):
        """Capture an image from the camera with face detection"""
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
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
    
    def perform_focus_liveness_check(self, cap, face_location):
        """
        Use autofocus manipulation to detect if a face is real or a flat image.
        
        Args:
            cap: OpenCV video capture
            face_location: Detected face coordinates (top, right, bottom, left)
            
        Returns:
            bool: True if the face is likely real, False if possibly a photo
        """
        # Store original settings to restore at the end
        original_autofocus = None
        original_focus = None
        
        try:
            if not self.use_focus_check:
                return True
                
            print("Performing focus-based liveness check...")
            
            # Verify face_location is valid
            if face_location is None or len(face_location) != 4:
                print("ERROR: Invalid face location")
                return True
                
            # Store current focus settings to restore later
            original_autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            original_focus = cap.get(cv2.CAP_PROP_FOCUS)
            
            # Save current cap to self.cap for the texture check
            self.cap = cap
            self.current_face_location = face_location
            
            # Run enhanced liveness checks
            focus_result = self.check_focus_depth(cap, face_location)
            texture_result, texture_confidence = self.check_facial_texture()
            
            # Combine results - require both checks to pass for real face
            is_real_face = focus_result and texture_result
            
            # Display final result
            ret, frame = cap.read()
            if ret:
                summary_frame = frame.copy()
                
                # Display final result
                result_text = f"PASS: Real Face" if is_real_face else f"FAIL: Possible Spoof"
                color = (0, 255, 0) if is_real_face else (0, 0, 255)
                
                top, right, bottom, left = face_location
                center_x = left + int((right - left) / 2)
                
                cv2.putText(summary_frame, result_text, (center_x - 150, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Show focus test result
                focus_status = "PASS" if focus_result else "FAIL"
                focus_color = (0, 255, 0) if focus_result else (0, 0, 255)
                cv2.putText(summary_frame, f"Focus Test: {focus_status}", (20, 150),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, focus_color, 2)
                
                # Show texture test result
                texture_status = "PASS" if texture_result else "FAIL"
                texture_color = (0, 255, 0) if texture_result else (0, 0, 255)
                cv2.putText(summary_frame, f"Texture Test: {texture_status} ({int(texture_confidence*100)}%)", (20, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, texture_color, 2)
                
                # Add instruction to press any key to continue
                cv2.putText(summary_frame, "Press any key to continue...", 
                           (20, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Focus Test Result", summary_frame)
                cv2.imwrite("focus_test_result.jpg", summary_frame)
                cv2.waitKey(0)  # Wait for user to press any key before continuing
            
            return is_real_face
            
        except Exception as e:
            print(f"Error in focus liveness check: {e}")
            print(traceback.format_exc())
            return True  # Default to allowing the user through if focus check fails
        finally:
            # Restore original camera settings if they were saved
            if original_autofocus is not None and original_focus is not None:
                try:
                    print("Restoring original camera settings")
                    cap.set(cv2.CAP_PROP_AUTOFOCUS, original_autofocus)
                    cap.set(cv2.CAP_PROP_FOCUS, original_focus)
                except Exception as e:
                    print(f"Error restoring camera settings: {e}")
    
    def check_focus_depth(self, cap, face_location):
        """
        Check if the face has proper 3D depth based on focus changes.
        Returns:
            tuple: (is_real_face, confidence_score)
        """
        try:
            # Extract face region for analysis
            top, right, bottom, left = face_location
            face_height = bottom - top
            face_width = right - left
            
            print(f"Face dimensions for focus test: {face_width}x{face_height}")
            
            # Define focus test points with better separation
            # Use eye region vs nose region (should have different depths)
            eye_y = top + int(face_height * 0.25)  # Higher on face
            nose_y = top + int(face_height * 0.55)  # Lower on face - increased separation
            center_x = left + int((right - left) / 2)
            
            # Points to analyze - ensure all coordinates are integers
            eye_point = (int(center_x), int(eye_y))
            nose_point = (int(center_x), int(nose_y))
            
            # Add background point with better separation
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            bg_offset = int(max(100, face_width * 0.75))  # Ensure more distance from face
            bg_x = max(30, left - bg_offset)  # Point far to the left of face
            if bg_x < 30:  # If face too close to left edge, use right side
                bg_x = min(frame_width - 30, right + bg_offset)
            bg_point = (int(bg_x), int(eye_y))
            
            print(f"Focus test points: Eye={eye_point}, Nose={nose_point}, Background={bg_point}")
            
            # Test with more extreme focus levels
            focus_levels = [0, 100, 250]  # More distributed focus values
            clarity_values = []
            
            # Disable autofocus for manual control - be more aggressive
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            time.sleep(1.0)  # Give more time for autofocus to turn off
            
            # Store diagnostic images
            diagnostic_images = []
            
            for focus in focus_levels:
                # Set focus
                print(f"Setting focus to {focus}")
                cap.set(cv2.CAP_PROP_FOCUS, focus)
                
                # Give more time for focus to adjust
                time.sleep(1.5)  # Increased from 1.0 to 1.5
                
                # Verify focus actually changed
                current_focus = cap.get(cv2.CAP_PROP_FOCUS)
                print(f"Actual focus: {current_focus}")
                
                # Capture multiple frames and use the last one to allow focus to stabilize
                frames = []
                for _ in range(7):  # Increased from 5 to 7
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                    time.sleep(0.2)  # Increased time between frames
                
                if not frames:
                    print("Failed to capture frames")
                    continue
                
                # Use last captured frame
                frame = frames[-1]
                
                # Convert to grayscale for clarity analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate local clarity at test points (using Laplacian variance)
                eye_clarity = self.calculate_clarity(gray, eye_point, region_size=30)
                nose_clarity = self.calculate_clarity(gray, nose_point, region_size=30)
                bg_clarity = self.calculate_clarity(gray, bg_point, region_size=30)
                
                print(f"Clarity values - Eye: {eye_clarity:.2f}, Nose: {nose_clarity:.2f}, BG: {bg_clarity:.2f}")
                
                # Store values for analysis
                clarity_values.append((focus, eye_clarity, nose_clarity, bg_clarity))
                
                # Create diagnostic visualization
                debug_frame = frame.copy()
                
                # Highlight test regions with rectangles
                region_size = 30
                cv2.rectangle(debug_frame, 
                             (eye_point[0]-region_size//2, eye_point[1]-region_size//2),
                             (eye_point[0]+region_size//2, eye_point[1]+region_size//2),
                             (0, 255, 0), 2)
                cv2.rectangle(debug_frame, 
                             (nose_point[0]-region_size//2, nose_point[1]-region_size//2),
                             (nose_point[0]+region_size//2, nose_point[1]+region_size//2),
                             (0, 0, 255), 2)
                cv2.rectangle(debug_frame, 
                             (bg_point[0]-region_size//2, bg_point[1]-region_size//2),
                             (bg_point[0]+region_size//2, bg_point[1]+region_size//2),
                             (255, 0, 0), 2)
                
                # Add clarity values to the image
                cv2.putText(debug_frame, f"Eye: {eye_clarity:.2f}", (20, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(debug_frame, f"Nose: {nose_clarity:.2f}", (20, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                cv2.putText(debug_frame, f"BG: {bg_clarity:.2f}", (20, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
                cv2.putText(debug_frame, f"Focus: {focus} (actual: {current_focus:.1f})", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Store debug frame
                diagnostic_images.append(debug_frame.copy())
                
                # Show the diagnostic image
                cv2.imshow(f"Focus Test - Level {focus}", debug_frame)
                cv2.waitKey(1)
                
                # Save diagnostic image
                cv2.imwrite(f"focus_test_{focus}.jpg", debug_frame)
            
            # Calculate differences in how regions respond to focus
            if len(clarity_values) < 3:
                print("Not enough clarity values to analyze")
                return False, 0.0
                
            # Check if focus changes had any significant effect (to detect if focus control is working)
            eye_clarity_values = [c[1] for c in clarity_values]
            nose_clarity_values = [c[2] for c in clarity_values]
            bg_clarity_values = [c[3] for c in clarity_values]
            
            # Calculate range of clarity values
            eye_range = max(eye_clarity_values) - min(eye_clarity_values)
            nose_range = max(nose_clarity_values) - min(nose_clarity_values)
            bg_range = max(bg_clarity_values) - min(bg_clarity_values)
            
            print(f"Clarity ranges - Eye: {eye_range:.2f}, Nose: {nose_range:.2f}, BG: {bg_range:.2f}")
            
            # If no region shows significant focus change, the test won't work
            if max(eye_range, nose_range, bg_range) < 5.0:
                print("WARNING: Focus changes had minimal effect on image clarity")
                print("Focus control might not be working properly")
                return True, 0.3  # Default to allow with low confidence
                
            # Calculate correlations
            eye_nose_correlation = np.corrcoef(eye_clarity_values, nose_clarity_values)[0, 1]
            eye_bg_correlation = np.corrcoef(eye_clarity_values, bg_clarity_values)[0, 1]
            nose_bg_correlation = np.corrcoef(nose_clarity_values, bg_clarity_values)[0, 1]
            
            print(f"\nCorrelation - Eye-Nose: {eye_nose_correlation:.3f}, Eye-BG: {eye_bg_correlation:.3f}, Nose-BG: {nose_bg_correlation:.3f}")
            
            # Calculate differences between correlations
            eye_nose_bg_diff = abs(eye_nose_correlation - eye_bg_correlation)
            print(f"Eye-Nose vs Eye-BG difference: {eye_nose_bg_diff:.3f}")
            
            # Create a collage of all test images
            if len(diagnostic_images) >= 3:
                h, w = diagnostic_images[0].shape[:2]
                collage = np.zeros((h, w*3, 3), dtype=np.uint8)
                for i, img in enumerate(diagnostic_images[:3]):
                    collage[0:h, i*w:(i+1)*w] = img
                
                # Resize for display
                display_collage = cv2.resize(collage, (0, 0), fx=0.7, fy=0.7)
                cv2.imshow("Focus Test Collage", display_collage)
                cv2.imwrite("focus_test_collage.jpg", collage)
            
            # Create summary image with results
            summary_frame = diagnostic_images[-1].copy()
            
            # Adjusted criteria with reduced threshold for real-world conditions
            # 1. The face is in focus (at least one focus setting has good clarity)
            # 2. There should be some measurable difference in correlations
            correlation_diff = max(
                abs(eye_nose_correlation - eye_bg_correlation),
                abs(nose_bg_correlation - eye_bg_correlation)
            )
            both_highly_positive = (
                eye_nose_correlation > 0.95 and 
                eye_bg_correlation > 0.95 and
                nose_bg_correlation > 0.95
            )
            
            # Real face passes if there's some difference in correlations or not all highly positive
            is_real_face = correlation_diff > 0.1 or not both_highly_positive
            
            # Calculate confidence score (0.0-1.0)
            confidence = 0.0
            if is_real_face:
                # Calculate confidence based on correlation difference (bigger difference = higher confidence)
                diff_confidence = min(correlation_diff / 0.3, 1.0) * 0.7  # 0.3 is an ideal diff value
                
                # Add extra confidence if clarity ranges show good depth sensitivity
                depth_sensitivity = max(eye_range, nose_range) / max(1.0, bg_range)
                depth_confidence = min(depth_sensitivity / 2.0, 1.0) * 0.3
                
                confidence = diff_confidence + depth_confidence
            else:
                # If it's a photo, confidence is based on how close the correlations are
                if both_highly_positive:
                    confidence = 0.7 + 0.3 * min(1.0, (eye_nose_correlation + eye_bg_correlation + nose_bg_correlation) / 3.0 - 0.95) / 0.05
                else:
                    confidence = max(0.5, 1.0 - correlation_diff)
                    
                # Inverse the confidence since this indicates confidence it's NOT real
                confidence = 1.0 - confidence
            
            # Add results to summary frame
            result_text = "REAL FACE" if is_real_face else "POSSIBLE SPOOF"
            result_color = (0, 255, 0) if is_real_face else (0, 0, 255)
            cv2.putText(summary_frame, result_text, (20, 180), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
            
            cv2.putText(summary_frame, f"Eye-Nose Corr: {eye_nose_correlation:.3f}", (20, 210),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(summary_frame, f"Eye-BG Corr: {eye_bg_correlation:.3f}", (20, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(summary_frame, f"Nose-BG Corr: {nose_bg_correlation:.3f}", (20, 270),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(summary_frame, f"Difference: {correlation_diff:.3f}", (20, 300),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(summary_frame, f"Confidence: {confidence:.2f}", (20, 330),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.imshow("Focus Test Summary", summary_frame)
            cv2.imwrite("focus_test_summary.jpg", summary_frame)
            
            return is_real_face
            
        except Exception as e:
            print(f"Error in focus depth check: {e}")
            traceback.print_exc()
            return False
    
    def check_facial_texture(self):
        """
        Perform facial texture analysis to detect if face is real or a photo
        Returns:
            tuple: (is_real_face, confidence_score)
        """
        print("\n--- Starting Facial Texture Analysis ---")
        
        # Initialize face detector and shape predictor if not already done
        if not hasattr(self, 'face_detector'):
            self.face_detector = dlib.get_frontal_face_detector()
        if not hasattr(self, 'shape_predictor'):
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                      "shape_predictor_68_face_landmarks.dat")
            if not os.path.exists(model_path):
                print(f"ERROR: Could not find shape predictor model at {model_path}")
                return False, 0.0
            self.shape_predictor = dlib.shape_predictor(model_path)
        
        # If this was called from perform_focus_liveness_check, use the saved cap and face_location
        use_existing_face = hasattr(self, 'current_face_location') and self.current_face_location is not None
        
        # Save current camera settings
        if not hasattr(self, 'cap') or self.cap is None:
            # If no camera object available, create one
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        original_resolution = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Original resolution: {original_resolution}")
        
        # Set to higher resolution for better texture analysis
        target_resolution = (1920, 1080)
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_resolution[1])
            current_res = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            print(f"Camera set to: {current_res}")
        except Exception as e:
            print(f"Failed to set resolution: {str(e)}")
        
        # Calculate resolution scale factor compared to 720p reference
        # This is used to adjust thresholds based on the actual resolution
        reference_res = 1280 * 720
        actual_res = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) * self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        scale_factor = math.sqrt(actual_res / reference_res)
        print(f"Resolution scale factor: {scale_factor:.2f}")
        
        # Number of frames to analyze for texture
        num_frames = 3
        frames = []
        
        # Prepare threshold values for different metrics
        # These baseline values work well at 720p resolution
        # Real skin typically has moderate gradients and high-frequency details
        # Photos may be very smooth or have sharp, artificial gradients
        
        # Baseline thresholds for 720p resolution
        gradient_min_threshold = 15  # Minimum average gradient (was 20)
        gradient_max_threshold = 180  # Maximum average gradient (was 150)
        std_min_threshold = 10       # Minimum standard deviation (was 15)
        std_max_threshold = 120      # Maximum standard deviation (was 100)
        detail_energy_threshold = 0.45  # Minimum high-frequency energy (was 0.6)
        
        # Scale thresholds based on resolution
        # Linear scaling for gradient and std thresholds
        gradient_min_threshold *= scale_factor * 0.65  # Reduce scaling impact
        gradient_max_threshold *= scale_factor * 0.65  # Reduce scaling impact
        std_min_threshold *= scale_factor * 0.65       # Reduce scaling impact
        std_max_threshold *= scale_factor * 0.65       # Reduce scaling impact
        
        # Energy threshold doesn't scale linearly with resolution
        # For higher resolution, we need a lower threshold since we have more detail
        detail_energy_threshold /= (scale_factor * 0.5)
        
        print(f"Adjusted thresholds for current resolution:")
        print(f"  Gradient range: {gradient_min_threshold:.2f}-{gradient_max_threshold:.2f}")
        print(f"  Std dev range: {std_min_threshold:.2f}-{std_max_threshold:.2f}")
        print(f"  Detail energy threshold: {detail_energy_threshold:.2f}")
        
        # If we have a pre-existing face location, use it to create a dlib rectangle
        face_rect = None
        if use_existing_face:
            top, right, bottom, left = self.current_face_location
            face_rect = dlib.rectangle(left, top, right, bottom)
            
            # Capture just one frame since we already have a face location
            ret, frame = self.cap.read()
            if ret:
                frames.append(frame)
        else:
            # Capture frames for analysis
            for i in range(num_frames):
                print(f"Capturing frame {i+1}/{num_frames}...")
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to capture frame")
                    continue
                frames.append(frame)
        
        if len(frames) == 0:
            print("No frames captured for texture analysis")
            # Restore original camera settings
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_resolution[1])
            return False, 0.0
        
        # Create diagnostic image
        diagnostic_img = frames[0].copy()
        
        # Save a copy of the high-res image
        cv2.imwrite("highres_texture_check.jpg", diagnostic_img)
        
        # Initialize texture metrics
        gradient_averages = []
        std_averages = []
        detail_energy_averages = []
        
        # Store ROIs for visualization
        roi_regions = []
        
        # Process each frame
        for frame_idx, frame in enumerate(frames):
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use existing face location or detect faces
            if face_rect is not None:
                faces = [face_rect]
            else:
                # Detect faces
                faces = self.face_detector(gray)
                
            if len(faces) == 0:
                print(f"No face detected in frame {frame_idx+1}")
                continue
            
            # Get largest face
            largest_face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Get facial landmarks
            shape = self.shape_predictor(gray, largest_face)
            landmarks = face_utils.shape_to_np(shape)
            
            # Define regions to analyze (forehead, left cheek, right cheek, chin)
            face_width = largest_face.width()
            face_height = largest_face.height()
            
            # Print face size for diagnostics
            print(f"Face size: {face_width}x{face_height} pixels")
            
            # Calculate forehead region (above eyes)
            left_eye_avg = np.mean(landmarks[36:42], axis=0).astype(int)
            right_eye_avg = np.mean(landmarks[42:48], axis=0).astype(int)
            eye_y = min(left_eye_avg[1], right_eye_avg[1])
            forehead_y = max(eye_y - int(face_height * 0.15), largest_face.top())
            forehead_h = eye_y - forehead_y
            forehead_roi = (
                largest_face.left() + int(face_width * 0.25), 
                forehead_y,
                int(face_width * 0.5), 
                forehead_h
            )
            
            # Get cheek regions
            nose_tip = landmarks[33]
            left_cheek_roi = (
                largest_face.left() + int(face_width * 0.1),
                nose_tip[1],
                int(face_width * 0.25),
                int(face_height * 0.2)
            )
            
            right_cheek_roi = (
                nose_tip[0] + int(face_width * 0.15),
                nose_tip[1],
                int(face_width * 0.25),
                int(face_height * 0.2)
            )
            
            # Get chin region
            mouth_bottom = landmarks[57]
            chin_roi = (
                largest_face.left() + int(face_width * 0.33),
                mouth_bottom[1],
                int(face_width * 0.33),
                int(face_height * 0.15)
            )
            
            # Store regions for the first frame for visualization
            if frame_idx == 0:
                roi_regions = [forehead_roi, left_cheek_roi, right_cheek_roi, chin_roi]
                
            # Analyze texture in each region
            regions = [
                ("Forehead", gray[forehead_roi[1]:forehead_roi[1]+forehead_roi[3], 
                              forehead_roi[0]:forehead_roi[0]+forehead_roi[2]]),
                ("Left Cheek", gray[left_cheek_roi[1]:left_cheek_roi[1]+left_cheek_roi[3], 
                                 left_cheek_roi[0]:left_cheek_roi[0]+left_cheek_roi[2]]),
                ("Right Cheek", gray[right_cheek_roi[1]:right_cheek_roi[1]+right_cheek_roi[3], 
                                  right_cheek_roi[0]:right_cheek_roi[0]+right_cheek_roi[2]]),
                ("Chin", gray[chin_roi[1]:chin_roi[1]+chin_roi[3], 
                           chin_roi[0]:chin_roi[0]+chin_roi[2]])
            ]
            
            # Process each facial region
            frame_gradients = []
            frame_stds = []
            frame_energies = []
            
            for region_name, region_img in regions:
                if region_img.size == 0:
                    print(f"Warning: {region_name} region is empty")
                    continue
                
                # Calculate gradient magnitude (Sobel)
                sobelx = cv2.Sobel(region_img, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(region_img, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                avg_gradient = np.mean(gradient_magnitude)
                
                # Calculate standard deviation (local contrast)
                std_dev = np.std(region_img.astype(float))
                
                # Calculate high-frequency energy using DCT
                dct = cv2.dct(region_img.astype(float))
                # Mask out low frequencies (keep top-right, bottom-left, bottom-right quadrants)
                mask = np.ones(dct.shape, dtype=float)
                mask[:dct.shape[0]//2, :dct.shape[1]//2] = 0  # Zero out top-left quadrant
                masked_dct = dct * mask
                # Energy ratio of high frequencies to total
                total_energy = np.sum(dct**2)
                high_freq_energy = np.sum(masked_dct**2)
                energy_ratio = high_freq_energy / total_energy if total_energy > 0 else 0
                
                # Store metrics
                frame_gradients.append(avg_gradient)
                frame_stds.append(std_dev)
                frame_energies.append(energy_ratio)
                
                if frame_idx == 0:  # Only print detailed region metrics for first frame
                    print(f"{region_name}: Grad={avg_gradient:.2f}, Std={std_dev:.2f}, Energy={energy_ratio:.2f}")
            
            # Average the metrics across regions for this frame
            if frame_gradients:
                gradient_averages.append(np.mean(frame_gradients))
                std_averages.append(np.mean(frame_stds))
                detail_energy_averages.append(np.mean(frame_energies))
        
        # Clear the stored face location after processing
        if hasattr(self, 'current_face_location'):
            self.current_face_location = None
            
        # Restore original camera settings
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_resolution[1])
        
        # Final results - average across frames
        if not gradient_averages:
            print("No valid texture measurements")
            return False, 0.0
        
        avg_gradient = np.mean(gradient_averages)
        avg_std = np.mean(std_averages)
        avg_energy = np.mean(detail_energy_averages)
        
        print("\nFinal texture metrics:")
        print(f"Average gradient: {avg_gradient:.2f}")
        print(f"Average std: {avg_std:.2f}")
        print(f"Average detail energy: {avg_energy:.4f}")
        
        # Visualize regions on the diagnostic image
        for i, roi in enumerate(roi_regions):
            color = (0, 255, 0)  # Green
            cv2.rectangle(diagnostic_img, 
                         (roi[0], roi[1]), 
                         (roi[0] + roi[2], roi[1] + roi[3]), 
                         color, 2)
            region_names = ["Forehead", "Left Cheek", "Right Cheek", "Chin"]
            cv2.putText(diagnostic_img, region_names[i], 
                       (roi[0], roi[1] - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Save the texture analysis visualization
        cv2.imwrite("texture_analysis.jpg", diagnostic_img)
        
        # Check if texture metrics fall within expected ranges for real skin
        gradient_check = gradient_min_threshold <= avg_gradient <= gradient_max_threshold
        std_check = std_min_threshold <= avg_std <= std_max_threshold
        energy_check = avg_energy >= detail_energy_threshold
        
        # Calculate confidence score (0-100%)
        confidence = 0.0
        
        # Gradient confidence
        if avg_gradient < gradient_min_threshold:
            gradient_conf = max(0, avg_gradient / gradient_min_threshold * 100)
        elif avg_gradient > gradient_max_threshold:
            gradient_conf = max(0, (100 - (avg_gradient - gradient_max_threshold) / gradient_max_threshold * 100))
        else:
            # Within range - calculate how centered it is in the acceptable range
            gradient_range = gradient_max_threshold - gradient_min_threshold
            optimal_gradient = (gradient_min_threshold + gradient_max_threshold) / 2
            distance_from_optimal = abs(avg_gradient - optimal_gradient)
            gradient_conf = 100 - (distance_from_optimal / (gradient_range / 2) * 20)  # Penalize up to 20%
        
        # Standard deviation confidence
        if avg_std < std_min_threshold:
            std_conf = max(0, avg_std / std_min_threshold * 100)
        elif avg_std > std_max_threshold:
            std_conf = max(0, (100 - (avg_std - std_max_threshold) / std_max_threshold * 100))
        else:
            # Within range - calculate how centered it is in the acceptable range
            std_range = std_max_threshold - std_min_threshold
            optimal_std = (std_min_threshold + std_max_threshold) / 2
            distance_from_optimal = abs(avg_std - optimal_std)
            std_conf = 100 - (distance_from_optimal / (std_range / 2) * 20)  # Penalize up to 20%
        
        # Energy confidence - linear scaling
        energy_conf = min(100, avg_energy / detail_energy_threshold * 100)
        
        # Weighted average of confidences
        confidence = (gradient_conf * 0.4 + std_conf * 0.4 + energy_conf * 0.2) / 100.0
        
        # Result with threshold checks
        is_real_texture = gradient_check and std_check and energy_check
        
        # Create a result visualization
        result_img = diagnostic_img.copy()
        
        # Add text with results
        text_lines = [
            f"Gradient: {avg_gradient:.2f} ({gradient_check})",
            f"Std Dev: {avg_std:.2f} ({std_check})",
            f"Energy: {avg_energy:.4f} ({energy_check})",
            f"Real Texture: {is_real_texture}",
            f"Confidence: {confidence:.2f}"
        ]
        
        # Draw a semi-transparent box for text background
        overlay = result_img.copy()
        x, y = 10, 30
        h_line = 25  # height per line
        cv2.rectangle(overlay, (5, 5), (380, y + h_line * len(text_lines) + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, result_img, 0.3, 0, result_img)
        
        # Add each line of text
        for i, line in enumerate(text_lines):
            y_pos = y + i * h_line
            text_color = (0, 255, 0) if "True" in line else (0, 255, 255) if "Confidence" in line else (0, 0, 255)
            cv2.putText(result_img, line, (10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # Save the result visualization
        cv2.imwrite("texture_analysis_result.jpg", result_img)
        
        print(f"Texture analysis result: {is_real_texture} (Confidence: {confidence:.2f})")
        return is_real_texture, confidence
    
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
        # Reset focus check at the start of each detection session
        self.focus_check_passed = False
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        # Use slightly higher resolution for better eye landmark detection
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.liveness_resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.liveness_resolution[1])
        # Disable autofocus to reduce processing overhead
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        
        if not cap.isOpened():
            print("Error: Could not open camera")
            return False, None
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
        print("Blink detection started. Please look at the camera and blink normally.")
        print("The program will continue running for the full time period to allow for testing.")
        
        # Variables for blink detection
        blink_counter = 0
        ear_thresh_counter = 0
        blink_detected = False
        start_time = time.time()
        
        # Keep the best face image for return
        best_face_image = None
        max_face_size = 0
        
        # Downsample factor for face detection (to speed up processing)
        downsample = 0.25
        
        # Frame counter for processing only every nth frame
        frame_count = 0
        
        # Store the last processed face location to use for intermediate frames
        last_face_location = None
        last_landmarks = None
        
        # For adaptive EAR threshold
        ear_values = []
        
        # Create tracker once at the beginning
        tracker = self.create_tracker()
        tracking_active = False
        
        # Set frame processing rate - process more frequently to catch blinks better
        self.process_nth_frame = 2 if tracker is not None else 1
        print(f"Processing every {self.process_nth_frame}th frame for detection")
        
        # Position smoothing to reduce jitter
        position_history = []
        position_history_max = 5  # Increased number of frames for smoother tracking
        
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
        
        # Increase timeout slightly to allow for adaptation
        actual_timeout = timeout + 0.5
        
        # Track FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        
        # Initialize ear variable outside the scope to prevent reference errors
        ear = 0.0
        
        while (time.time() - start_time) < actual_timeout:
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
                gray1 = cv2.cvtColor(previous_frame, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
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
                    # Super strict threshold - almost no movement allowed
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
            
            # Only process every nth frame to improve performance
            process_this_frame = (frame_count % self.process_nth_frame == 0)
            frame_count += 1
            
            # For blink detection, try to process every frame when possible
            process_blink_this_frame = True
            
            # Current face box (to be determined in this frame)
            current_face_box = None
            
            # Try to update tracker if active
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
                # Downscale for faster processing
                small_frame = cv2.resize(frame, (0, 0), fx=downsample, fy=downsample)
                
                # Detect face locations
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                if face_locations:
                    # Scale back up face locations
                    top, right, bottom, left = face_locations[0]
                    top = int(top / downsample)
                    right = int(right / downsample)
                    bottom = int(bottom / downsample)
                    left = int(left / downsample)
                    
                    current_face_box = (top, right, bottom, left)
                    
                    # Initialize tracker if available and not active
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
                            tracking_active = False
                    
                    # Get landmarks - only if we found a face
                    landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [face_locations[0]])
                    
                    if landmarks_list:
                        # Convert landmarks to full scale
                        scaled_landmarks = {}
                        for feature, points in landmarks_list[0].items():
                            scaled_landmarks[feature] = [(int(p[0] / downsample), int(p[1] / downsample)) for p in points]
                        
                        last_landmarks = scaled_landmarks
            
            # Apply position smoothing if we have a new face position
            if current_face_box is not None:
                # Add current position to history
                position_history.append(current_face_box)
                # Keep history at max length
                if len(position_history) > position_history_max:
                    position_history.pop(0)
                
                # Calculate smoothed position
                if len(position_history) > 1:
                    # Simple averaging for smoother transitions
                    smoothed_top = sum(pos[0] for pos in position_history) // len(position_history)
                    smoothed_right = sum(pos[1] for pos in position_history) // len(position_history)
                    smoothed_bottom = sum(pos[2] for pos in position_history) // len(position_history)
                    smoothed_left = sum(pos[3] for pos in position_history) // len(position_history)
                    
                    last_face_location = (smoothed_top, smoothed_right, smoothed_bottom, smoothed_left)
                else:
                    # If only one position, use it directly
                    last_face_location = current_face_box
            
            # Use the last detected face and landmarks for processing
            if last_face_location is not None:
                top, right, bottom, left = last_face_location
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Save the largest face image for return if we haven't already detected a blink
                if not blink_detected:
                    face_size = (right - left) * (bottom - top)
                    if face_size > max_face_size:
                        max_face_size = face_size
                        best_face_image = frame.copy()
                        
                        # Perform focus-based liveness check when we have a good face
                        # Only do this once per session to avoid slowing down the process
                        if not self.focus_check_passed and face_size > 10000:  # Large enough face
                            self.focus_check_passed = self.perform_focus_liveness_check(cap, current_face_box)
                            
                            # If focus check failed, reset blink counter as additional security
                            if not self.focus_check_passed:
                                blink_counter = 0
                                print("Focus check failed, resetting blink counter")
                
                # Process landmarks if available
                if last_landmarks is not None and process_blink_this_frame:
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
                        
                        # Add to history for adaptive thresholding
                        ear_values.append(ear)
                        
                        # Calculate adaptive threshold if we have enough data
                        # (only after the first second or so)
                        adaptive_threshold = self.EAR_THRESHOLD
                        if len(ear_values) > 10:
                            # Use the 70th percentile as our baseline open eye EAR
                            open_eye_ear = np.percentile(ear_values, 70)
                            # Set threshold to 75% of the open eye EAR (less sensitive)
                            adaptive_threshold = open_eye_ear * 0.75
                            # But don't go below the minimum threshold
                            adaptive_threshold = max(adaptive_threshold, self.EAR_THRESHOLD)
                        
                        # Draw eye contours
                        for eye in [left_eye, right_eye]:
                            eye_hull = cv2.convexHull(np.array(eye))
                            cv2.drawContours(display_frame, [eye_hull], -1, (0, 255, 0), 1)
                        
                        # Only process blinks if we're very still
                        process_blink = is_still_enough
                        
                        # Check for blink pattern - need stable EAR history
                        if len(ear_history) >= 3 and process_blink:
                            # Check if current EAR is below threshold - indicates closed eyes
                            if ear < adaptive_threshold:
                                ear_thresh_counter += 1
                                # Draw RED eye contours when eyes are detected as closed
                                for eye in [left_eye, right_eye]:
                                    eye_hull = cv2.convexHull(np.array(eye))
                                    cv2.drawContours(display_frame, [eye_hull], -1, (0, 0, 255), 2)
                            else:
                                # Eyes are open now, if they were closed before, check if it was a blink
                                if ear_thresh_counter >= self.EAR_CONSEC_FRAMES:
                                    # Check if the EAR change is natural (photos will have unnatural changes)
                                    if len(ear_history) >= 5:
                                        # Calculate the EAR slope - needs to change gradually for real blinks
                                        ear_diff = abs(ear_history[-1] - ear_history[-3])
                                        ear_diff2 = abs(ear_history[-3] - ear_history[-5])
                                        
                                        # More strict natural blink detection
                                        natural_blink_pattern = (
                                            # Check for proper sequence: open -> closed -> open
                                            ear_history[-5] > ear_history[-3] and 
                                            ear_history[-3] < ear_history[-1] and
                                            # Check reasonable differences
                                            ear_diff > 0.01 and ear_diff2 > 0.005 and
                                            # Real blinks have a larger change
                                            abs(max(ear_history[-5:]) - min(ear_history[-5:])) > 0.03
                                        )
                                        
                                        if natural_blink_pattern:
                                            # Simpler blink detection - just check if we were below threshold
                                            blink_counter += 1
                                            print(f"Blink detected! EAR: {ear:.3f}, Threshold: {adaptive_threshold:.3f}")
                                
                                # Reset counter after eyes reopen
                                ear_thresh_counter = 0

                # Display EAR value and blink count - only if face was detected
                if last_face_location is not None and last_landmarks is not None:
                    # Format EAR value for display
                    ear_text = f"EAR: {ear:.2f}"
                    cv2.putText(display_frame, ear_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Display blink counter
                    cv2.putText(display_frame, f"Blinks: {blink_counter}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check for multiple blinks (2+) to validate liveness
                    if blink_counter >= 2:
                        #LOGGER.info("Multiple blinks detected - liveness check passed")
                        cv2.putText(display_frame, f"LIVENESS PASSED - {blink_counter} BLINKS", 
                                    (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        liveness_passed = True
                        
                        # Only log this once per session to avoid log spam
                        if not self.has_logged_liveness_passed and liveness_passed:
                            LOGGER.info("Liveness check passed - Multiple blinks detected")
                            self.has_logged_liveness_passed = True
                    
                    # Show high motion warning if detected
                    if high_motion_detected:
                        cv2.putText(display_frame, "HIGH MOTION DETECTED", 
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

                # Display focus check status
                focus_status = "Focus Check: PASSED" if self.focus_check_passed else "Focus Check: PENDING"
                focus_color = (0, 255, 0) if self.focus_check_passed else (0, 165, 255)
                cv2.putText(display_frame, focus_status, (10, display_frame.shape[0] - 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, focus_color, 1)

                # Display time remaining
                time_left = int(actual_timeout - (time.time() - start_time))
                cv2.putText(display_frame, f"Time: {time_left}s", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display motion status
                motion_status = "High Motion" if high_motion_detected else "Stable"
                motion_color = (0, 0, 255) if high_motion_detected else (0, 255, 0)
                cv2.putText(display_frame, motion_status, (display_frame.shape[1] - 120, 90),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
                
                # Display frame rate
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Display tracking status
                tracking_status = "Tracking" if (tracker is not None and tracking_active) else "Detecting"
                cv2.putText(display_frame, tracking_status, (display_frame.shape[1] - 120, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Display instructions
                if blink_counter > 0:
                    blink_detected = True
                    status = f"Blink detected! ({blink_counter} blinks) - Continuing for testing"
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
                    # Store result before exiting
                    liveness_result = False
                    best_face_img = None if best_face_image is None else best_face_image.copy()
                    
                    # Clean up main camera
                    cap.release()
                    cv2.destroyWindow("Liveness Detection")
                    
                    # Create result message for canceled test
                    result_frame = np.zeros((480, 640, 3), dtype=np.uint8) if best_face_img is None else best_face_img
                    cv2.putText(result_frame, "LIVENESS CHECK CANCELED", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.putText(result_frame, "Press any key to exit...", 
                                (20, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.imshow("Liveness Result", result_frame)
                    cv2.waitKey(0)  # Wait for key press
                    cv2.destroyAllWindows()
                    
                    # No need to clean up diagnostic images in ESC case
                    # as they may contain useful debug information
                    
                    return False, None
                
                # Continue running even after blink is detected (removed early exit)
                # Save the face image when a blink is first detected
                if blink_detected and best_face_image is None and max_face_size > 0:
                    best_face_image = frame.copy()
                
                # Check for multiple blinks (2+) to validate liveness
                if blink_counter >= 2:
                    #LOGGER.info("Multiple blinks detected - liveness check passed")
                    cv2.putText(display_frame, f"LIVENESS PASSED - {blink_counter} BLINKS", 
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    liveness_passed = True
                    
                    # Only log this once per session to avoid log spam
                    if not self.has_logged_liveness_passed and liveness_passed:
                        LOGGER.info("Liveness check passed - Multiple blinks detected")
                        self.has_logged_liveness_passed = True
                        
                    # Show high motion warning if detected
                    if high_motion_detected:
                        cv2.putText(display_frame, "HIGH MOTION DETECTED", 
                                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # If we've detected a blink but don't have a good face image, capture one now
        if blink_detected and best_face_image is None:
            best_face_image = self.capture_face()
        
        # Show final result and wait for user to acknowledge
        if blink_counter > 0:
            result_message = "PASSED" if blink_counter >= 2 else "INCOMPLETE"
            result_color = (0, 255, 0) if blink_counter >= 2 else (0, 165, 255)
        else:
            result_message = "FAILED"
            result_color = (0, 0, 255)
            
        result_frame = best_face_image.copy() if best_face_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(result_frame, f"LIVENESS CHECK {result_message}", 
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
        cv2.putText(result_frame, f"Detected {blink_counter} blinks", 
                    (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add focus test result if it was performed
        if hasattr(self, 'focus_check_passed'):
            focus_result = "PASSED" if self.focus_check_passed else "FAILED"
            focus_color = (0, 255, 0) if self.focus_check_passed else (0, 0, 255)
            cv2.putText(result_frame, f"Focus Check: {focus_result}", 
                        (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, focus_color, 2)
            
        cv2.putText(result_frame, "Press any key to continue...", 
                    (20, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.imshow("Liveness Result", result_frame)
        cv2.waitKey(0)  # Wait for any key press
        cv2.destroyAllWindows()
        
        # Clean up any diagnostic images generated during focus testing
        self.cleanup_diagnostic_images()
        
        return blink_detected, best_face_image
    
    def cleanup_diagnostic_images(self):
        """Clean up any diagnostic images from previous tests"""
        try:
            # Define files to clean
            files_to_clean = [
                "clarity_test.jpg",
                "focus_test_summary.jpg", 
                "focus_area_eye.jpg", 
                "focus_area_nose.jpg", 
                "focus_area_bg.jpg",
                "highres_texture_check.jpg",
                "texture_analysis.jpg",
                "texture_analysis_result.jpg"
            ]
            
            # Add focus test images for different levels
            for level in [0, 100, 250]:
                files_to_clean.append(f"focus_test_{level}.jpg")
            
            # Try to remove each file
            for file in files_to_clean:
                if os.path.exists(file):
                    os.remove(file)
                    print(f"Removed: {file}")
                    
        except Exception as e:
            print(f"Warning: Couldn't clean up some files: {str(e)}")
    
    # Replace the simplified placeholder with our real implementation
    def capture_face_with_liveness(self):
        """
        Capture a face with liveness detection (blink detection)
        
        Returns:
            image or None: Face image if liveness check passed, None otherwise
        """
        print("Starting liveness detection (blink check)...")
        success, face_image = self.detect_blink()
        
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

    def checkFaceIsReal(self):
        """Perform a sequence of tests to determine if a face is real or a photo"""
        self.cleanup_diagnostic_images()  # Clean up any previous test files

        print("Starting facial reality check...")
        
        # Ensure we have a camera object
        if not hasattr(self, 'cap') or self.cap is None:
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Capture a face to work with
        frame = self.capture_face()
        if frame is None:
            print("No face captured, aborting test")
            return False, 0.0
            
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize face detector if not already done
        if not hasattr(self, 'face_detector'):
            self.face_detector = dlib.get_frontal_face_detector()
            
        # Detect face
        faces = self.face_detector(gray)
        if not faces:
            print("No face detected in captured image")
            return False, 0.0
            
        # Get the largest face
        face = max(faces, key=lambda rect: rect.width() * rect.height())
        
        # Convert dlib rectangle to face_location tuple (top, right, bottom, left)
        face_location = (face.top(), face.right(), face.bottom(), face.left())
        
        # Save face location for focus and texture tests
        self.current_face_location = face_location

        # Step 1: Check focus/depth test - passing the face location
        focus_result, focus_score = self.check_focus_depth(self.cap, face_location)
        
        # Step 2: Check texture detail - this will use the saved face location
        texture_result, texture_score = self.check_facial_texture()
        
        # Step 3: Calculate overall result and confidence
        is_real = focus_result and texture_result
        
        # Calculate confidence as average of normalized scores
        confidence = (focus_score + texture_score) / 2.0
        
        result_str = "REAL FACE" if is_real else "FAKE (PHOTO)"
        conf_pct = int(confidence * 100)
        
        print(f"\nFINAL RESULT: {result_str} (Confidence: {conf_pct}%)")
        print(f"  - Focus/Depth Test: {'PASSED' if focus_result else 'FAILED'} ({int(focus_score*100)}%)")
        print(f"  - Texture Test: {'PASSED' if texture_result else 'FAILED'} ({int(texture_score*100)}%)")
        
        # Clean up
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
            self.cap = None
        
        return is_real, confidence
        
    def check_facial_texture(self):
        """Check facial texture for signs of a real face vs a photo"""
        print("\nPerforming facial texture analysis...")
        
        # Save current camera settings to restore later
        original_resolution = (self.cap.get(cv2.CAP_PROP_FRAME_WIDTH), self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Original camera resolution: {original_resolution}")
        
        # Set to 1080p resolution for texture analysis
        target_resolution = (1920, 1080)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_resolution[1])
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        print(f"Set camera to {actual_width}x{actual_height} resolution for texture analysis")
        
        # Capture frames for analysis
        frames = []
        face_rects = []
        
        # Capture multiple frames for stability
        for i in range(5):
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect face
            faces = self.face_detector(gray)
            if len(faces) == 0:
                continue
                
            # Get the largest face
            face = max(faces, key=lambda rect: rect.area())
            face_rects.append(face)
            frames.append((frame, gray))
        
        if not frames:
            print("Could not capture any frames with faces for texture analysis")
            return False, 0.0
            
        # Use the most recent stable frame
        frame, gray = frames[-1]
        face = face_rects[-1]
        
        # Extract facial regions (forehead, cheek, chin)
        landmarks = self.shape_predictor(gray, face)
        
        # Calculate face size in pixels
        face_width = face.width()
        face_height = face.height()
        print(f"Face size: {face_width}x{face_height} pixels")
        
        # Size of the region to analyze - adjust based on face size
        region_size = min(50, int(min(face_width, face_height) / 10))
        
        # Get regions of interest
        regions = []
        region_names = []

        # Forehead
        forehead_x = (landmarks.part(27).x + landmarks.part(21).x + landmarks.part(22).x) // 3
        forehead_y = landmarks.part(21).y - region_size
        forehead_y = max(forehead_y, 0)  # Ensure within image bounds
        regions.append((forehead_x, forehead_y, region_size))
        region_names.append("Forehead")
        
        # Left cheek
        left_cheek_x = landmarks.part(31).x - region_size
        left_cheek_y = landmarks.part(31).y 
        regions.append((left_cheek_x, left_cheek_y, region_size))
        region_names.append("Left Cheek")
        
        # Right cheek
        right_cheek_x = landmarks.part(35).x
        right_cheek_y = landmarks.part(35).y
        regions.append((right_cheek_x, right_cheek_y, region_size))
        region_names.append("Right Cheek")
        
        # Chin
        chin_x = landmarks.part(8).x
        chin_y = landmarks.part(8).y - region_size
        regions.append((chin_x, chin_y, region_size))
        region_names.append("Chin")
        
        # Create copy of frame for visualization
        vis_frame = frame.copy()
        
        # Calculate texture statistics for each region
        avg_gradient = 0
        avg_std = 0
        avg_detail_energy = 0
        valid_regions = 0
        
        texture_results = []
        
        # Set fixed thresholds based on typical real skin values at 1080p
        # These are empirical values that work well at 1080p resolution
        gradient_min = 10.0
        gradient_max = 180.0
        std_min = 6.0
        std_max = 100.0
        energy_threshold = 0.5
        
        for i, (x, y, size) in enumerate(regions):
            # Make sure region is within image bounds
            if x < 0 or y < 0 or x + size >= gray.shape[1] or y + size >= gray.shape[0]:
                continue
                
            # Extract region
            region = gray[y:y+size, x:x+size]
            
            # Calculate gradient magnitude
            sobel_x = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
            avg_region_gradient = np.mean(gradient_mag)
            
            # Calculate standard deviation of intensity (local contrast)
            std_dev = np.std(region.astype(np.float64))
            
            # Calculate high-frequency energy (detail)
            # Use Laplacian for edge detection
            laplacian = cv2.Laplacian(region, cv2.CV_64F)
            energy = np.mean(np.abs(laplacian)) / 255.0  # Normalize
            
            # Add to averages
            avg_gradient += avg_region_gradient
            avg_std += std_dev
            avg_detail_energy += energy
            valid_regions += 1
            
            # Draw region on visualization
            cv2.rectangle(vis_frame, (x, y), (x+size, y+size), (0, 255, 0), 2)
            cv2.putText(vis_frame, region_names[i], (x, y-5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save results for this region
            texture_results.append({
                "region": region_names[i],
                "gradient": avg_region_gradient,
                "std_dev": std_dev,
                "energy": energy
            })
            
        # Save high-res frame with regions marked
        cv2.imwrite("highres_texture_check.jpg", vis_frame)
        
        if valid_regions == 0:
            print("No valid regions found for texture analysis")
            
            # Restore original resolution
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_resolution[1])
            return False, 0.0
            
        # Calculate averages
        avg_gradient /= valid_regions
        avg_std /= valid_regions
        avg_detail_energy /= valid_regions
        
        # Create visualization for texture analysis
        texture_vis = np.zeros((400, 800, 3), dtype=np.uint8)
        
        # Display the results
        y_offset = 30
        cv2.putText(texture_vis, "Facial Texture Analysis", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        y_offset += 40
        
        for result in texture_results:
            text = f"{result['region']}: Gradient={result['gradient']:.2f}, StdDev={result['std_dev']:.2f}, Energy={result['energy']:.2f}"
            cv2.putText(texture_vis, text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            y_offset += 30
            
        y_offset += 10
        cv2.putText(texture_vis, f"Average Gradient: {avg_gradient:.2f} (Threshold: {gradient_min:.2f}-{gradient_max:.2f})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        y_offset += 30
        cv2.putText(texture_vis, f"Average StdDev: {avg_std:.2f} (Threshold: {std_min:.2f}-{std_max:.2f})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
        y_offset += 30
        cv2.putText(texture_vis, f"Average Detail Energy: {avg_detail_energy:.2f} (Threshold: {energy_threshold:.2f})", 
                   (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2)
                   
        # Determine if texture appears to be real skin
        # Real skin typically has moderate gradients and high-frequency details
        # Photos may be very smooth (low gradient) or have sharp edges (high gradient)
        gradient_in_range = gradient_min <= avg_gradient <= gradient_max
        std_in_range = std_min <= avg_std <= std_max
        sufficient_detail = avg_detail_energy >= energy_threshold
        
        # Decision
        is_real_texture = gradient_in_range and std_in_range and sufficient_detail
        texture_score = 0.0
        
        # Calculate confidence score (0.0 to 1.0)
        if is_real_texture:
            # If all checks pass, calculate how well within the ranges
            gradient_score = 1.0 - min(abs(avg_gradient - (gradient_min + gradient_max)/2), (gradient_max - gradient_min)/2) / ((gradient_max - gradient_min)/2)
            std_score = 1.0 - min(abs(avg_std - (std_min + std_max)/2), (std_max - std_min)/2) / ((std_max - std_min)/2)
            energy_score = min(avg_detail_energy / (energy_threshold * 2), 1.0)
            texture_score = (gradient_score + std_score + energy_score) / 3.0
        else:
            # If any check fails, calculate partial score
            checks_passed = 0
            if gradient_in_range:
                checks_passed += 1
            if std_in_range:
                checks_passed += 1
            if sufficient_detail:
                checks_passed += 1
            texture_score = checks_passed / 3.0 * 0.6  # Max 60% confidence if not all checks pass
        
        # Add result to visualization
        y_offset += 40
        result_text = "REAL TEXTURE" if is_real_texture else "FAKE TEXTURE (PHOTO)"
        color = (0, 255, 0) if is_real_texture else (0, 0, 255)
        cv2.putText(texture_vis, f"RESULT: {result_text}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        y_offset += 30
        cv2.putText(texture_vis, f"Confidence: {int(texture_score*100)}%", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Save visualization
        cv2.imwrite("texture_analysis.jpg", texture_vis)
        
        # Create final result image
        result_img = np.zeros((600, 1000, 3), dtype=np.uint8)
        
        # Add captured image with detected regions
        captured_h, captured_w = vis_frame.shape[:2]
        scale_factor = min(500 / captured_h, 500 / captured_w)
        resized_capture = cv2.resize(vis_frame, (int(captured_w * scale_factor), int(captured_h * scale_factor)))
        
        h, w = resized_capture.shape[:2]
        result_img[50:50+h, 50:50+w] = resized_capture
        
        # Add result text
        y_offset = 50 + h + 40
        result_text = "REAL TEXTURE" if is_real_texture else "FAKE TEXTURE (PHOTO)"
        color = (0, 255, 0) if is_real_texture else (0, 0, 255)
        cv2.putText(result_img, f"Texture Analysis Result: {result_text}", 
                   (50, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                   
        # Save the result image
        cv2.imwrite("texture_analysis_result.jpg", result_img)
        
        # Print results
        print(f"Texture Analysis Results:")
        print(f"  Average Gradient: {avg_gradient:.2f} (Threshold: {gradient_min:.2f}-{gradient_max:.2f})")
        print(f"  Average StdDev: {avg_std:.2f} (Threshold: {std_min:.2f}-{std_max:.2f})")
        print(f"  Average Detail Energy: {avg_detail_energy:.2f} (Threshold: {energy_threshold:.2f})")
        print(f"  Texture appears to be {'real skin' if is_real_texture else 'a photo'}")
        print(f"  Confidence: {int(texture_score*100)}%")
        
        # Restore original resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_resolution[1])
        print(f"Restored camera to original resolution: {original_resolution}")
        
        return is_real_texture, texture_score

# For demonstration
if __name__ == "__main__":
    camera = CameraSystem()
    
    # Empty database to start
    known_encodings = []
    known_names = []
    
    # Run facial recognition
    known_encodings, known_names = camera.run_facial_recognition(known_encodings, known_names) 