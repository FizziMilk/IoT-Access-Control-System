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
            
            # Run enhanced liveness checks
            focus_result = self.check_focus_depth(cap, face_location)
            texture_result = self.check_facial_texture(cap, face_location)
            
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
                cv2.putText(summary_frame, f"Texture Test: {texture_status}", (20, 180),
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
        """
        # Get our original focus test logic but in a separate function
        try:
            # Extract face region for analysis
            top, right, bottom, left = face_location
            face_height = bottom - top
            
            # Define focus test points - try different regions
            # Use eye region vs nose region (should have different depths)
            eye_y = top + int(face_height * 0.3)
            nose_y = top + int(face_height * 0.5)
            center_x = left + int((right - left) / 2)
            
            # Points to analyze - ensure all coordinates are integers
            eye_point = (int(center_x), int(eye_y))
            nose_point = (int(center_x), int(nose_y))
            
            # Add background point for comparison - ensure all values are integers
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            bg_x = max(0, left - int((right - left) * 0.5))  # Point to the left of face
            if bg_x < 20:  # If face too close to left edge, use right side
                bg_x = min(frame_width - 20, right + int((right - left) * 0.5))
            bg_point = (int(bg_x), int(eye_y))
            
            # Test more extreme focus levels with longer adjustment time
            focus_levels = [0, 125, 250]  # Min, mid, max focus values
            clarity_values = []
            
            # Disable autofocus for manual control - be more aggressive
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            time.sleep(0.5)  # Give time for autofocus to turn off
            
            # Store diagnostic images
            diagnostic_images = []
            
            for focus in focus_levels:
                # Set focus
                print(f"Setting focus to {focus}")
                cap.set(cv2.CAP_PROP_FOCUS, focus)
                
                # Give more time for focus to adjust
                time.sleep(1.0)
                
                # Verify focus actually changed
                current_focus = cap.get(cv2.CAP_PROP_FOCUS)
                
                # Capture multiple frames and use the last one
                for _ in range(5):
                    ret, frame = cap.read()
                    time.sleep(0.1)
                    if not ret:
                        break
                
                if not ret:
                    print("Failed to capture frame")
                    continue
                
                # Convert to grayscale for clarity analysis
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculate local clarity at test points (using Laplacian variance)
                eye_clarity = self.calculate_clarity(gray, eye_point, region_size=30)
                nose_clarity = self.calculate_clarity(gray, nose_point, region_size=30)
                bg_clarity = self.calculate_clarity(gray, bg_point, region_size=30)
                
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
                return False
                
            eye_nose_correlation = np.corrcoef(
                [c[1] for c in clarity_values], 
                [c[2] for c in clarity_values]
            )[0, 1]
            
            eye_bg_correlation = np.corrcoef(
                [c[1] for c in clarity_values], 
                [c[3] for c in clarity_values]
            )[0, 1]
            
            print(f"\nCorrelation - Eye-Nose: {eye_nose_correlation:.3f}, Eye-BG: {eye_bg_correlation:.3f}")
            print(f"Difference: {abs(eye_nose_correlation - eye_bg_correlation):.3f}")
            
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
            
            # Stricter criteria:
            # 1. The difference between eye-nose and eye-bg correlations must be significant (> 0.3)
            # 2. AND the correlations should NOT both be highly positive (indicates photo)
            correlation_diff = abs(eye_nose_correlation - eye_bg_correlation)
            both_positive = eye_nose_correlation > 0.8 and eye_bg_correlation > 0.8
            
            # Is this a real face based on focus test?
            return correlation_diff > 0.3 and not both_positive
            
        except Exception as e:
            print(f"Error in focus depth check: {e}")
            return False
    
    def check_facial_texture(self, cap, face_location):
        """
        Check for natural skin textures that are difficult to reproduce in photos.
        Uses highest possible resolution for maximum detail.
        """
        try:
            # Save current camera settings
            original_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            original_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            original_autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            
            # Try maximum resolutions in descending order - added even higher resolutions
            resolutions = [(3840, 2160), (2560, 1440), (1920, 1080), (1280, 720), (800, 600)]
            max_resolution_set = False
            
            for width, height in resolutions:
                try:
                    print(f"Attempting to set camera resolution to {width}x{height}")
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    
                    # Verify the resolution was actually set
                    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    
                    if actual_width > original_width or actual_height > original_height:
                        print(f"Successfully set resolution to {actual_width}x{actual_height}")
                        max_resolution_set = True
                        break
                    else:
                        print(f"Failed to increase resolution, got {actual_width}x{actual_height}")
                except Exception as e:
                    print(f"Error setting resolution to {width}x{height}: {e}")
            
            if not max_resolution_set:
                print("Using original resolution for texture analysis")
                
            # Enable autofocus and wait for camera to adjust
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            
            # Wait longer for camera to adjust to new resolution and focus
            time.sleep(2.5)  # Increased wait time for camera adjustment
                
            # Capture multiple frames to let camera adjust and stabilize
            for i in range(10):  # Increased from 8 to 10 frames
                ret, frame = cap.read()
                time.sleep(0.3)  # Maintain wait time between frames
                if not ret:
                    break
                    
            # Capture final high-res frame for analysis
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame for texture analysis")
                return True  # Default to pass if we can't capture a frame
            
            # Save the high-resolution image for verification
            cv2.imwrite("highres_texture_check.jpg", frame)
            
            # Print actual resolution used for analysis
            print(f"Final image resolution for texture analysis: {frame.shape[1]}x{frame.shape[0]}")
                
            # Extract face details
            top, right, bottom, left = face_location
            
            # Scale face location to match higher resolution if needed
            scale_x = frame.shape[1] / original_width
            scale_y = frame.shape[0] / original_height
            
            if scale_x != 1.0 or scale_y != 1.0:
                print(f"Scaling face location by factors: x={scale_x:.2f}, y={scale_y:.2f}")
                top = int(top * scale_y)
                right = int(right * scale_x)
                bottom = int(bottom * scale_y)
                left = int(left * scale_x)
            
            face_width = right - left
            face_height = bottom - top
            
            print(f"Face dimensions: {face_width}x{face_height} pixels")
            
            # Only analyze faces that are large enough for meaningful texture analysis
            min_face_size = 100  # Lowered from 120 to allow analysis on smaller faces
            if face_width < min_face_size or face_height < min_face_size:
                print(f"Face too small for reliable texture analysis: {face_width}x{face_height}")
                return True  # Default to pass for small faces
            
            # Convert to grayscale for texture analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Extract multiple regions for more reliable analysis
            # 1. Forehead region
            forehead_x = left + int(face_width * 0.5)  # Center
            forehead_y = top + int(face_height * 0.2)  # Upper part
            
            # 2. Cheek region
            cheek_x = left + int(face_width * 0.7)  # Right cheek
            cheek_y = top + int(face_height * 0.5)  # Middle of face
            
            # 3. Chin region
            chin_x = left + int(face_width * 0.5)  # Center
            chin_y = top + int(face_height * 0.8)  # Lower part
            
            # Region size proportional to face size - slightly increased size for more detail
            region_size = int(min(face_width, face_height) * 0.2)  # Increased from 0.18 to 0.2
            
            # Define regions to analyze
            regions = [
                ("Forehead", forehead_x, forehead_y),
                ("Cheek", cheek_x, cheek_y),
                ("Chin", chin_x, chin_y)
            ]
            
            # Add regions for the left cheek and nose for more comprehensive analysis
            regions.extend([
                ("L-Cheek", left + int(face_width * 0.3), top + int(face_height * 0.5)),  # Left cheek
                ("Nose", left + int(face_width * 0.5), top + int(face_height * 0.55))     # Nose area
            ])
            
            # Create debug visualization
            debug_texture = frame.copy()
            
            # Process each region
            region_results = []
            for region_name, center_x, center_y in regions:
                # Ensure region is within image bounds
                x1 = max(0, center_x - region_size // 2)
                y1 = max(0, center_y - region_size // 2)
                x2 = min(frame.shape[1] - 1, center_x + region_size // 2)
                y2 = min(frame.shape[0] - 1, center_y + region_size // 2)
                
                # Extract region
                region = gray[y1:y2, x1:x2]
                
                if region.size == 0:
                    print(f"Warning: {region_name} region has zero size")
                    continue
                
                # Calculate gradient magnitudes using Sobel operator with larger kernel for higher res
                sobelx = cv2.Sobel(region, cv2.CV_64F, 1, 0, ksize=5)
                sobely = cv2.Sobel(region, cv2.CV_64F, 0, 1, ksize=5)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                
                # Calculate texture statistics
                mean_gradient = np.mean(gradient_magnitude)
                std_gradient = np.std(gradient_magnitude)
                
                # Apply multi-scale analysis for high-resolution images
                # First-level high-pass filter (fine details)
                blurred1 = cv2.GaussianBlur(region, (5, 5), 0)
                high_freq1 = region.astype(np.float32) - blurred1.astype(np.float32)
                high_freq_energy1 = np.sum(high_freq1**2) / region.size
                
                # Second-level high-pass filter (medium details)
                blurred2 = cv2.GaussianBlur(region, (9, 9), 0)
                high_freq2 = blurred1.astype(np.float32) - blurred2.astype(np.float32)
                high_freq_energy2 = np.sum(high_freq2**2) / region.size
                
                # Combine energies with more weight on fine details
                high_freq_energy = high_freq_energy1 * 0.7 + high_freq_energy2 * 0.3
                
                # Store results
                region_results.append({
                    "name": region_name,
                    "gradient": mean_gradient,
                    "std": std_gradient,
                    "energy": high_freq_energy,
                    "rect": (x1, y1, x2, y2)
                })
                
                # Visualize region on debug image
                cv2.rectangle(debug_texture, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(debug_texture, region_name, (x1, y1 - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Display overall statistics
            avg_gradient = np.mean([r["gradient"] for r in region_results])
            avg_std = np.mean([r["std"] for r in region_results])
            avg_energy = np.mean([r["energy"] for r in region_results])
            
            y_offset = 30
            cv2.putText(debug_texture, f"Avg Gradient: {avg_gradient:.2f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(debug_texture, f"Avg Std: {avg_std:.2f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(debug_texture, f"Avg Detail: {avg_energy:.2f}", (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
            cv2.putText(debug_texture, f"Resolution: {frame.shape[1]}x{frame.shape[0]}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Show texture analysis image
            cv2.imshow("Texture Analysis", debug_texture)
            cv2.imwrite("texture_analysis.jpg", debug_texture)
            
            # Print detailed results
            print("\nSkin Texture Analysis Results:")
            print("-" * 60)
            print(f"{'Region':<10} {'Gradient':<10} {'Std Dev':<10} {'Detail Energy':<15}")
            print("-" * 60)
            for result in region_results:
                print(f"{result['name']:<10} {result['gradient']:<10.2f} {result['std']:<10.2f} {result['energy']:<15.2f}")
            print("-" * 60)
            print(f"{'AVERAGE':<10} {avg_gradient:<10.2f} {avg_std:<10.2f} {avg_energy:<15.2f}")
            print("-" * 60)
            
            # Adaptive thresholds based on resolution
            base_gradient_min = 5.0
            base_gradient_max = 60.0
            base_std_min = 3.0
            base_std_max = 40.0
            base_energy_min = 15.0
            
            # Scale thresholds based on actual resolution compared to 720p baseline
            resolution_scale = np.sqrt((frame.shape[0] * frame.shape[1]) / (720 * 1280))
            gradient_min = base_gradient_min * resolution_scale
            gradient_max = base_gradient_max * resolution_scale
            std_min = base_std_min * resolution_scale
            std_max = base_std_max * resolution_scale
            energy_min = base_energy_min * resolution_scale
            
            print(f"Resolution scale factor: {resolution_scale:.2f}")
            print(f"Adjusted thresholds - Gradient: {gradient_min:.2f}-{gradient_max:.2f}, Std: {std_min:.2f}-{std_max:.2f}, Energy: {energy_min:.2f}")
            
            # Decision logic for texture analysis with adaptive thresholds
            is_real_texture = (
                avg_gradient > gradient_min and avg_gradient < gradient_max and
                avg_std > std_min and avg_std < std_max and
                avg_energy > energy_min
            )
            
            # Decision text
            decision = "REAL SKIN" if is_real_texture else "PHOTO DETECTED"
            color = (0, 255, 0) if is_real_texture else (0, 0, 255)
            cv2.putText(debug_texture, decision, (10, debug_texture.shape[0] - 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            
            # Display result for user feedback
            cv2.imshow("Texture Analysis", debug_texture)
            
            print(f"\nTexture Analysis Decision: {decision}")
            print(f"Texture test passed: {is_real_texture}")
            
            # Restore original camera settings
            try:
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_height)
                cap.set(cv2.CAP_PROP_AUTOFOCUS, original_autofocus)
                time.sleep(0.5)  # Allow camera to adjust back
            except:
                print("Warning: Could not restore original camera settings")
                
            return is_real_texture
            
        except Exception as e:
            print(f"Texture analysis error: {str(e)}")
            traceback.print_exc()
            return True  # Fail safe - if texture analysis fails, allow the user to proceed
    
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
        """Clean up diagnostic images created during focus testing"""
        try:
            # Focus levels used in the test
            focus_levels = [0, 125, 250]
            
            # Remove all focus test images
            files_to_clean = [f"focus_test_{level}.jpg" for level in focus_levels]
            files_to_clean.extend([
                "focus_test_collage.jpg", 
                "focus_test_result.jpg",
                "highres_texture_check.jpg",
                "texture_analysis.jpg",
                "texture_analysis_result.jpg"
            ])
            
            for filename in files_to_clean:
                if os.path.exists(filename):
                    os.remove(filename)
                    print(f"Removed {filename}")
                else:
                    print(f"File not found: {filename}")
        except Exception as e:
            print(f"Error cleaning up diagnostic images: {e}")
            # Continue even if cleanup fails
        
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

# For demonstration
if __name__ == "__main__":
    camera = CameraSystem()
    
    # Empty database to start
    known_encodings = []
    known_names = []
    
    # Run facial recognition
    known_encodings, known_names = camera.run_facial_recognition(known_encodings, known_names) 