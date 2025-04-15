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
from sklearn.metrics.pairwise import euclidean_distances
from skimage import feature as skimage_feature

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
            
            # Capture frame for texture analysis
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to capture frame for texture analysis")
                return True
            
            # Run enhanced liveness checks
            focus_result = self.check_focus_depth(cap, face_location)
            texture_result = self.analyze_facial_texture(frame)
            
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
        Enhanced test for liveness by analyzing focus patterns across multiple depth settings.
        Uses gradient analysis and focus consistency to detect flat photos vs 3D faces.
        """
        try:
            print("Starting enhanced focus depth test...")
            # Completely different approach: always use center of frame with large region
            
            # Capture a full frame first to get dimensions
            ret, full_frame = cap.read()
            if not ret:
                raise Exception("Failed to get frame for region selection")
                
            frame_height, frame_width = full_frame.shape[:2]
            
            # ALWAYS use center of frame with a large region (60% of frame height)
            center_x = frame_width // 2
            center_y = frame_height // 2
            roi_size = int(frame_height * 0.6)  # Much larger region
            
            # Calculate the region boundaries
            half_size = roi_size // 2
            left = center_x - half_size
            top = center_y - half_size
            right = center_x + half_size
            bottom = center_y + half_size
            
            # Ensure coordinates are valid
            left = max(0, left)
            top = max(0, top)
            right = min(frame_width, right)
            bottom = min(frame_height, bottom)
            
            print(f"FOCUS TEST REGION: center=({center_x}, {center_y}), size={roi_size}")
            print(f"Coordinates: left={left}, top={top}, right={right}, bottom={bottom}")
            
            # Show the selected region on the full frame
            debug_frame = full_frame.copy()
            cv2.rectangle(debug_frame, (left, top), (right, bottom), (0, 255, 0), 3)
            cv2.putText(debug_frame, "FOCUS TEST REGION", (left, top - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Focus Test Region", debug_frame)
            cv2.waitKey(1000)  # Show for a full second
            
            # Save original camera settings
            original_resolution = (cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            original_autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            original_focus = cap.get(cv2.CAP_PROP_FOCUS)
            
            # Temporarily increase resolution for better focus analysis
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Recalculate region for higher resolution frame
            scale_x = 1920 / original_resolution[0]
            scale_y = 1080 / original_resolution[1]
            
            # Scale the coordinates to the new resolution
            scaled_left = int(left * scale_x)
            scaled_top = int(top * scale_y)
            scaled_right = int(right * scale_x)
            scaled_bottom = int(bottom * scale_y)
            
            print(f"Scaled coordinates for higher resolution: left={scaled_left}, top={scaled_top}, right={scaled_right}, bottom={scaled_bottom}")
            
            # Disable autofocus
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            
            # Use more focus points for better depth analysis
            focus_measurements = []
            focus_values = [0, 50, 100, 150, 200, 250]  # More test points
            focus_images = []  # Store images for analysis and debugging
            gradient_measures = []  # Store gradient measures
            
            # Set baseline measurements including gradient analysis
            focus_gradients = []
            
            for focus_val in focus_values:
                # Set focus value (0-255 range for most webcams)
                cap.set(cv2.CAP_PROP_FOCUS, focus_val)
                time.sleep(0.5)  # Give camera time to adjust focus
                
                # Capture multiple frames and average the focus measures
                frame_focus_measures = []
                frame_gradient_measures = []
                
                for _ in range(3):
                    ret, frame = cap.read()
                    if not ret:
                        raise Exception("Failed to capture frame during focus test")
                    
                    # Calculate new center for high-res frame to ensure we're analyzing the face
                    high_res_h, high_res_w = frame.shape[:2]
                    
                    # Draw rectangle on frame to show the region being analyzed (for debugging)
                    debug_frame = frame.copy()
                    cv2.rectangle(debug_frame, (scaled_left, scaled_top), (scaled_right, scaled_bottom), (0, 255, 0), 2)
                    cv2.imshow("Focus Region", debug_frame)
                    cv2.waitKey(1)  # Show briefly without blocking
                    
                    # Extract the region - use safe coordinates
                    try:
                        face_frame = frame[scaled_top:scaled_bottom, scaled_left:scaled_right]
                        if face_frame.size == 0:
                            raise ValueError("Extracted region is empty")
                    except Exception as e:
                        print(f"Error extracting region: {e}")
                        print(f"Frame shape: {frame.shape}, Region: ({scaled_top}:{scaled_bottom}, {scaled_left}:{scaled_right})")
                        # Use a safe region in the center as fallback
                        center_y, center_x = high_res_h // 2, high_res_w // 2
                        region_size = min(high_res_h, high_res_w) // 3
                        safe_top = center_y - region_size
                        safe_bottom = center_y + region_size
                        safe_left = center_x - region_size
                        safe_right = center_x + region_size
                        face_frame = frame[safe_top:safe_bottom, safe_left:safe_right]
                    
                    if _ == 0:  # Save first frame for visualization
                        focus_images.append(face_frame.copy())
                    
                    # Calculate focus measure (using Laplacian variance)
                    gray_face = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate image gradient information (key for 3D vs 2D detection)
                    sobelx = cv2.Sobel(gray_face, cv2.CV_64F, 1, 0, ksize=3)
                    sobely = cv2.Sobel(gray_face, cv2.CV_64F, 0, 1, ksize=3)
                    
                    # Magnitude of gradient
                    gradient_magnitude = cv2.magnitude(sobelx, sobely)
                    
                    # Calculate gradient consistency and distribution
                    # Photos usually have more uniform gradient distribution
                    gradient_mean = np.mean(gradient_magnitude)
                    gradient_std = np.std(gradient_magnitude)
                    gradient_ratio = gradient_std / (gradient_mean + 1e-5)
                    
                    # Real faces show more gradient variation across different focus settings
                    frame_gradient_measures.append(gradient_ratio)
                    
                    # Traditional focus measure
                    focus_measure = self.variance_of_laplacian(gray_face)
                    frame_focus_measures.append(focus_measure)
                    time.sleep(0.1)
                
                # Use median focus measure to reduce outliers
                median_focus = np.median(frame_focus_measures)
                median_gradient = np.median(frame_gradient_measures)
                
                focus_measurements.append(median_focus)
                gradient_measures.append(median_gradient)
                
                print(f"Focus setting {focus_val}: measure = {median_focus:.2f}, gradient = {median_gradient:.4f}")
            
            # Analyze focus pattern across measurements
            max_focus = max(focus_measurements)
            min_focus = min(focus_measurements)
            focus_range = max_focus - min_focus
            
            # Calculate focus pattern consistency
            # Photos tend to show more consistent focusing pattern 
            focus_std = np.std(focus_measurements)
            focus_variance_ratio = focus_std / (np.mean(focus_measurements) + 1e-5)
            
            # Calculate gradient pattern across focus settings
            # Real faces show more variation in gradient measures across focus settings
            gradient_range = max(gradient_measures) - min(gradient_measures)
            gradient_std = np.std(gradient_measures)
            
            print(f"Focus range: {focus_range:.2f}, Focus variance ratio: {focus_variance_ratio:.4f}")
            print(f"Gradient range: {gradient_range:.4f}, Gradient std: {gradient_std:.4f}")
            
            # Combined decision metrics
            # Based on updated real-world data from multiple tests
            # Real faces consistently show focus_variance_ratio < 0.025 and gradient_range < 0.04
            is_real_face = (focus_variance_ratio < 0.025 and gradient_range < 0.04)
            
            print(f"Focus test result: {is_real_face}")
            
            # For debugging: Display the focus images side by side
            if len(focus_images) >= 3:
                # Take subset of images for display
                display_images = [focus_images[0], focus_images[len(focus_images)//2], focus_images[-1]]
                
                # Resize images to the same size
                h, w = display_images[0].shape[:2]
                combined_img = np.zeros((h, w*3, 3), dtype=np.uint8)
                for i, img in enumerate(display_images):
                    combined_img[0:h, i*w:(i+1)*w] = img
                    # Add focus measure text
                    focus_idx = 0 if i == 0 else (len(focus_measurements)//2 if i == 1 else -1)
                    cv2.putText(combined_img, f"Focus: {focus_measurements[focus_idx]:.2f}", 
                               (i*w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add focus setting text
                    focus_setting = focus_values[0] if i == 0 else (focus_values[len(focus_values)//2] if i == 1 else focus_values[-1])
                    cv2.putText(combined_img, f"Focus: {focus_setting}", 
                              (i*w + 10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Draw rectangle to show this is the analyzed region
                    cv2.rectangle(combined_img, (i*w, 0), (i*w + w, h), (0, 255, 0), 2)
                
                # Add header
                header_img = np.zeros((50, w*3, 3), dtype=np.uint8)
                cv2.putText(header_img, "FOCUS TEST IMAGES", (w*3//2 - 150, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                
                # Combine header and images
                final_display = np.vstack((header_img, combined_img))
                
                cv2.imwrite("focus_test_debug.jpg", final_display)
                cv2.imshow("Focus Test Images", final_display)
                cv2.waitKey(1)  # Display briefly but don't block
            
            # Restore original settings
            cap.set(cv2.CAP_PROP_AUTOFOCUS, original_autofocus)
            cap.set(cv2.CAP_PROP_FOCUS, original_focus)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, original_resolution[0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, original_resolution[1])
            
            return is_real_face
            
        except Exception as e:
            print(f"Error in focus depth check: {e}")
            print(traceback.format_exc())
            return False  # Default to fail on error
            
    def variance_of_laplacian(self, image):
        """
        Compute the Laplacian of the image and return the variance.
        This is a measure of image focus/sharpness.
        """
        return cv2.Laplacian(image, cv2.CV_64F).var()
    
    def analyze_facial_texture(self, face_image):
        """
        Enhanced texture analysis that better distinguishes between real faces and photos.
        Uses multi-scale texture analysis, micro-texture patterns, and reflectance properties.
        """
        try:
            print("Starting advanced texture analysis...")
            # Convert to grayscale
            gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
            
            # Apply histogram equalization to enhance texture
            equalized = cv2.equalizeHist(gray)
            
            # Apply Gaussian blur to reduce noise
            blurred = cv2.GaussianBlur(equalized, (5, 5), 0)
            
            # Calculate Local Binary Pattern at multiple scales
            # This helps capture both micro and macro texture patterns
            results = {}
            
            # Original LBP calculation
            radius1 = 1
            n_points1 = 8 * radius1
            lbp1 = skimage_feature.local_binary_pattern(blurred, n_points1, radius1, method="uniform")
            
            # Medium scale LBP - captures more structure
            radius2 = 2
            n_points2 = 8 * radius2
            lbp2 = skimage_feature.local_binary_pattern(blurred, n_points2, radius2, method="uniform")
            
            # Larger scale LBP - captures even more structure
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
            # Calculate uniformity metrics (sum of squared probabilities)
            uniformity1 = np.sum(hist1 * hist1)
            uniformity2 = np.sum(hist2 * hist2)
            uniformity3 = np.sum(hist3 * hist3)
            
            # Calculate multi-scale uniformity ratio - key for detecting printed materials
            uniformity_ratio = (uniformity1 + uniformity2) / (2 * uniformity3 + 1e-7)
            
            # Analyze reflectance properties using gradient information
            # Photos typically have more uniform gradient distribution
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = cv2.magnitude(sobelx, sobely)
            
            # Calculate micro-texture features that differentiate real skin
            gradient_mean = np.mean(magnitude)
            gradient_std = np.std(magnitude)
            gradient_entropy = self.calculate_entropy(magnitude)
            
            # For real skin, gradient distribution is distinctive
            gradient_ratio = gradient_std / (gradient_mean + 1e-7)
            
            # Analyze frequency distribution using FFT
            # Printed/displayed faces often have regular patterns or moiré effects
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Split spectrum into high/low frequency regions
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Define high-frequency region (outer 50% of spectrum)
            mask_high = np.zeros((h, w), dtype=np.uint8)
            mask_radius = min(center_h, center_w) // 2
            cv2.circle(mask_high, (center_w, center_h), mask_radius, 1, -1)
            mask_high = 1 - mask_high  # Invert to get high frequencies
            
            # Calculate energy in different frequency bands
            high_freq_energy = np.sum(magnitude_spectrum * mask_high) / np.sum(mask_high)
            total_energy = np.mean(magnitude_spectrum)
            freq_energy_ratio = high_freq_energy / (total_energy + 1e-7)
            
            # For debugging
            print(f"Texture entropy: multi-scale {entropy1:.2f}/{entropy2:.2f}/{entropy3:.2f}")
            print(f"Uniformity ratio: {uniformity_ratio:.4f}, Gradient ratio: {gradient_ratio:.4f}")
            print(f"Frequency energy ratio: {freq_energy_ratio:.4f}")
            
            # Combined decision metrics using multiple texture properties
            # Thresholds adjusted based on extensive real test data
            
            # 1. Real faces have higher entropy values, especially at larger scales
            entropy_score = (entropy2 > 3.65 and entropy3 > 4.10)
            
            # 2. Real faces show higher gradient ratio based on all test results
            gradient_score = (gradient_ratio > 1.25)
            
            # 3. Real faces show higher uniformity ratio consistently
            uniformity_score = (uniformity_ratio > 1.85)
            
            # 4. Real faces show lower frequency energy ratio
            frequency_score = (freq_energy_ratio < 0.952)
            
            # Compute scores and final decision
            # Need at least 2 of 4 texture metrics to pass, with entropy being critical
            passing_scores = sum([entropy_score, gradient_score, uniformity_score, frequency_score])
            print(f"Texture scores: Entropy={entropy_score}, Gradient={gradient_score}, "
                  f"Uniformity={uniformity_score}, Frequency={frequency_score}")
            print(f"Passing texture scores: {passing_scores}/4")
            
            # Require at least 2 metrics and EITHER entropy OR gradient score (the most reliable indicators)
            is_real_texture = passing_scores >= 2 and (entropy_score or gradient_score)
            
            print(f"Texture test result: {is_real_texture}")
            
            # Save debug visualization
            # Create more informative visualization showing multiple scales
            micro_texture = cv2.normalize(lbp1.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            macro_texture = cv2.normalize(lbp3.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            gradient_viz = cv2.normalize(magnitude.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX)
            
            # Create visualization for FFT
            freq_viz = np.log(np.abs(f_shift) + 1)
            freq_viz = cv2.normalize(freq_viz, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            
            # Create a nicer visualization with title and organized layout
            h, w = face_image.shape[:2]
            viz_size = (w, h)
            
            # Create labels for each image
            labels = [
                "Original Image", "Grayscale", "Histogram Equalized",
                "LBP (r=1)", "LBP (r=2)", "LBP (r=3)",
                "Gradient Magnitude", "Frequency Spectrum"
            ]
            
            # Resize each image to the same size
            face_resized = cv2.resize(face_image, viz_size)
            gray_resized = cv2.resize(gray, viz_size)
            equalized_resized = cv2.resize(equalized, viz_size)
            
            # Make sure LBP visualizations are 3-channel
            lbp1_viz = cv2.resize(cv2.cvtColor(micro_texture, cv2.COLOR_GRAY2BGR), viz_size)
            lbp2_viz = cv2.resize(cv2.cvtColor(cv2.normalize(lbp2.astype(np.uint8), None, 0, 255, cv2.NORM_MINMAX), cv2.COLOR_GRAY2BGR), viz_size)
            lbp3_viz = cv2.resize(cv2.cvtColor(macro_texture, cv2.COLOR_GRAY2BGR), viz_size)
            
            gradient_viz = cv2.resize(cv2.cvtColor(gradient_viz, cv2.COLOR_GRAY2BGR), viz_size)
            freq_viz = cv2.resize(cv2.cvtColor(freq_viz, cv2.COLOR_GRAY2BGR), viz_size)
            
            # Make sure gray images are 3-channel for display
            gray_resized = cv2.cvtColor(gray_resized, cv2.COLOR_GRAY2BGR)
            equalized_resized = cv2.cvtColor(equalized_resized, cv2.COLOR_GRAY2BGR)
            
            # Create the visualization layout
            row1 = np.hstack([face_resized, gray_resized, equalized_resized])
            row2 = np.hstack([lbp1_viz, lbp2_viz, lbp3_viz])
            row3 = np.hstack([gradient_viz, freq_viz, np.zeros_like(freq_viz)])
            
            # Create header with title
            header = np.zeros((60, row1.shape[1], 3), dtype=np.uint8)
            cv2.putText(header, "TEXTURE ANALYSIS VISUALIZATION", (row1.shape[1]//2 - 200, 40), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Add labels to each image
            for i, img in enumerate([row1, row2, row3]):
                for j in range(3):
                    if i * 3 + j < len(labels):
                        label = labels[i * 3 + j]
                        x = j * viz_size[0] + 10
                        y = 30
                        cv2.putText(img, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Add test results at the bottom of the image
            footer = np.zeros((60, row1.shape[1], 3), dtype=np.uint8)
            entropy_text = f"Entropy (r1/r2/r3): {entropy1:.2f}/{entropy2:.2f}/{entropy3:.2f}"
            cv2.putText(footer, entropy_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            # Add pass/fail indicator
            entropy_result = "PASS" if entropy_score else "FAIL"
            entropy_color = (0, 255, 0) if entropy_score else (0, 0, 255)
            cv2.putText(footer, f"Entropy test: {entropy_result}", (row1.shape[1] - 250, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, entropy_color, 2)
            
            # Combine everything
            texture_visualization = np.vstack([header, row1, row2, row3, footer])
            
            # Save and display
            cv2.imwrite("texture_analysis_debug.jpg", texture_visualization)
            cv2.imshow("Texture Analysis", texture_visualization)
            cv2.waitKey(1)  # Show image but don't block
            
            return is_real_texture
            
        except Exception as e:
            print(f"Error in texture analysis: {e}")
            print(traceback.format_exc())
            return False  # Default to fail on error
            
    def calculate_entropy(self, image):
        """Calculate entropy of an image"""
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
        hist = hist.astype(float) / (np.sum(hist) + 1e-7)
        return -np.sum(hist * np.log2(hist + 1e-7))
    
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
        # Use lower resolution for blink detection to maintain high framerate
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
        
        # Variables for blink detection
        blink_counter = 0
        ear_thresh_counter = 0
        blink_detected = False
        start_time = time.time()
        
        # Keep the best face image for return
        best_face_image = None
        max_face_size = 0
        face_location = None
        
        # For tracking
        frame_count = 0
        
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
        
        # For adaptive EAR threshold
        ear_values = []
        
        # Position smoothing to reduce jitter
        position_history = []
        position_history_max = 5
        
        # Track FPS
        fps_start_time = time.time()
        fps_frame_count = 0
        fps = 0
        last_face_location = None
        last_landmarks = None
        
        # Initialize ear variable outside the scope to prevent reference errors
        ear = 0.0
        
        # Create tracker
        tracker = self.create_tracker()
        tracking_active = False
        
        # Process every frame for better blink detection reliability
        while (time.time() - start_time) < timeout:
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
                # Use lower resolution for motion detection to save CPU
                motion_scale = 0.25
                small_prev = cv2.resize(previous_frame, (0, 0), fx=motion_scale, fy=motion_scale)
                small_curr = cv2.resize(frame, (0, 0), fx=motion_scale, fy=motion_scale)
                
                gray1 = cv2.cvtColor(small_prev, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(small_curr, cv2.COLOR_BGR2GRAY)
                
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
                    # Adjust motion threshold based on testing
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
            
            # Alternate between tracking and detection to maintain performance
            process_this_frame = (frame_count % 3 == 0)
            frame_count += 1
            
            # Try to update tracker if active
            current_face_box = None
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
                # Detect face locations with efficient downsampling
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                if face_locations:
                    # Scale back up face locations
                    top, right, bottom, left = face_locations[0]
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4
                    
                    current_face_box = (top, right, bottom, left)
                    face_location = current_face_box  # Store this for later focus check
                    
                    # Initialize tracker if available
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
                    
                    # Get landmarks for blink detection
                    landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [face_locations[0]])
                    
                    if landmarks_list:
                        # Convert landmarks to full scale
                        scaled_landmarks = {}
                        for feature, points in landmarks_list[0].items():
                            scaled_landmarks[feature] = [(p[0] * 4, p[1] * 4) for p in points]
                        
                        last_landmarks = scaled_landmarks
            
            # Apply smoothing for face position
            if current_face_box is not None:
                position_history.append(current_face_box)
                if len(position_history) > position_history_max:
                    position_history.pop(0)
                
                # Calculate smoothed position
                if len(position_history) > 1:
                    smoothed_top = sum(pos[0] for pos in position_history) // len(position_history)
                    smoothed_right = sum(pos[1] for pos in position_history) // len(position_history)
                    smoothed_bottom = sum(pos[2] for pos in position_history) // len(position_history)
                    smoothed_left = sum(pos[3] for pos in position_history) // len(position_history)
                    
                    last_face_location = (smoothed_top, smoothed_right, smoothed_bottom, smoothed_left)
                else:
                    last_face_location = current_face_box
            
            # Use the last detected face and landmarks for blink processing
            if last_face_location is not None:
                top, right, bottom, left = last_face_location
                
                # Draw rectangle around face
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                # Save the largest face image for return
                face_size = (right - left) * (bottom - top)
                if face_size > max_face_size:
                    max_face_size = face_size
                    best_face_image = frame.copy()
                
                # Process eye landmarks if available
                if last_landmarks is not None:
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
                        
                        # Add to baseline values for adaptive thresholding
                        ear_values.append(ear)
                        
                        # Calculate adaptive threshold after sufficient samples
                        adaptive_threshold = self.EAR_THRESHOLD
                        if len(ear_values) > 10:
                            # Use the 70th percentile as our baseline open eye EAR
                            open_eye_ear = np.percentile(ear_values, 70)
                            # Set threshold to 75% of the open eye EAR
                            adaptive_threshold = open_eye_ear * 0.75
                            # But don't go below the minimum threshold
                            adaptive_threshold = max(adaptive_threshold, self.EAR_THRESHOLD)
                        
                        # Draw eye contours
                        for eye in [left_eye, right_eye]:
                            eye_hull = cv2.convexHull(np.array(eye))
                            cv2.drawContours(display_frame, [eye_hull], -1, (0, 255, 0), 1)
                        
                        # Only process blinks when relatively still
                        if is_still_enough and len(ear_history) >= 3:
                            # Check if current EAR is below threshold (eyes closed)
                            if ear < adaptive_threshold:
                                ear_thresh_counter += 1
                                # Draw RED eye contours for closed eyes
                                for eye in [left_eye, right_eye]:
                                    eye_hull = cv2.convexHull(np.array(eye))
                                    cv2.drawContours(display_frame, [eye_hull], -1, (0, 0, 255), 2)
                            else:
                                # Eyes are open now, check if it was a blink
                                if ear_thresh_counter >= self.EAR_CONSEC_FRAMES:
                                    # Check for natural blink pattern
                                    if len(ear_history) >= 5:
                                        # Calculate EAR differences to verify natural transition
                                        ear_diff = abs(ear_history[-1] - ear_history[-3])
                                        ear_diff2 = abs(ear_history[-3] - ear_history[-5])
                                        
                                        # Verify blink pattern
                                        natural_blink_pattern = (
                                            # Sequence: open -> closed -> open
                                            ear_history[-5] > ear_history[-3] and 
                                            ear_history[-3] < ear_history[-1] and
                                            # Check reasonable differences
                                            ear_diff > 0.01 and ear_diff2 > 0.005 and
                                            # Real blinks have a larger change
                                            abs(max(ear_history[-5:]) - min(ear_history[-5:])) > 0.03
                                        )
                                        
                                        if natural_blink_pattern:
                                            blink_counter += 1
                                            print(f"Blink detected! EAR: {ear:.3f}, Threshold: {adaptive_threshold:.3f}")
                                            blink_detected = True
                                
                                # Reset counter after eyes reopen
                                ear_thresh_counter = 0
                
                # Display EAR value and blink count
                ear_text = f"EAR: {ear:.2f}"
                cv2.putText(display_frame, ear_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display blink counter
                cv2.putText(display_frame, f"Blinks: {blink_counter}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Display time remaining
                time_left = int(timeout - (time.time() - start_time))
                cv2.putText(display_frame, f"Time: {time_left}s", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display FPS
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 120, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
                # Display motion status
                motion_status = "High Motion" if high_motion_detected else "Stable"
                motion_color = (0, 0, 255) if high_motion_detected else (0, 255, 0)
                cv2.putText(display_frame, motion_status, (display_frame.shape[1] - 120, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, motion_color, 1)
                
                # Display blink status
                if blink_counter > 0:
                    status = f"Blink detected! ({blink_counter})"
                    color = (0, 255, 0)
                else:
                    status = "Please blink naturally..."
                    color = (0, 0, 255)
                cv2.putText(display_frame, status, (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                
                # Mark the area that will be used for focus testing
                # This helps the user see what region will be analyzed
                face_center_x = left + (right - left) // 2
                face_center_y = top + (bottom - top) // 2
                focus_test_size = min(right - left, bottom - top) * 4 // 5
                focus_left = max(0, face_center_x - focus_test_size // 2)
                focus_top = max(0, face_center_y - focus_test_size // 2)
                focus_right = focus_left + focus_test_size
                focus_bottom = focus_top + focus_test_size
                
                # Draw rectangle around focus test region with dashed lines
                dash_length = 5
                for i in range(0, focus_test_size, dash_length * 2):
                    # Top line
                    cv2.line(display_frame, (focus_left + i, focus_top), 
                            (min(focus_left + i + dash_length, focus_right), focus_top), (0, 255, 255), 1)
                    # Bottom line
                    cv2.line(display_frame, (focus_left + i, focus_bottom), 
                            (min(focus_left + i + dash_length, focus_right), focus_bottom), (0, 255, 255), 1)
                    
                    if i < focus_test_size:  # For vertical lines
                        # Left line
                        cv2.line(display_frame, (focus_left, focus_top + i), 
                                (focus_left, min(focus_top + i + dash_length, focus_bottom)), (0, 255, 255), 1)
                        # Right line
                        cv2.line(display_frame, (focus_right, focus_top + i), 
                                (focus_right, min(focus_top + i + dash_length, focus_bottom)), (0, 255, 255), 1)
                
                # Add label for focus region
                cv2.putText(display_frame, "Focus Test Region", (focus_left, focus_top - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                
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
                cap.release()
                cv2.destroyAllWindows()
                return False, None
            
            # Early exit if enough blinks detected
            if blink_counter >= 2 and time.time() - start_time > timeout/2:
                print("Multiple blinks detected - proceeding to additional checks")
                break
        
        # Clean up from blink detection
        cap.release()
        cv2.destroyAllWindows()
        
        # STEP 2: Only perform focus and texture tests AFTER blink detection is complete
        # This keeps the framerate high during the critical blink detection phase
        self.focus_check_passed = False
        
        if blink_detected and face_location is not None:
            # Create new camera for these tests to ensure clean state
            focus_cap = cv2.VideoCapture(self.camera_id)
            
            # Wait for camera to initialize
            time.sleep(0.5)
            
            print("Blink check complete. Now performing additional security checks...")
            
            # Run focus test first if we have a valid face
            if focus_cap.isOpened():
                # Capture fresh frame for texture analysis
                ret, texture_frame = focus_cap.read()
                
                if ret:
                    # Run focus and texture tests
                    focus_result = self.perform_focus_liveness_check(focus_cap, face_location)
                    texture_result = self.analyze_facial_texture(texture_frame)
                    
                    # Store focus result
                    self.focus_check_passed = focus_result
                    
                    # Final liveness decision
                    # Multiple blinks are required
                    # But we only need one of the other tests to pass - either focus or texture
                    # If a test is borderline, look at the test scores for a more nuanced decision
                    liveness_confirmed = False
                    
                    if blink_counter >= 2:
                        # Strong success: both supplementary tests pass
                        if focus_result and texture_result:
                            liveness_confirmed = True
                            print("STRONG PASS: All liveness tests passed")
                        # Good success: focus test passes (more reliable than texture)
                        elif focus_result:
                            liveness_confirmed = True
                            print("PASS: Blink detection and focus test passed")
                        # Acceptable success: texture test passes with good scores
                        elif texture_result:
                            liveness_confirmed = True
                            print("PASS: Blink detection and texture test passed")
                        # Failure: neither supplementary test passed
                        else:
                            print("FAIL: Blink detection passed but other tests failed")
                    else:
                        print("FAIL: Not enough blinks detected")
                    
                    # Display results
                    result_frame = best_face_image.copy() if best_face_image is not None else np.zeros((480, 640, 3), dtype=np.uint8)
                    
                    # Display final liveness results
                    if liveness_confirmed:
                        result_message = "PASSED"
                        result_color = (0, 255, 0)
                    else:
                        result_message = "FAILED"
                        result_color = (0, 0, 255)
                        
                    cv2.putText(result_frame, f"LIVENESS CHECK {result_message}", 
                                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
                    cv2.putText(result_frame, f"Detected {blink_counter} blinks", 
                                (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Add focus test result
                    focus_result_text = "PASSED" if self.focus_check_passed else "FAILED"
                    focus_color = (0, 255, 0) if self.focus_check_passed else (0, 0, 255)
                    cv2.putText(result_frame, f"Focus Check: {focus_result_text}", 
                                (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, focus_color, 2)
                    
                    # Add texture test result
                    texture_result_text = "PASSED" if texture_result else "FAILED"
                    texture_color = (0, 255, 0) if texture_result else (0, 0, 255)
                    cv2.putText(result_frame, f"Texture Check: {texture_result_text}", 
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, texture_color, 2)
                
                # Clean up focus camera
                focus_cap.release()
            
            # Final step: Show results and wait for user acknowledgment
            cv2.putText(result_frame, "Press any key to continue...", 
                        (20, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Liveness Result", result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Return the image only if liveness is confirmed
            return liveness_confirmed, best_face_image
        else:
            # Show failed result
            result_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(result_frame, "LIVENESS CHECK FAILED", 
                        (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(result_frame, "No blinks detected.", 
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(result_frame, "Press any key to continue...", 
                        (20, result_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("Liveness Result", result_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
            # Return the image only if liveness is confirmed
            return False, None
    
    def cleanup_diagnostic_images(self):
        """Clean up diagnostic images created during focus testing"""
        try:
            # Focus levels used in the test
            focus_levels = [0, 125, 250]
            
            # Remove all focus test images
            files_to_clean = [f"focus_test_{level}.jpg" for level in focus_levels]
            files_to_clean.extend(["focus_test_collage.jpg", "focus_test_result.jpg"])
            
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

    def check_liveness(self, frame, face_location):
        """Perform liveness detection on the detected face."""
        try:
            print("\nPerforming liveness checks...")
            
            # Get video capture object
            cap = self.video_capture
            
            # Step 1: Conduct eye blink detection (basic)
            print("Starting blink detection...")
            blink_result = self.detect_blinks(cap, face_location)
            
            # Step 2: Check focus depth response
            print("Starting focus depth check...")
            focus_result = self.check_focus_depth(cap, face_location)
            
            # Step 3: Check facial texture
            print("Starting texture analysis...")  
            texture_result = self.analyze_facial_texture(frame)
            
            # Weighted decision based on test results
            # We require at least 2 of 3 checks to pass for robust liveness detection
            checks_passed = sum([blink_result, focus_result, texture_result])
            liveness_confirmed = checks_passed >= 2
            
            # Log results
            print("\nLiveness Check Results:")
            print(f"✓ Blink Detection: {'PASS' if blink_result else 'FAIL'}")
            print(f"✓ Focus Test: {'PASS' if focus_result else 'FAIL'}")
            print(f"✓ Texture Analysis: {'PASS' if texture_result else 'FAIL'}")
            print(f"Overall: {checks_passed}/3 checks passed")
            print(f"Final Decision: {'LIVE FACE CONFIRMED' if liveness_confirmed else 'SPOOFING ATTEMPT DETECTED'}")
            
            # Display result on frame
            top, right, bottom, left = face_location
            label = "LIVE FACE" if liveness_confirmed else "FAKE DETECTED"
            color = (0, 255, 0) if liveness_confirmed else (0, 0, 255)
            cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
            
            return liveness_confirmed
            
        except Exception as e:
            print(f"Error in liveness check: {e}")
            print(traceback.format_exc())
            return False  # Default to rejecting on error

# For demonstration
if __name__ == "__main__":
    camera = CameraSystem()
    
    # Empty database to start
    known_encodings = []
    known_names = []
    
    # Run facial recognition
    known_encodings, known_names = camera.run_facial_recognition(known_encodings, known_names) 