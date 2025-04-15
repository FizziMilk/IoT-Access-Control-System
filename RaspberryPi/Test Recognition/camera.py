import cv2
import time
import face_recognition
import numpy as np
import dlib
from scipy.spatial import distance as dist
import threading
import traceback

class CameraSystem:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
        # EAR threshold for blink detection
        self.EAR_THRESHOLD = 0.25  # Higher threshold to detect blinks more easily
        # Number of consecutive frames the eye must be below threshold to count as a blink
        self.EAR_CONSEC_FRAMES = 1  # Reduced to detect faster blinks
        # Use a much lower resolution for liveness detection
        self.liveness_resolution = (320, 240)
        # Frame process rate (1 = process every frame, 2 = every other frame, etc.)
        self.process_nth_frame = 3
        # Use a separate thread for face detection to prevent UI blocking
        self.face_detection_thread = None
        self.stop_detection_thread = False
    
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
        """Create a tracking object compatible with the installed OpenCV version"""
        # Print OpenCV version for diagnostic purposes
        print(f"OpenCV version: {cv2.__version__}")
        
        try:
            # Try the best performing tracker first (CSRT - accurate but slower)
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT'):
                print("Using legacy.TrackerCSRT - best accuracy")
                return cv2.legacy.TrackerCSRT.create()
            elif hasattr(cv2, 'TrackerCSRT'):
                print("Using TrackerCSRT - best accuracy")
                return cv2.TrackerCSRT.create()
                
            # Try KCF next (good balance of speed and accuracy)
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerKCF'):
                print("Using legacy.TrackerKCF - good balance")
                return cv2.legacy.TrackerKCF.create()
            elif hasattr(cv2, 'TrackerKCF'):
                print("Using TrackerKCF - good balance")
                return cv2.TrackerKCF.create()
                
            # Try MOSSE (fastest but less accurate)
            if hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerMOSSE'):
                print("Using legacy.TrackerMOSSE - fastest")
                return cv2.legacy.TrackerMOSSE.create()
            elif hasattr(cv2, 'TrackerMOSSE'):
                print("Using TrackerMOSSE - fastest")
                return cv2.TrackerMOSSE.create()
                
            print("No suitable tracker found in opencv-contrib-python")
            return None
            
        except Exception as e:
            print(f"Error creating tracker: {e}")
            print("Traceback:", traceback.format_exc())
            return None
    
    def detect_blink(self, timeout=7):
        """
        Detect if a person is blinking (liveness check)
        
        Args:
            timeout: Timeout in seconds
            
        Returns:
            tuple: (success, face_image)
        """
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        # Use lower resolution for liveness detection
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
        downsample = 0.3  # Even smaller for better performance
        
        # Frame counter for processing only every nth frame
        frame_count = 0
        
        # Store the last processed face location to use for intermediate frames
        last_face_location = None
        last_landmarks = None
        
        # For adaptive EAR threshold
        ear_values = []
        
        # Try to create a tracker
        tracker = self.create_tracker()
        tracking_active = False
        
        # If tracker is available, process fewer frames
        if tracker is not None:
            self.process_nth_frame = 5  # Process every 5th frame for detection
            print(f"Tracking enabled - processing every {self.process_nth_frame}th frame")
        else:
            # Process more frames if tracking isn't available
            self.process_nth_frame = 2
            print("Tracking not available - processing every 2nd frame")
        
        # Increase timeout slightly to allow for adaptation
        actual_timeout = timeout + 0.5
        
        while (time.time() - start_time) < actual_timeout:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Create a copy for display
            display_frame = frame.copy()
            
            # Only process every nth frame to improve performance
            process_this_frame = (frame_count % self.process_nth_frame == 0)
            frame_count += 1
            
            # Try to update tracker if active
            if tracker is not None and tracking_active:
                try:
                    ok, bbox = tracker.update(frame)
                    if ok:
                        # Convert from (x, y, width, height) to (top, right, bottom, left)
                        x, y, w, h = [int(v) for v in bbox]
                        left, top, right, bottom = x, y, x + w, y + h
                        last_face_location = (top, right, bottom, left)
                    else:
                        # If tracking failed, force detection on next frame
                        tracking_active = False
                        process_this_frame = True
                except Exception as e:
                    print(f"Tracking error: {e}")
                    tracking_active = False
                    process_this_frame = True
            
            # Run face detection when needed
            if (process_this_frame or not tracking_active) and tracker is not None:
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
                    
                    last_face_location = (top, right, bottom, left)
                    
                    # Initialize or reinitialize tracker
                    if tracker is not None:
                        try:
                            # Release previous tracker
                            tracker = self.create_tracker()
                            if tracker is not None:
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
            # If no tracker is available, always use face detection
            elif tracker is None and process_this_frame:
                small_frame = cv2.resize(frame, (0, 0), fx=downsample, fy=downsample)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
                
                if face_locations:
                    top, right, bottom, left = face_locations[0]
                    top = int(top / downsample)
                    right = int(right / downsample)
                    bottom = int(bottom / downsample)
                    left = int(left / downsample)
                    
                    last_face_location = (top, right, bottom, left)
                    
                    landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [face_locations[0]])
                    
                    if landmarks_list:
                        scaled_landmarks = {}
                        for feature, points in landmarks_list[0].items():
                            scaled_landmarks[feature] = [(int(p[0] / downsample), int(p[1] / downsample)) for p in points]
                        
                        last_landmarks = scaled_landmarks
            
            # Use the last detected face and landmarks for processing (even in frames we didn't run detection)
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
                
                # Process landmarks if available
                if last_landmarks is not None:
                    # Get eye landmarks
                    left_eye, right_eye = self.get_eye_landmarks(last_landmarks)
                    
                    if left_eye and right_eye:
                        # Calculate eye aspect ratio (EAR)
                        left_ear = self.eye_aspect_ratio(left_eye)
                        right_ear = self.eye_aspect_ratio(right_eye)
                        
                        # Average the EAR of both eyes
                        ear = (left_ear + right_ear) / 2.0
                        
                        # Add to history for adaptive thresholding
                        ear_values.append(ear)
                        
                        # Calculate adaptive threshold if we have enough data
                        # (only after the first second or so)
                        adaptive_threshold = self.EAR_THRESHOLD
                        if len(ear_values) > 10:
                            # Use the 70th percentile as our baseline open eye EAR
                            open_eye_ear = np.percentile(ear_values, 70)
                            # Set threshold to 80% of the open eye EAR
                            adaptive_threshold = open_eye_ear * 0.8
                            # But don't go below the minimum threshold
                            adaptive_threshold = max(adaptive_threshold, self.EAR_THRESHOLD)
                        
                        # Draw eye contours
                        for eye in [left_eye, right_eye]:
                            eye_hull = cv2.convexHull(np.array(eye))
                            cv2.drawContours(display_frame, [eye_hull], -1, (0, 255, 0), 1)
                        
                        # Check if EAR is below threshold
                        if ear < adaptive_threshold:
                            ear_thresh_counter += 1
                            # Draw RED eye contours when eyes are detected as closed
                            for eye in [left_eye, right_eye]:
                                eye_hull = cv2.convexHull(np.array(eye))
                                cv2.drawContours(display_frame, [eye_hull], -1, (0, 0, 255), 2)
                        else:
                            # If eyes were closed for enough frames, count as a blink
                            if ear_thresh_counter >= self.EAR_CONSEC_FRAMES:
                                blink_counter += 1
                            ear_thresh_counter = 0
                        
                        # Check if we've detected a blink
                        if blink_counter > 0:
                            blink_detected = True
                            
                        # Display EAR value and blink count
                        cv2.putText(display_frame, f"EAR: {ear:.2f}", (10, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.putText(display_frame, f"Thresh: {adaptive_threshold:.2f}", (display_frame.shape[1] - 190, 30),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                        cv2.putText(display_frame, f"Blinks: {blink_counter}", (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display time remaining
            time_left = int(actual_timeout - (time.time() - start_time))
            cv2.putText(display_frame, f"Time: {time_left}s", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Display frame rate (every ~10 frames)
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_time)
                cv2.putText(display_frame, f"FPS: {fps:.1f}", (display_frame.shape[1] - 120, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display instructions
            if blink_detected:
                status = f"Blink detected! ({blink_counter} blinks) - Continuing for testing"
                color = (0, 255, 0)
            else:
                status = "Please blink..."
                color = (0, 0, 255)
            cv2.putText(display_frame, status, (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
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
            
            # Continue running even after blink is detected (removed early exit)
            # Save the face image when a blink is first detected
            if blink_detected and best_face_image is None and max_face_size > 0:
                best_face_image = frame.copy()
        
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # If we've detected a blink but don't have a good face image, capture one now
        if blink_detected and best_face_image is None:
            best_face_image = self.capture_face()
        
        return blink_detected, best_face_image
        
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