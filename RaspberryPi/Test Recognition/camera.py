import cv2
import time
import face_recognition
import numpy as np
from collections import deque
import random

class CameraSystem:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
    
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
    
    def detect_texture(self, image, debug_output=True):
        """
        Detect if the image is a print/photo based on texture analysis
        Returns true if it appears to be a real face, false if it looks like a photo
        """
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian variance (texture detail analysis)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_variance = laplacian.var()
        
        # 2. Screen pattern detection (look for pixel grid patterns)
        # Apply high-pass filter to enhance screen patterns
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        
        # Check for regular patterns using FFT
        f_transform = np.fft.fft2(sharpened)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20*np.log(np.abs(f_shift))
        
        # Get the high-frequency components (excluding the DC component)
        center_y, center_x = magnitude_spectrum.shape[0]//2, magnitude_spectrum.shape[1]//2
        mask_size = 10
        center_mask = magnitude_spectrum[center_y-mask_size:center_y+mask_size, center_x-mask_size:center_x+mask_size]
        high_freq_energy = np.sum(magnitude_spectrum) - np.sum(center_mask)
        
        # 3. Color variance analysis (real faces have more color variation than screens)
        color_std = np.std(image, axis=(0, 1)).mean()  # Average std dev across color channels
        
        # 4. Gradients analysis (real faces have smoother gradients)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        sobel_mag = np.sqrt(sobelx**2 + sobely**2)
        gradient_mean = np.mean(sobel_mag)
        
        # 5. Local Binary Pattern (LBP) analysis for texture patterns
        # Simple version of LBP
        shifted = np.roll(gray, 1, axis=0)
        lbp = np.zeros_like(gray)
        lbp[gray > shifted] = 1
        lbp_variance = np.var(lbp)
        
        # Higher thresholds for more strictness against screen photos
        texture_threshold = 150  # Increased from 70
        high_freq_threshold = 1e8  # Threshold for high frequency components
        color_std_threshold = 25   # Minimum color variation for real faces
        
        # Debug output
        if debug_output:
            print(f"Texture variance: {lap_variance:.2f} (threshold: {texture_threshold})")
            print(f"High frequency energy: {high_freq_energy:.2e} (threshold: {high_freq_threshold:.2e})")
            print(f"Color standard deviation: {color_std:.2f} (threshold: {color_std_threshold})")
            print(f"Gradient mean: {gradient_mean:.2f}")
            print(f"LBP variance: {lbp_variance:.2f}")
        
        # Combined decision based on multiple metrics
        texture_score = (lap_variance > texture_threshold)
        high_freq_score = (high_freq_energy < high_freq_threshold)  # Lower is better for real faces
        color_score = (color_std > color_std_threshold)
        
        # Calculate a combined score (require at least 2 positive indicators)
        positives = sum([texture_score, high_freq_score, color_score])
        
        # Show debug visualization if requested
        if debug_output:
            # Create a visualization of the analysis
            # Normalize for visualization
            lap_vis = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            sobel_vis = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            lbp_vis = (lbp * 255).astype(np.uint8)
            
            # Create visualization grid
            h, w = gray.shape
            debug_img = np.zeros((h*2, w*2, 3), dtype=np.uint8)
            
            # Original grayscale image (top-left)
            debug_img[0:h, 0:w, :] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            
            # Laplacian (top-right)
            debug_img[0:h, w:w*2, :] = cv2.cvtColor(lap_vis, cv2.COLOR_GRAY2BGR)
            
            # Sobel (bottom-left)
            debug_img[h:h*2, 0:w, :] = cv2.cvtColor(sobel_vis, cv2.COLOR_GRAY2BGR)
            
            # LBP (bottom-right)
            debug_img[h:h*2, w:w*2, :] = cv2.cvtColor(lbp_vis, cv2.COLOR_GRAY2BGR)
            
            # Add text labels
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(debug_img, "Original", (10, 30), font, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_img, f"Laplacian ({lap_variance:.1f})", (w+10, 30), font, 0.6, (0, 255, 0) if texture_score else (0, 0, 255), 2)
            cv2.putText(debug_img, f"Gradient ({gradient_mean:.1f})", (10, h+30), font, 0.6, (0, 255, 0), 2)
            cv2.putText(debug_img, f"LBP ({lbp_variance:.1f})", (w+10, h+30), font, 0.6, (0, 255, 0), 2)
            
            # Add overall result
            cv2.putText(debug_img, f"RESULT: {'REAL' if positives >= 2 else 'FAKE'} ({positives}/3)", 
                      (w//2-100, h*2-20), font, 0.8, (0, 255, 0) if positives >= 2 else (0, 0, 255), 2)
            
            # Show the debug visualization
            cv2.imshow("Texture Analysis Debug", debug_img)
            cv2.waitKey(1)  # Update the window
        
        # Return true if this looks like a real face
        return positives >= 2
    
    def detect_liveness_multi(self, timeout=30):  # Increased timeout from 20 to 30 seconds
        """
        Advanced liveness detection using multiple techniques:
        1. Blink detection with higher thresholds
        2. Random head movement challenges
        3. Texture analysis to detect printed photos
        
        Args:
            timeout: Maximum time in seconds for the entire verification
            
        Returns:
            tuple: (is_live, face_image)
        """
        print("Enhanced liveness detection started...")
        
        # Enable skipping for troubleshooting - set these to True to skip specific tests
        # In a production environment, these should all be False
        SKIP_TEXTURE = False    # For troubleshooting only
        SKIP_BLINK = False      # For troubleshooting only
        SKIP_MOVEMENT = False   # For troubleshooting only
        
        # Allow user to toggle which tests to run with keyboard during detection
        # These will be updated during the detection loop if user presses keys
        skip_texture = SKIP_TEXTURE
        skip_blink = SKIP_BLINK
        skip_movement = SKIP_MOVEMENT
        
        # Initialize stage tracking
        stage_passed = {
            "texture": skip_texture,
            "blink": skip_blink,
            "head_movement": skip_movement
        }
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
        # Parameters for blink detection - more strict thresholds
        EYE_AR_THRESH = 0.3  # Increased from 0.25 for lower framerates
        EYE_AR_CONSEC_FRAMES = 2
        
        # Store eye aspect ratios for blink detection
        ear_history = deque(maxlen=8)  # Longer history for better analysis
        
        # For head movement challenge
        movement_type = random.choice(["left", "right", "up", "down"])
        movement_start_time = None
        movement_completed = False
        movement_positions = []  # To track nose positions
        
        # Track progress and time
        start_time = time.time()
        current_stage = "init"
        face_image = None
        
        # Function to calculate eye aspect ratio
        def eye_aspect_ratio(eye):
            # Compute euclidean distance between eye landmarks
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            # Calculate ratio
            ear = (A + B) / (2.0 * C)
            return ear
        
        # Begin with texture analysis - this should happen first with a static face
        print("Stage 1: Looking for a real face (not a printout)...")
        current_stage = "texture"
        texture_frames = 0
        texture_passed_count = 0
        
        while True:
            # Check for overall timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("Liveness detection timeout")
                break
                
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
                
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process at a smaller size for speed
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Display frame with overlays
            display_frame = frame.copy()
            
            # Calculate time remaining
            time_left = max(0, timeout - elapsed_time)
            
            # Add keyboard shortcuts for toggling tests (for troubleshooting)
            cv2.putText(display_frame, "Press T to toggle texture test", 
                      (10, display_frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press B to toggle blink test", 
                      (10, display_frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press M to toggle movement test", 
                      (10, display_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(small_frame)
            
            if face_locations:
                # Scale back face location to full size
                for (top, right, bottom, left) in face_locations:
                    # Scale back up
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    
                    # Draw face rectangle
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Extract face region for texture analysis
                    if current_stage == "texture" and texture_frames % 10 == 0:  # Check every 10 frames
                        face_region = frame[top:bottom, left:right]
                        if face_region.size > 0:  # Ensure the region is valid
                            has_texture = self.detect_texture(face_region)
                            if has_texture:
                                texture_passed_count += 1
                            # Require 3 passes to confirm it's a real face
                            if texture_passed_count >= 3:
                                stage_passed["texture"] = True
                                print("✓ Texture check passed! It appears to be a real face.")
                                # Move to blink detection
                                current_stage = "blink"
                                print("Stage 2: Please blink naturally...")
                    texture_frames += 1
                
                # Get facial landmarks for further analysis
                face_landmarks = face_recognition.face_landmarks(small_frame, face_locations)
                
                for landmarks in face_landmarks:
                    # Check if eye landmarks exist for blink detection
                    if current_stage in ["blink", "head_movement"] and 'left_eye' in landmarks and 'right_eye' in landmarks:
                        # Convert landmarks to numpy arrays
                        left_eye = np.array(landmarks['left_eye'])
                        right_eye = np.array(landmarks['right_eye'])
                        
                        # Scale back up to full size
                        left_eye = left_eye * 2
                        right_eye = right_eye * 2
                        
                        # Draw eyes
                        if current_stage == "blink":
                            for eye_point in left_eye:
                                cv2.circle(display_frame, tuple(eye_point), 2, (0, 255, 255), -1)
                            for eye_point in right_eye:
                                cv2.circle(display_frame, tuple(eye_point), 2, (0, 255, 255), -1)
                            
                            # Calculate eye aspect ratio
                            left_ear = eye_aspect_ratio(left_eye)
                            right_ear = eye_aspect_ratio(right_eye)
                            
                            # Average the eye aspect ratio
                            ear = (left_ear + right_ear) / 2.0
                            ear_history.append(ear)
                            
                            # Display EAR value
                            cv2.putText(display_frame, f"EAR: {ear:.2f}", 
                                      (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
                            # Check for blink - more stringent pattern matching
                            if len(ear_history) >= 8:
                                # Modified pattern detection for low framerates
                                # We now look for ANY significant drop in EAR followed by a rise
                                
                                # Get min and max values in history
                                min_ear = min(ear_history)
                                max_ear = max(ear_history)
                                
                                # Check for significant difference between min and max
                                ear_diff = max_ear - min_ear
                                
                                # Check for any of these blink patterns:
                                # 1. Original strict pattern (slightly relaxed)
                                strict_pattern = (ear_history[0] > EYE_AR_THRESH * 0.9 and
                                                 ear_history[1] > EYE_AR_THRESH * 0.9 and
                                                 any(e < EYE_AR_THRESH * 0.8 for e in ear_history[2:5]) and
                                                 ear_history[6] > EYE_AR_THRESH * 0.9 and
                                                 ear_history[7] > EYE_AR_THRESH * 0.9)
                                
                                # 2. Simple drop and rise pattern
                                drop_rise_pattern = (ear_diff > 0.1 and  # Significant EAR change
                                                    min_ear < EYE_AR_THRESH and  # Eyes were closed at some point
                                                    ear_history[0] > min_ear * 1.3 and  # Started relatively open
                                                    ear_history[-1] > min_ear * 1.3)  # Ended relatively open
                                
                                # 3. Local minimum pattern (useful for slower framerates)
                                # Find if there's a minimum flanked by higher values
                                local_min_pattern = False
                                for i in range(1, len(ear_history)-1):
                                    if (ear_history[i] < EYE_AR_THRESH and
                                        ear_history[i] < ear_history[i-1] * 0.8 and
                                        ear_history[i] < ear_history[i+1] * 0.8):
                                        local_min_pattern = True
                                        break
                                
                                print(f"Blink patterns: strict={strict_pattern}, drop_rise={drop_rise_pattern}, local_min={local_min_pattern}")
                                print(f"EAR history: {[round(e,2) for e in ear_history]}, min={min_ear:.2f}, max={max_ear:.2f}, diff={ear_diff:.2f}")
                                
                                # If any pattern is detected, consider it a blink
                                if strict_pattern or drop_rise_pattern or local_min_pattern:
                                    stage_passed["blink"] = True
                                    print("✓ Blink detected! Moving to head movement test.")
                                    current_stage = "head_movement"
                                    # Save the head position challenge type
                                    print(f"Stage 3: Please move your head {movement_type.upper()}")
                                    movement_start_time = time.time()
                    
                    # Track nose position for head movement challenge
                    if current_stage == "head_movement" and 'nose_tip' in landmarks:
                        # Get nose point - handle it differently to avoid indexing errors
                        nose_tip = np.array(landmarks['nose_tip'])
                        # Use mean position of all nose points
                        nose_point = np.mean(nose_tip, axis=0) * 2  # Scale back up
                        
                        # Draw nose point
                        cv2.circle(display_frame, tuple(nose_point.astype(int)), 5, (0, 0, 255), -1)
                        
                        # Track nose positions
                        movement_positions.append(nose_point)
                        
                        # Only analyze after collecting enough positions
                        if len(movement_positions) > 10:
                            # Calculate movement direction
                            start_pos = np.mean(movement_positions[:5], axis=0)
                            current_pos = np.mean(movement_positions[-5:], axis=0)
                            
                            # Ensure positions are integers for drawing
                            start_point = tuple(start_pos.astype(int))
                            current_point = tuple(current_pos.astype(int))
                            
                            # Draw movement vector (ensure coordinates are valid)
                            if all(p >= 0 for p in start_point) and all(p >= 0 for p in current_point):
                                cv2.arrowedLine(display_frame, 
                                              start_point, 
                                              current_point,
                                              (255, 0, 0), 2)
                            
                            # Calculate differences
                            x_diff = current_pos[0] - start_pos[0]
                            y_diff = current_pos[1] - start_pos[1]
                            
                            # Minimum pixel movement required - reduced for low frame rates
                            min_movement = 20  # Reduced from 30
                            
                            # Display movement values for debugging
                            cv2.putText(display_frame, f"Movement: x={x_diff:.1f}, y={y_diff:.1f}", 
                                      (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                            
                            # Check if movement matches the required direction
                            if movement_type == "left" and x_diff < -min_movement:
                                movement_completed = True
                            elif movement_type == "right" and x_diff > min_movement:
                                movement_completed = True
                            elif movement_type == "up" and y_diff < -min_movement:
                                movement_completed = True
                            elif movement_type == "down" and y_diff > min_movement:
                                movement_completed = True
                                
                            if movement_completed:
                                stage_passed["head_movement"] = True
                                face_image = frame
                                print(f"✓ Head movement {movement_type} detected! Liveness confirmed.")
            
            # Draw stage information
            stages_info = [
                f"1. Texture Analysis: {'✓' if stage_passed['texture'] else '...'}",
                f"2. Blink Detection: {'✓' if stage_passed['blink'] else '...' if stage_passed['texture'] else 'Waiting'}",
                f"3. Head Movement ({movement_type}): {'✓' if stage_passed['head_movement'] else '...' if stage_passed['blink'] else 'Waiting'}"
            ]
            
            for i, info in enumerate(stages_info):
                cv2.putText(display_frame, info, 
                          (10, 150 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (0, 255, 0) if "✓" in info else (255, 255, 255), 1)
            
            # Display time remaining
            cv2.putText(display_frame, f"Time left: {int(time_left)}s", 
                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display current instruction
            instruction = ""
            if current_stage == "texture":
                instruction = "Hold still while we analyze your face"
            elif current_stage == "blink":
                instruction = "Please blink naturally"
            elif current_stage == "head_movement":
                instruction = f"Please move your head {movement_type.upper()}"
                
            cv2.putText(display_frame, instruction, 
                      (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Check if all stages passed
            all_passed = all(stage_passed.values())
            if all_passed:
                cv2.putText(display_frame, "LIVENESS CONFIRMED!", 
                          (display_frame.shape[1]//2 - 150, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # Show the frame
            cv2.imshow("Liveness Detection", display_frame)
            
            # Check for keyboard input to toggle tests
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):  # Toggle texture test
                skip_texture = not skip_texture
                if skip_texture:
                    stage_passed["texture"] = True
                    print("Texture test skipped")
                else:
                    stage_passed["texture"] = False
                    print("Texture test enabled")
            elif key == ord('b'):  # Toggle blink test
                skip_blink = not skip_blink
                if skip_blink:
                    stage_passed["blink"] = True
                    print("Blink test skipped")
                else:
                    stage_passed["blink"] = False
                    print("Blink test enabled")
            elif key == ord('m'):  # Toggle movement test
                skip_movement = not skip_movement
                if skip_movement:
                    stage_passed["head_movement"] = True
                    print("Movement test skipped")
                else:
                    stage_passed["head_movement"] = False
                    print("Movement test enabled")
            elif key == 27:  # ESC key
                break
            elif all_passed:
                # Wait a moment to show the success message
                time.sleep(2)
                break
                
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Return the results
        return all_passed, face_image
    
    def detect_liveness(self, timeout=10):
        """
        Legacy liveness detection method - now redirects to the more secure version
        """
        print("Using enhanced liveness detection for better security...")
        return self.detect_liveness_multi(timeout)
        
    def capture_face_with_liveness(self):
        """Capture a face image with liveness verification"""
        is_live, face_image = self.detect_liveness_multi()
        
        if not is_live:
            print("Liveness check failed. Please try again.")
            return None
            
        print("Liveness check passed!")
        
        # If we got a face during liveness detection, use it
        if face_image is not None:
            return face_image
            
        # Otherwise capture a new face
        return self.capture_face() 