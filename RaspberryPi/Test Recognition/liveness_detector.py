import cv2
import numpy as np
import time
import random
from collections import deque
import threading

class LivenessDetector:
    """
    Advanced liveness detection using multiple techniques:
    1. Blink detection with higher thresholds
    2. Random head movement challenges
    3. Texture analysis to detect printed photos
    """
    def __init__(self, camera, blink_detector, texture_analyzer):
        """
        Initialize liveness detector with camera and analysis components
        
        Args:
            camera: Camera object for video capture
            blink_detector: BlinkDetector for blink detection
            texture_analyzer: TextureAnalyzer for texture analysis
        """
        self.camera = camera
        self.blink_detector = blink_detector
        self.texture_analyzer = texture_analyzer
        
    def detect_liveness_multi(self, timeout=30):
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
        cap = cv2.VideoCapture(self.camera.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.camera.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.camera.resolution[1])
        
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
        
        # Initialize visualization
        window_name = "Liveness Detection"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        
        try:
            while True:
                # Check if time limit exceeded
                elapsed_time = time.time() - start_time
                if elapsed_time > timeout:
                    print("Timeout: Verification failed")
                    cv2.destroyWindow(window_name)
                    cap.release()
                    return False, None
                
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not capture frame")
                    break
                
                # Create a copy for visualization
                display_frame = frame.copy()
                h, w = frame.shape[:2]
                
                # Add elapsed time indicator
                time_left = timeout - elapsed_time
                cv2.rectangle(display_frame, (10, 10), (w-10, 50), (40, 40, 40), -1)
                cv2.putText(display_frame, f"Time left: {time_left:.1f}s", (20, 35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Progress indicator
                progress_width = int((w - 40) * (1 - time_left/timeout))
                cv2.rectangle(display_frame, (20, 60), (20 + progress_width, 70), (0, 255, 0), -1)
                cv2.rectangle(display_frame, (20, 60), (w-20, 70), (255, 255, 255), 1)
                
                # Detect face and landmarks
                face_result = self.camera.detect_face(frame)
                
                # If no face detected, display warning and continue
                if not face_result[0]:
                    cv2.putText(display_frame, "No face detected - please center your face", 
                                (w//4, h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    cv2.imshow(window_name, display_frame)
                    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
                        break
                    continue
                
                face_bbox, landmarks = face_result[1], face_result[2]
                
                # Draw face bounding box
                x, y, w_face, h_face = face_bbox
                cv2.rectangle(display_frame, (x, y), (x + w_face, y + h_face), (0, 255, 0), 2)
                
                # Extract face image for texture analysis or later use
                face_margin = 20
                x1 = max(0, x - face_margin)
                y1 = max(0, y - face_margin)
                x2 = min(w, x + w_face + face_margin)
                y2 = min(h, y + h_face + face_margin)
                face_image = frame[y1:y2, x1:x2].copy()
                
                # ===================== STAGE LOGIC =====================
                # Initialize stage if not set
                if current_stage == "init":
                    if not skip_texture:
                        current_stage = "texture"
                    elif not skip_blink:
                        current_stage = "blink"
                    elif not skip_movement:
                        current_stage = "head_movement"
                        movement_start_time = time.time()
                    else:
                        # All stages skipped
                        print("All stages skipped (debug mode)")
                        cv2.destroyWindow(window_name)
                        cap.release()
                        return True, face_image
                
                # ===================== TEXTURE ANALYSIS STAGE =====================
                if current_stage == "texture":
                    # Run texture analysis on the face region
                    texture_result, texture_debug = self.texture_analyzer.analyze_texture(
                        face_image, debug=True)
                    
                    # Show texture debug visualization if available
                    if texture_debug is not None:
                        debug_resized = cv2.resize(texture_debug, (w, h))
                        # Blend with original frame
                        display_frame = cv2.addWeighted(display_frame, 0.3, debug_resized, 0.7, 0)
                    
                    # Display stage information
                    cv2.putText(display_frame, "Stage: Texture Analysis", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    result_text = "PASS" if texture_result else "FAIL"
                    result_color = (0, 255, 0) if texture_result else (0, 0, 255)
                    cv2.putText(display_frame, f"Result: {result_text}", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                    
                    # Move to next stage if passed
                    if texture_result:
                        stage_passed["texture"] = True
                        print("Texture analysis passed")
                        
                        # Determine next stage
                        if not skip_blink:
                            current_stage = "blink"
                        elif not skip_movement:
                            current_stage = "head_movement"
                            movement_start_time = time.time()
                        else:
                            # All remaining stages skipped
                            break
                    else:
                        # Failed texture analysis
                        print("Texture analysis failed")
                        cv2.destroyWindow(window_name)
                        cap.release()
                        return False, face_image
                
                # ===================== BLINK DETECTION STAGE =====================
                elif current_stage == "blink":
                    # Use the blink detector with movement rejection
                    blink_result, vis_frame, debug_data = self.blink_detector.detect_blink_with_movement_rejection(
                        frame, face_bbox, landmarks, visualize=True)
                    
                    # Overlay blink visualization on display frame
                    if vis_frame is not None:
                        display_frame = vis_frame
                    
                    # Display stage information
                    cv2.putText(display_frame, "Stage: Blink Detection", (20, 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, "Please blink naturally", (20, 130),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Move to next stage if passed
                    if blink_result:
                        stage_passed["blink"] = True
                        print("Blink detection passed")
                        
                        # Determine next stage
                        if not skip_movement:
                            current_stage = "head_movement"
                            movement_start_time = time.time()
                        else:
                            # All remaining stages skipped
                            break
                
                # ===================== HEAD MOVEMENT STAGE =====================
                elif current_stage == "head_movement":
                    # Get nose position for tracking movement
                    nose_tip = landmarks[30]  # Index 30 is nose tip in 68-point landmarks
                    
                    # Start collecting nose positions for movement analysis
                    if movement_positions:
                        # Calculate movement from start
                        start_x, start_y = movement_positions[0]
                        current_x, current_y = nose_tip
                        
                        # Relative movement from start position
                        delta_x = current_x - start_x
                        delta_y = current_y - start_y
                        
                        # Distance moved
                        distance = np.sqrt(delta_x**2 + delta_y**2)
                        
                        # Direction of movement
                        moved_right = delta_x > 15
                        moved_left = delta_x < -15
                        moved_up = delta_y < -15
                        moved_down = delta_y > 15
                        
                        # Check if the requested movement was completed
                        if (movement_type == "right" and moved_right) or \
                           (movement_type == "left" and moved_left) or \
                           (movement_type == "up" and moved_up) or \
                           (movement_type == "down" and moved_down):
                            movement_completed = True
                    else:
                        # First frame, store initial position
                        movement_positions.append(nose_tip)
                    
                    # Keep track of nose positions
                    movement_positions.append(nose_tip)
                    
                    # Draw visualization of nose tracking
                    for i in range(1, len(movement_positions)):
                        p1 = tuple(map(int, movement_positions[i-1]))
                        p2 = tuple(map(int, movement_positions[i]))
                        cv2.line(display_frame, p1, p2, (0, 255, 255), 2)
                    
                    # Display stage information
                    cv2.putText(display_frame, f"Stage: Head Movement ({movement_type.upper()})", 
                                (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Please move your head {movement_type}", 
                                (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display completion indicator
                    progress_color = (0, 255, 0) if movement_completed else (0, 165, 255)
                    cv2.putText(display_frame, "Movement completed" if movement_completed else "Move head...", 
                                (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, progress_color, 2)
                    
                    # Move to next stage if completed
                    if movement_completed:
                        stage_passed["head_movement"] = True
                        print("Head movement challenge passed")
                        # All stages completed
                        break
                
                # Show frame
                cv2.imshow(window_name, display_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC to exit
                    print("User canceled verification")
                    break
            
            # Check if all required stages passed
            is_live = all([
                stage_passed["texture"] or skip_texture,
                stage_passed["blink"] or skip_blink,
                stage_passed["head_movement"] or skip_movement
            ])
            
            # Cleanup and return result
            cv2.destroyWindow(window_name)
            cap.release()
            
            if is_live:
                print("Liveness verification successful!")
            else:
                print("Liveness verification failed")
                
            return is_live, face_image
            
        except Exception as e:
            print(f"Error during liveness detection: {str(e)}")
            cv2.destroyWindow(window_name)
            cap.release()
            return False, None 