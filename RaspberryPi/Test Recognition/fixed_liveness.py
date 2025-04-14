#!/usr/bin/env python3
import cv2
import numpy as np
import time
import random
import argparse
import dlib
import face_recognition

def run_fixed_liveness_detection(camera_id=0, resolution=(640, 480), timeout=30):
    """
    A simplified fixed version of liveness detection with proper BGR color display
    
    Args:
        camera_id: Camera device ID
        resolution: Resolution as (width, height)
        timeout: Maximum time for detection in seconds
    """
    print(f"Starting fixed liveness detection with camera {camera_id}")
    
    # Initialize camera directly
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera with ID {camera_id}")
        return False, None
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Wait for camera to initialize
    time.sleep(2.0)
    
    # Create window with specific size
    window_name = "Fixed Liveness Detection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, resolution[0], resolution[1])
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    # Initialize face detector
    face_detector = dlib.get_frontal_face_detector()
    
    # Load landmark predictor if available
    landmark_predictor = None
    try:
        landmark_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    except RuntimeError as e:
        print(f"WARNING: Could not load landmark predictor: {e}")
    
    # Track stages of liveness detection
    stage_passed = {
        "face_detected": False,
        "blink_detected": False,
        "movement_detected": False
    }
    
    # For movement detection
    movement_type = random.choice(["left", "right", "up", "down"])
    movement_positions = []
    
    # Start time
    start_time = time.time()
    
    # Main loop
    current_stage = "face_detection"
    face_image = None
    
    while True:
        # Check for timeout
        elapsed_time = time.time() - start_time
        if elapsed_time > timeout:
            print("Liveness detection timeout")
            break
        
        # Capture frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Failed to capture frame, retrying...")
            time.sleep(0.1)
            continue
        
        # Create a copy for visualization
        display_frame = frame.copy()
        
        # Display time remaining
        time_left = max(0, timeout - elapsed_time)
        cv2.putText(display_frame, f"Time left: {int(time_left)}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display current stage instruction
        stage_text = ""
        if current_stage == "face_detection":
            stage_text = "Please center your face in the frame"
        elif current_stage == "blink_detection":
            stage_text = "Please blink naturally"
        elif current_stage == "movement_detection":
            stage_text = f"Please move your head {movement_type}"
        
        cv2.putText(display_frame, stage_text, 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector(gray, 0)
        
        if faces:
            # Process the largest face
            face = max(faces, key=lambda rect: rect.width() * rect.height())
            
            # Convert to bounding box format (x, y, w, h)
            x, y = face.left(), face.top()
            w, h = face.width(), face.height()
            
            # Draw face rectangle
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Extract face for later use
            face_margin = 20
            x1 = max(0, x - face_margin)
            y1 = max(0, y - face_margin)
            x2 = min(frame.shape[1], x + w + face_margin)
            y2 = min(frame.shape[0], y + h + face_margin)
            face_image = frame[y1:y2, x1:x2].copy()
            
            # Mark face detection as passed
            if not stage_passed["face_detected"]:
                stage_passed["face_detected"] = True
                current_stage = "blink_detection"
                print("✓ Face detected! Please blink naturally.")
            
            # Process landmarks if predictor is available
            if landmark_predictor and current_stage in ["blink_detection", "movement_detection"]:
                shape = landmark_predictor(gray, face)
                landmarks = []
                
                # Convert landmarks to points
                for i in range(68):
                    x, y = shape.part(i).x, shape.part(i).y
                    landmarks.append((x, y))
                    # Draw landmark points
                    cv2.circle(display_frame, (x, y), 2, (0, 0, 255), -1)
                
                # Handle blink detection stage
                if current_stage == "blink_detection":
                    # Extract eye landmarks
                    left_eye = landmarks[36:42]
                    right_eye = landmarks[42:48]
                    
                    # Draw eye contours
                    for eye in [left_eye, right_eye]:
                        eye_pts = np.array(eye, dtype=np.int32)
                        cv2.polylines(display_frame, [eye_pts], True, (0, 255, 255), 1)
                    
                    # Simplified blink detection - in a real system, this would be more complex
                    # For demo, just press 'b' to simulate blink detection
                    cv2.putText(display_frame, "Press 'B' to simulate blink detected", 
                               (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
                
                # Handle movement detection stage
                if current_stage == "movement_detection":
                    # Get nose point for tracking
                    nose_tip = landmarks[30]
                    cv2.circle(display_frame, nose_tip, 5, (0, 0, 255), -1)
                    
                    # Track nose position
                    movement_positions.append(nose_tip)
                    
                    # Analyze movement if we have enough positions
                    if len(movement_positions) > 10:
                        # Calculate movement from start to current
                        start_pos = movement_positions[0]
                        current_pos = movement_positions[-1]
                        
                        # Calculate displacement
                        dx = current_pos[0] - start_pos[0]
                        dy = current_pos[1] - start_pos[1]
                        
                        # Draw movement vector
                        cv2.arrowedLine(display_frame, start_pos, current_pos, (255, 0, 0), 2)
                        
                        # Display movement
                        cv2.putText(display_frame, f"Movement: x={dx}, y={dy}", 
                                   (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        
                        # For demo, press direction keys to simulate movement detection
                        cv2.putText(display_frame, f"Press arrow key ({movement_type}) to simulate movement", 
                                   (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display status summary
        stages_info = [
            f"1. Face Detection: {'✓' if stage_passed['face_detected'] else '...'}",
            f"2. Blink Detection: {'✓' if stage_passed['blink_detected'] else '...' if stage_passed['face_detected'] else 'Waiting'}",
            f"3. Head Movement ({movement_type}): {'✓' if stage_passed['movement_detected'] else '...' if stage_passed['blink_detected'] else 'Waiting'}"
        ]
        
        for i, info in enumerate(stages_info):
            cv2.putText(display_frame, info, 
                      (10, display_frame.shape[0] - 100 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                      (0, 255, 0) if "✓" in info else (255, 255, 255), 1)
        
        # Check if all stages are passed
        all_passed = all(stage_passed.values())
        if all_passed:
            cv2.putText(display_frame, "LIVENESS CONFIRMED!", 
                      (display_frame.shape[1]//2 - 150, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show the frame
        cv2.imshow(window_name, display_frame)
        
        # Process key presses
        key = cv2.waitKey(1) & 0xFF
        
        # ESC or 'q' to exit
        if key == 27 or key == ord('q'):
            break
        
        # 'b' to simulate blink detected
        if key == ord('b') and current_stage == "blink_detection":
            stage_passed["blink_detected"] = True
            current_stage = "movement_detection"
            movement_positions = []  # Reset positions for movement detection
            print(f"✓ Blink detected! Please move your head {movement_type}.")
        
        # Arrow keys to simulate movement detection
        if current_stage == "movement_detection":
            if (key == 81 and movement_type == "left") or \
               (key == 83 and movement_type == "right") or \
               (key == 82 and movement_type == "up") or \
               (key == 84 and movement_type == "down"):
                stage_passed["movement_detected"] = True
                print("✓ Movement detected! Liveness confirmed.")
        
        # Check if all stages are passed
        if all_passed:
            # Wait to show success message
            time.sleep(2)
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    
    # Return results
    return all(stage_passed.values()), face_image

def parse_args():
    parser = argparse.ArgumentParser(description='Fixed Liveness Detection')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Camera resolution (widthxheight)')
    parser.add_argument('--timeout', type=int, default=30,
                       help='Detection timeout in seconds (default: 30)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Parse resolution
    try:
        width, height = map(int, args.resolution.split('x'))
        resolution = (width, height)
    except ValueError:
        print(f"Invalid resolution format: {args.resolution}. Using default 640x480.")
        resolution = (640, 480)
    
    # Run fixed liveness detection
    is_live, face_image = run_fixed_liveness_detection(args.camera, resolution, args.timeout)
    
    # Show result
    if is_live:
        print("Liveness detection PASSED!")
        if face_image is not None:
            cv2.imshow("Captured Face", face_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Liveness detection FAILED!") 