#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse

from camera import Camera
from blink_detector import BlinkDetector
from texture_analyzer import TextureAnalyzer
from liveness_detector import LivenessDetector
from recognition import FaceRecognition
from storage import FaceDatabase

def fix_white_balance(frame):
    """Apply white balance correction to reduce blue tint"""
    if frame is None:
        return None
        
    # Split into channels
    b, g, r = cv2.split(frame)
    
    # Apply correction
    b = cv2.addWeighted(b, 0.8, np.zeros_like(b), 0, 0)  # Reduce blue
    r = cv2.addWeighted(r, 1.1, np.zeros_like(r), 0, 0)  # Boost red
    
    # Merge back
    return cv2.merge([b, g, r])

def run_fixed_demo(camera_id=0, resolution=(640, 480), debug=False):
    """
    Run a simplified version of the full demo with additional diagnostics
    
    Args:
        camera_id: Camera device ID
        resolution: Resolution as (width, height)
        debug: Enable debug output
    """
    print("Starting fixed demo...")
    
    # First clean up any existing windows
    cv2.destroyAllWindows()
    
    # Initialize components with diagnostic output
    print("Initializing camera...")
    camera = Camera(camera_id=camera_id, resolution=resolution)
    
    # Test camera directly first
    print("Testing basic camera functionality...")
    test_window = "Camera Test"
    cv2.namedWindow(test_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(test_window, resolution[0], resolution[1])
    
    # Create a direct camera capture
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera with ID {camera_id}")
        return
        
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Wait for camera initialization
    time.sleep(2.0)
    
    # Display test frames
    print("Press any key to continue to menu...")
    for i in range(30):  # Show 30 frames
        ret, frame = cap.read()
        if ret and frame is not None:
            # Apply white balance
            frame = fix_white_balance(frame)
            
            # Add info text
            cv2.putText(frame, "Camera Test - Press any key to continue", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame {i+1}/30", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                      
            cv2.imshow(test_window, frame)
            
            if cv2.waitKey(1) != -1:
                break
                
        time.sleep(0.1)
    
    # Release capture
    cap.release()
    cv2.destroyAllWindows()
    
    # Initialize other components
    print("Initializing detection modules...")
    blink_detector = BlinkDetector(camera)
    texture_analyzer = TextureAnalyzer()
    
    print("Initializing liveness detector...")
    liveness_detector = LivenessDetector(camera, blink_detector, texture_analyzer)
    
    print("Initializing face recognition and database...")
    face_recognition = FaceRecognition()
    face_database = FaceDatabase('faces.db')
    
    # Menu loop
    while True:
        print("\nFixed Demo Options:")
        print("1. Test Camera")
        print("2. Test Liveness Detection")
        print("3. Enroll New Face")
        print("4. Recognize Face")
        print("5. List Enrolled Faces")
        print("6. Exit")
        
        choice = input("Enter your choice (1-6): ")
        
        if choice == '1':
            test_camera(camera_id, resolution)
        elif choice == '2':
            run_liveness_test(liveness_detector)
        elif choice == '3':
            name = input("Enter name for enrollment: ")
            run_enrollment(camera, liveness_detector, face_recognition, face_database, name, debug)
        elif choice == '4':
            run_recognition(camera, liveness_detector, face_recognition, face_database, debug)
        elif choice == '5':
            list_faces(face_database)
        elif choice == '6':
            print("Exiting demo...")
            break
        else:
            print("Invalid choice. Please try again.")

def test_camera(camera_id, resolution):
    """Run a simple camera test"""
    print("Starting camera test...")
    
    # Create window
    window_name = "Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, resolution[0], resolution[1])
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera with ID {camera_id}")
        return
        
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Wait for initialization
    time.sleep(1.0)
    
    print("Press ESC or 'q' to exit camera test")
    start_time = time.time()
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            # Fix white balance
            frame = fix_white_balance(frame)
                
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add info text
            cv2.putText(frame, f"FPS: {fps:.1f}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "Press ESC or 'q' to exit", 
                      (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                      
            # Show frame
            cv2.imshow(window_name, frame)
            
            # Check for exit
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Camera test interrupted")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
    print(f"Camera test complete. Processed {frame_count} frames at {fps:.1f} FPS")

def run_liveness_test(liveness_detector):
    """Run liveness detection test"""
    # First destroy any existing windows
    cv2.destroyAllWindows()
    
    print("Starting liveness detection test...")
    
    # Run liveness detection with longer timeout
    is_live, face_image = liveness_detector.detect_liveness_multi(timeout=40)
    
    # Explicitly destroy windows again
    cv2.destroyAllWindows()
    
    if is_live:
        print("Liveness test passed!")
        if face_image is not None:
            # Apply white balance to face image
            face_image = fix_white_balance(face_image)
            
            # Create a new window for the face
            cv2.namedWindow("Captured Face", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Captured Face", 640, 480)
            cv2.imshow("Captured Face", face_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Liveness test failed.")

def run_enrollment(camera, liveness_detector, face_recognition, face_database, name, debug=False):
    """Enroll a new face in the database"""
    print(f"Starting enrollment for {name}...")
    
    # First perform liveness detection
    is_live, face_image = liveness_detector.detect_liveness_multi()
    
    if not is_live:
        print("Liveness check failed. Cannot enroll.")
        return
    
    print("Liveness verification successful! Proceeding to enrollment...")
    
    # Extract face features
    if face_image is not None:
        # Apply white balance
        face_image = fix_white_balance(face_image)
        
        face_encoding = face_recognition.get_face_encoding(face_image)
        
        if face_encoding is not None:
            # Save to database
            face_database.add_face(name, face_encoding, face_image)
            print(f"Successfully enrolled {name} in the database!")
            
            if debug:
                cv2.namedWindow("Enrolled Face", cv2.WINDOW_NORMAL)
                cv2.imshow("Enrolled Face", face_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Failed to extract face encoding. Please try again.")
    else:
        print("No valid face image captured.")

def run_recognition(camera, liveness_detector, face_recognition, face_database, debug=False):
    """Run face recognition with liveness detection"""
    print("Starting face recognition with liveness check...")
    
    # First perform liveness detection
    is_live, face_image = liveness_detector.detect_liveness_multi()
    
    if not is_live:
        print("Liveness check failed. Access denied.")
        return
    
    print("Liveness verification successful! Proceeding to recognition...")
    
    # Perform face recognition on the captured face
    if face_image is not None:
        # Apply white balance
        face_image = fix_white_balance(face_image)
        
        # Get all known faces from database
        known_faces = face_database.get_all_faces()
        if not known_faces:
            print("No faces in database. Please enroll users first.")
            return
        
        # Perform recognition
        name, confidence = face_recognition.recognize_face(face_image, known_faces)
        
        if name:
            print(f"Welcome, {name}! (Confidence: {confidence:.2f})")
            if debug:
                cv2.namedWindow("Recognized Face", cv2.WINDOW_NORMAL)
                cv2.imshow("Recognized Face", face_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Face not recognized. Access denied.")
    else:
        print("No valid face image captured.")

def list_faces(face_database):
    """List all enrolled faces"""
    faces = face_database.get_all_faces()
    
    if faces:
        print("\nEnrolled Faces:")
        for i, (name, _, _) in enumerate(faces, 1):
            print(f"{i}. {name}")
    else:
        print("No faces enrolled yet.")

def parse_args():
    parser = argparse.ArgumentParser(description='Fixed Face Recognition Demo')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Camera resolution (widthxheight)')
    parser.add_argument('--debug', action='store_true',
                       help='Enable debug visualizations')
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
        
    # Run the demo
    run_fixed_demo(args.camera, resolution, args.debug) 