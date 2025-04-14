#!/usr/bin/env python3
import cv2
import numpy as np
import argparse
import time

from camera import Camera
from recognition import FaceRecognition
from storage import FaceDatabase

def test_recognition(camera_id=0, resolution="640x480", mode="test", name=None):
    """
    Test face recognition functionality
    
    Args:
        camera_id: Camera device ID
        resolution: Resolution string "widthxheight"
        mode: "test" for recognition test, "enroll" to add a new face
        name: Name of the person for enrollment mode
    """
    # Parse resolution
    width, height = map(int, resolution.split('x'))
    res = (width, height)
    
    # Initialize components
    print("Initializing camera...")
    camera = Camera(camera_id=camera_id, resolution=res)
    
    print("Initializing face recognition...")
    face_recognition = FaceRecognition()
    
    print("Initializing face database...")
    face_database = FaceDatabase('faces.db')
    
    if mode == "enroll" and name:
        # Enrollment mode
        print(f"Starting enrollment for {name}...")
        
        # Create window for face capture
        cv2.namedWindow("Face Enrollment", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Enrollment", res[0], res[1])
        cv2.setWindowProperty("Face Enrollment", cv2.WND_PROP_TOPMOST, 1)
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        
        time.sleep(1.0)  # Wait for camera
        
        print("Position your face and press SPACE to capture")
        
        face_captured = False
        face_image = None
        
        while not face_captured:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            # Display the frame
            cv2.putText(frame, "Position face and press SPACE to capture", 
                      (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("Face Enrollment", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # SPACE
                face_image = frame.copy()
                face_captured = True
        
        cap.release()
        cv2.destroyAllWindows()
        
        if face_image is not None:
            # Extract face encoding
            face_encoding = face_recognition.get_face_encoding(face_image)
            
            if face_encoding is not None:
                # Save to database
                face_database.add_face(name, face_encoding, face_image)
                print(f"Successfully enrolled {name} in the database!")
                
                # Show the enrolled face
                cv2.imshow("Enrolled Face", face_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("Failed to extract face encoding. Please try again.")
        else:
            print("No face captured.")
    
    elif mode == "test":
        # Recognition test mode
        print("Starting face recognition test...")
        
        # Get all faces from database
        faces = face_database.get_all_faces()
        if not faces:
            print("No faces in database. Please enroll users first.")
            return
        
        print(f"Found {len(faces)} faces in database:")
        for i, (name, _, _) in enumerate(faces, 1):
            print(f"{i}. {name}")
        
        # Create window for face capture
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Face Recognition", res[0], res[1])
        cv2.setWindowProperty("Face Recognition", cv2.WND_PROP_TOPMOST, 1)
        
        # Initialize camera
        cap = cv2.VideoCapture(camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
        
        time.sleep(1.0)  # Wait for camera
        
        print("Position your face and press SPACE for recognition")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                time.sleep(0.1)
                continue
                
            # Display instructions
            cv2.putText(frame, "SPACE: Recognize, Q: Quit", 
                      (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                      
            cv2.imshow("Face Recognition", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == 32:  # SPACE
                # Capture face for recognition
                face_image = frame.copy()
                
                # Recognize face
                name, confidence = face_recognition.recognize_face(face_image, faces)
                
                if name:
                    result_text = f"Welcome, {name}! (Confidence: {confidence:.2f})"
                    result_color = (0, 255, 0)
                else:
                    result_text = "Face not recognized"
                    result_color = (0, 0, 255)
                
                # Display result
                result_frame = face_image.copy()
                cv2.putText(result_frame, result_text, 
                          (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, result_color, 2)
                cv2.imshow("Recognition Result", result_frame)
                cv2.waitKey(2000)  # Show result for 2 seconds
        
        cap.release()
        cv2.destroyAllWindows()
    
    else:
        print("Invalid mode. Use 'test' for recognition or 'enroll' with a name.")

def parse_args():
    parser = argparse.ArgumentParser(description='Test Face Recognition')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Camera resolution (widthxheight)')
    parser.add_argument('--mode', type=str, default='test',
                       choices=['test', 'enroll'],
                       help='Test mode: "test" for recognition, "enroll" to add a face')
    parser.add_argument('--name', type=str, default=None,
                       help='Name of person to enroll (required for enroll mode)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Validate enroll mode has name
    if args.mode == 'enroll' and not args.name:
        print("Error: --name is required for enrollment mode")
        exit(1)
    
    # Run the test
    test_recognition(
        camera_id=args.camera, 
        resolution=args.resolution,
        mode=args.mode,
        name=args.name
    ) 