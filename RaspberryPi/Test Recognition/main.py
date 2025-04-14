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

def parse_args():
    parser = argparse.ArgumentParser(description='Face Recognition and Liveness Detection')
    parser.add_argument('--mode', type=str, default='liveness', 
                        choices=['liveness', 'recognition', 'enroll', 'demo'],
                        help='Operation mode')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('--resolution', type=str, default='640x480',
                        help='Camera resolution (widthxheight)')
    parser.add_argument('--name', type=str, default=None,
                        help='Name of person for enrollment')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug visualizations')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Parse resolution
    width, height = map(int, args.resolution.split('x'))
    
    # Initialize components
    print("Initializing system components...")
    camera = Camera(camera_id=args.camera, resolution=(width, height))
    
    # Initialize detection and analysis modules
    blink_detector = BlinkDetector(camera)
    texture_analyzer = TextureAnalyzer()
    
    # Initialize combined liveness detection
    liveness_detector = LivenessDetector(camera, blink_detector, texture_analyzer)
    
    # Initialize face recognition and database
    face_recognition = FaceRecognition()
    face_database = FaceDatabase('faces.db')
    
    # Run selected mode
    if args.mode == 'liveness':
        run_liveness_test(liveness_detector)
    elif args.mode == 'recognition':
        run_recognition(camera, liveness_detector, face_recognition, face_database, debug=args.debug)
    elif args.mode == 'enroll':
        if not args.name:
            print("Error: --name is required for enrollment mode")
            return
        run_enrollment(camera, liveness_detector, face_recognition, face_database, args.name, debug=args.debug)
    elif args.mode == 'demo':
        run_demo(camera, liveness_detector, face_recognition, face_database, debug=args.debug)
    
    print("Program finished")

def run_liveness_test(liveness_detector):
    """Run a standalone liveness detection test"""
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
            # Create a new window for the face
            cv2.namedWindow("Captured Face", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Captured Face", 640, 480)
            cv2.imshow("Captured Face", face_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    else:
        print("Liveness test failed.")

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
                cv2.imshow("Recognized Face", face_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Face not recognized. Access denied.")
    else:
        print("No valid face image captured.")

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
        face_encoding = face_recognition.get_face_encoding(face_image)
        
        if face_encoding is not None:
            # Save to database
            face_database.add_face(name, face_encoding, face_image)
            print(f"Successfully enrolled {name} in the database!")
            
            if debug:
                cv2.imshow("Enrolled Face", face_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        else:
            print("Failed to extract face encoding. Please try again.")
    else:
        print("No valid face image captured.")

def run_demo(camera, liveness_detector, face_recognition, face_database, debug=False):
    """Run a full demo of the system"""
    print("Starting full system demo...")
    
    while True:
        print("\nDemo Options:")
        print("1. Test Liveness Detection")
        print("2. Enroll New Face")
        print("3. Recognize Face")
        print("4. List Enrolled Faces")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            run_liveness_test(liveness_detector)
        elif choice == '2':
            name = input("Enter name for enrollment: ")
            run_enrollment(camera, liveness_detector, face_recognition, face_database, name, debug)
        elif choice == '3':
            run_recognition(camera, liveness_detector, face_recognition, face_database, debug)
        elif choice == '4':
            faces = face_database.get_all_faces()
            if faces:
                print("\nEnrolled Faces:")
                for i, (name, _, _) in enumerate(faces, 1):
                    print(f"{i}. {name}")
            else:
                print("No faces enrolled yet.")
        elif choice == '5':
            print("Exiting demo...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 