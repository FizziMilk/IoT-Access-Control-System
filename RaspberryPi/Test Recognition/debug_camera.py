#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse

def debug_camera(camera_id=0, resolution=(640, 480)):
    """
    Debug camera by showing different color space views
    
    Args:
        camera_id: Camera device ID
        resolution: Resolution as (width, height)
    """
    print(f"Starting camera debug with camera {camera_id} at resolution {resolution}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera with ID {camera_id}")
        return
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Wait for camera to initialize
    time.sleep(2.0)
    
    # Create windows
    cv2.namedWindow("Original (BGR)", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)
    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    
    # Resize windows
    cv2.resizeWindow("Original (BGR)", resolution[0], resolution[1])
    cv2.resizeWindow("Grayscale", resolution[0], resolution[1])
    cv2.resizeWindow("RGB", resolution[0], resolution[1])
    
    # Set windows to be topmost
    cv2.setWindowProperty("Original (BGR)", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("Grayscale", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("RGB", cv2.WND_PROP_TOPMOST, 1)
    
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Failed to capture frame, retrying...")
            time.sleep(0.1)
            continue
        
        # Convert to different color spaces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create 3-channel grayscale for display
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Add labels
        cv2.putText(frame, "BGR (Original OpenCV format)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(gray_3ch, "Grayscale", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(rgb, "RGB (For face_recognition lib)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Display frames
        cv2.imshow("Original (BGR)", frame)
        cv2.imshow("Grayscale", gray_3ch)
        cv2.imshow("RGB", rgb)
        
        # Check key press
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC or q
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera debug complete")

def parse_args():
    parser = argparse.ArgumentParser(description='Camera Debug Utility')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Camera resolution (widthxheight)')
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
    
    # Run the debug utility
    debug_camera(args.camera, resolution) 