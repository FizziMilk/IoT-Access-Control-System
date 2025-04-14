#!/usr/bin/env python3
import cv2
import time
import argparse
import sys

def test_camera_display(camera_id=0, resolution=(640, 480)):
    """
    Test camera display with explicit window creation and handling.
    
    Args:
        camera_id: Camera device ID (default: 0)
        resolution: Desired resolution as (width, height)
    """
    print(f"Opening camera {camera_id} at resolution {resolution}")
    
    # Initialize camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"ERROR: Could not open camera with ID {camera_id}")
        return False
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    # Wait for camera to initialize
    time.sleep(2.0)
    
    # Create window with specific size
    window_name = "Camera Test"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, resolution[0], resolution[1])
    
    # Force window to be visible and on top
    cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
    
    frame_count = 0
    start_time = time.time()
    
    try:
        while True:
            # Capture frame
            ret, frame = cap.read()
            
            if not ret or frame is None or frame.size == 0:
                print("Error: Failed to capture frame")
                time.sleep(0.1)
                continue
            
            frame_count += 1
            elapsed = time.time() - start_time
            
            # Add FPS counter
            fps = frame_count / elapsed if elapsed > 0 else 0
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Show resolution
            h, w = frame.shape[:2]
            cv2.putText(frame, f"Resolution: {w}x{h}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display the frame
            cv2.imshow(window_name, frame)
            
            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("Camera test interrupted")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
    print(f"Camera test completed. Processed {frame_count} frames in {elapsed:.1f} seconds ({fps:.1f} FPS)")
    return True

def parse_args():
    parser = argparse.ArgumentParser(description='Camera Display Test')
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
    
    print("Camera Display Test")
    print("-------------------")
    print(f"Camera ID: {args.camera}")
    print(f"Resolution: {resolution[0]}x{resolution[1]}")
    print("Press 'q' to exit")
    
    # Run the camera test
    if not test_camera_display(args.camera, resolution):
        sys.exit(1) 