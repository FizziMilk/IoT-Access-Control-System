#!/usr/bin/env python3
import cv2
import numpy as np
import time
import argparse

def fix_white_balance(frame):
    """Apply simple white balance correction to the frame"""
    # Split the frame into BGR channels
    b, g, r = cv2.split(frame)
    
    # Calculate channel means
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    
    # Calculate scaling factors (normalize to green channel)
    if g_mean > 0:
        b_scale = g_mean / b_mean if b_mean > 0 else 1.0
        r_scale = g_mean / r_mean if r_mean > 0 else 1.0
    else:
        b_scale = 0.8  # Default scaling if calculation fails
        r_scale = 1.2
    
    # Apply scaling - reduce blue, increase red slightly
    b = cv2.addWeighted(b, b_scale * 0.8, np.zeros_like(b), 0, 0)
    r = cv2.addWeighted(r, r_scale * 1.1, np.zeros_like(r), 0, 0)
    
    # Merge channels back
    balanced = cv2.merge([b, g, r])
    return balanced

def adjust_temperature(frame, temp_adjustment=0):
    """Adjust color temperature (negative = cooler/blue, positive = warmer/red)"""
    # Skip if no adjustment
    if temp_adjustment == 0:
        return frame
    
    # Split channels
    b, g, r = cv2.split(frame)
    
    # Adjust blue and red channels
    if temp_adjustment < 0:
        # Cooler - reduce blue
        b = cv2.addWeighted(b, 1.0 + temp_adjustment*0.1, np.zeros_like(b), 0, 0)
    else:
        # Warmer - increase red, reduce blue
        r = cv2.addWeighted(r, 1.0 + temp_adjustment*0.1, np.zeros_like(r), 0, 0)
        b = cv2.addWeighted(b, 1.0 - temp_adjustment*0.05, np.zeros_like(b), 0, 0)
    
    # Merge back
    adjusted = cv2.merge([b, g, r])
    return adjusted

def test_camera_with_corrections(camera_id=0, resolution=(640, 480)):
    """
    Camera test with color correction options
    
    Args:
        camera_id: Camera device ID
        resolution: Resolution as (width, height)
    """
    print(f"Starting camera test with color correction. Camera: {camera_id}")
    
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
    cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Fixed White Balance", cv2.WINDOW_NORMAL)
    
    # Set window properties
    cv2.resizeWindow("Original", resolution[0], resolution[1])
    cv2.resizeWindow("Fixed White Balance", resolution[0], resolution[1])
    
    # Set top-most
    cv2.setWindowProperty("Original", cv2.WND_PROP_TOPMOST, 1)
    cv2.setWindowProperty("Fixed White Balance", cv2.WND_PROP_TOPMOST, 1)
    
    # Initialize temperature adjustment
    temp_adjust = 0
    wb_enabled = True
    
    # Main loop
    while True:
        # Capture frame
        ret, frame = cap.read()
        
        if not ret or frame is None:
            print("Failed to capture frame, retrying...")
            time.sleep(0.1)
            continue
        
        # Create a copy of the original
        original = frame.copy()
        
        # Apply white balance if enabled
        if wb_enabled:
            corrected = fix_white_balance(frame)
        else:
            corrected = frame.copy()
        
        # Apply temperature adjustment
        corrected = adjust_temperature(corrected, temp_adjust)
        
        # Add info text
        cv2.putText(original, "Original (BGR)", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        wb_status = "ON" if wb_enabled else "OFF"
        cv2.putText(corrected, f"Fixed (WB: {wb_status}, Temp: {temp_adjust})", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Add controls help
        controls = [
            "W: Toggle white balance",
            "Up/Down: Adjust temperature",
            "R: Reset corrections",
            "Q/ESC: Quit"
        ]
        
        for i, text in enumerate(controls):
            cv2.putText(corrected, text, (10, 70 + i*30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Display frames
        cv2.imshow("Original", original)
        cv2.imshow("Fixed White Balance", corrected)
        
        # Process keypress
        key = cv2.waitKey(1) & 0xFF
        
        # ESC or 'q' to exit
        if key == 27 or key == ord('q'):
            break
            
        # 'w' to toggle white balance
        elif key == ord('w'):
            wb_enabled = not wb_enabled
            print(f"White balance: {'ON' if wb_enabled else 'OFF'}")
            
        # Up/Down arrows to adjust temperature
        elif key == 82:  # Up arrow
            temp_adjust = min(temp_adjust + 1, 10)
            print(f"Temperature adjustment: {temp_adjust}")
        elif key == 84:  # Down arrow
            temp_adjust = max(temp_adjust - 1, -10)
            print(f"Temperature adjustment: {temp_adjust}")
            
        # 'r' to reset
        elif key == ord('r'):
            wb_enabled = True
            temp_adjust = 0
            print("Reset color corrections")
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Camera test complete")

def parse_args():
    parser = argparse.ArgumentParser(description='Camera Test with Color Correction')
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
    
    # Run the test
    test_camera_with_corrections(args.camera, resolution) 