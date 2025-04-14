#!/usr/bin/env python3
import cv2
import numpy as np
import argparse

from camera import Camera
from blink_detector import BlinkDetector
from texture_analyzer import TextureAnalyzer
from liveness_detector import LivenessDetector

def test_liveness(camera_id=0, resolution="640x480"):
    """
    Direct test of the liveness detector component
    
    Args:
        camera_id: Camera device ID
        resolution: Resolution in format "widthxheight"
    """
    print(f"Testing liveness detector with camera {camera_id} at resolution {resolution}")
    
    # Parse resolution
    width, height = map(int, resolution.split('x'))
    res = (width, height)
    
    # Initialize base components
    print("Initializing camera...")
    camera = Camera(camera_id=camera_id, resolution=res)
    
    print("Initializing blink detector...")
    blink_detector = BlinkDetector(camera)
    
    print("Initializing texture analyzer...")
    texture_analyzer = TextureAnalyzer()
    
    print("Initializing liveness detector...")
    liveness_detector = LivenessDetector(camera, blink_detector, texture_analyzer)
    
    # Force OpenCV to destroy any existing windows
    cv2.destroyAllWindows()
    
    # Run liveness detection
    print("Starting liveness test...")
    is_live, face_image = liveness_detector.detect_liveness_multi()
    
    # Show result
    if is_live:
        print("Liveness detection PASSED!")
        if face_image is not None:
            cv2.imshow("Captured Face", face_image)
            cv2.waitKey(0)
    else:
        print("Liveness detection FAILED!")
    
    # Clean up
    cv2.destroyAllWindows()
    
    return is_live

def parse_args():
    parser = argparse.ArgumentParser(description='Test Liveness Detection')
    parser.add_argument('--camera', type=int, default=0,
                       help='Camera device ID (default: 0)')
    parser.add_argument('--resolution', type=str, default='640x480',
                       help='Camera resolution (widthxheight)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Run the liveness test
    test_liveness(args.camera, args.resolution) 