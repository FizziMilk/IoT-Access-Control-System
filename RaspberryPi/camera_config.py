"""
Camera configuration for face recognition system.
This file contains settings related to the camera setup.
"""

# The camera index to use (0 is usually the built-in webcam)
# This value might need adjustment based on your specific hardware setup
CAMERA_INDEX = 0

# Define minimum face size for reliable recognition
MIN_FACE_WIDTH = 100  # Minimum width in pixels
MIN_FACE_HEIGHT = 100  # Minimum height in pixels

# Camera resolution for face recognition
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Number of frames to show during initial user guidance (5 seconds at ~15fps)
DISPLAY_FRAMES = 75

# Minimum number of blinks required for liveness detection
MIN_BLINKS_REQUIRED = 2 