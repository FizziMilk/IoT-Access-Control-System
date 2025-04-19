"""
Camera configuration file for the face recognition system.
"""

# Default camera index (standard webcam is usually 0)
CAMERA_INDEX = 0

# Camera resolution for face recognition
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# Number of frames to show during initial user guidance (5 seconds at ~15fps)
DISPLAY_FRAMES = 75

# Minimum number of blinks required for liveness detection
MIN_BLINKS_REQUIRED = 2 