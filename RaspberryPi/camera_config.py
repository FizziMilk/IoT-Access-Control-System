"""
Camera configuration file - edit this to specify which camera device to use
"""

# The index of the camera to use
# Default is 0, but you might need to use 10, 13, 14, etc. based on your system
CAMERA_INDEX = 10

# List of indices to try in order
CAMERA_INDICES = [10, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 31]

# Map standard indices to your system's indices
INDEX_MAP = {
    0: 10,  # First camera
    1: 13,  # Second camera
    2: 14   # Third camera
} 