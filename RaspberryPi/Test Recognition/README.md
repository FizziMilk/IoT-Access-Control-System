# Advanced Face Recognition System with Liveness Detection

This system provides robust face recognition with multi-modal liveness detection to prevent spoofing attempts. It uses a modular architecture for maintainability and extensibility.

## Architecture

The system is structured into the following components:

- **Camera**: Core camera functionality (capturing frames, basic face detection)
- **BlinkDetector**: Enhanced blink detection with movement rejection
- **TextureAnalyzer**: Image texture analysis for detecting digital displays/photos
- **LivenessDetector**: Multi-modal liveness detection combining multiple techniques
- **FaceRecognition**: Face recognition and encoding 
- **FaceDatabase**: Storage and retrieval of enrolled faces

## Liveness Detection Features

The system includes advanced anti-spoofing measures:

1. **Texture Analysis**: Analyzes image textures to detect printouts or digital displays using:
   - Laplacian variance for detail preservation
   - FFT analysis of high-frequency components
   - Color variance analysis
   - Gradient complexity
   - LBP pattern uniformity
   - Moiré pattern detection
   - Reflection detection

2. **Blink Detection**: Enhanced blink detection with movement rejection to prevent:
   - Camera shaking attacks
   - Video replay attacks
   - Analyzes eye aspect ratio (EAR) while monitoring face movement

3. **Head Movement Challenges**: Random head movement requests to verify user is live

## Usage

```bash
# Run liveness detection test
python main.py --mode liveness

# Enroll a new face
python main.py --mode enroll --name "John Doe"

# Run face recognition with liveness check
python main.py --mode recognition

# Run demo mode with all features
python main.py --mode demo

# Additional options
python main.py --camera 1 --resolution 1280x720 --debug
```

## Requirements

- Python 3.6+
- OpenCV
- NumPy
- scipy
- dlib (for facial landmark detection)
- face_recognition

## Installation

```bash
pip install numpy opencv-python scipy dlib face_recognition
```

## Advanced Configuration

For production deployment, you may want to adjust:

- Thresholds in TextureAnalyzer for different environments
- EAR thresholds in BlinkDetector for different cameras
- Movement thresholds in head movement challenges

## Code Structure

- `camera.py`: Core camera functionality
- `blink_detector.py`: Blink detection with movement analysis
- `texture_analyzer.py`: Image texture analysis 
- `liveness_detector.py`: Multi-modal liveness detection
- `recognition.py`: Face recognition
- `storage.py`: Face database management
- `main.py`: Command-line interface and demo application 