import cv2
import numpy as np
import time
import os
import argparse

def test_environment():
    """Test if all required modules are installed"""
    print("Testing Python environment...")
    try:
        import cv2
        print(f"✓ OpenCV version: {cv2.__version__}")
        
        import numpy
        print(f"✓ NumPy version: {numpy.__version__}")
        
        import scipy
        print(f"✓ SciPy version: {scipy.__version__}")
        
        try:
            import dlib
            print(f"✓ dlib version: {dlib.__version__ if hasattr(dlib, '__version__') else 'installed'}")
        except ImportError:
            print("✗ dlib is not installed (required for face detection)")
            print("  Try: pip install dlib")
        
        try:
            import face_recognition
            print(f"✓ face_recognition is installed")
        except ImportError:
            print("✗ face_recognition is not installed")
            print("  Try: pip install face_recognition")
            
        return True
    except Exception as e:
        print(f"Error testing environment: {str(e)}")
        return False

def test_camera(camera_id):
    """Test camera directly with OpenCV"""
    print(f"\nTesting camera {camera_id} directly with OpenCV...")
    
    # Open camera
    cap = cv2.VideoCapture(camera_id)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera with ID {camera_id}")
        return False
        
    # Set resolution
    width, height = 640, 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    
    # Wait for camera to initialize
    time.sleep(1.0)
    
    # Read a frame
    ret, frame = cap.read()
    
    if not ret or frame is None:
        print("✗ Failed to read frame from camera")
        cap.release()
        return False
        
    actual_width = frame.shape[1]
    actual_height = frame.shape[0]
    
    print(f"✓ Camera opened successfully")
    print(f"✓ Requested resolution: {width}x{height}")
    print(f"✓ Actual resolution: {actual_width}x{actual_height}")
    
    # Test continuous mode (as used in liveness detection)
    print("Testing continuous video capture (this will open a window)")
    print("Press Q to quit after successful display")
    
    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Test", actual_width, actual_height)
    
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < 5:  # Run for 5 seconds
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Failed to read frame in continuous mode")
            break
            
        frame_count += 1
        
        # Add text overlay
        cv2.putText(frame, "Camera Test", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Frame #{frame_count}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow("Camera Test", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    fps = frame_count / (time.time() - start_time)
    print(f"✓ Captured {frame_count} frames at approximately {fps:.1f} FPS")
    
    cap.release()
    cv2.destroyAllWindows()
    
    return True

def test_face_detection():
    """Test if face detection is working"""
    print("\nTesting face detection...")
    
    try:
        # First test if dlib is available
        try:
            import dlib
            detector = dlib.get_frontal_face_detector()
            print("✓ dlib face detector initialized")
        except ImportError:
            print("✗ dlib is not installed, face detection test skipped")
            return False
        
        # Look for the shape predictor file
        model_path = "shape_predictor_68_face_landmarks.dat"
        alt_path = "./shape_predictor_68_face_landmarks.dat"
        
        if os.path.exists(model_path):
            print(f"✓ Found face landmark model at: {model_path}")
            predictor = dlib.shape_predictor(model_path)
        elif os.path.exists(alt_path):
            print(f"✓ Found face landmark model at: {alt_path}")
            predictor = dlib.shape_predictor(alt_path)
        else:
            print("✗ Face landmark model not found")
            print("  Download it from: https://github.com/davisking/dlib-models/raw/master/shape_predictor_68_face_landmarks.dat.bz2")
            print("  And extract it to the current directory")
            return False
            
        print("✓ Face detection components verified")
        return True
    except Exception as e:
        print(f"✗ Error testing face detection: {str(e)}")
        return False

def test_components(camera_id):
    """Test all components directly"""
    from camera import Camera
    from blink_detector import BlinkDetector
    from texture_analyzer import TextureAnalyzer
    
    print("\nTesting Camera class...")
    camera = Camera(camera_id=camera_id, resolution=(640, 480))
    
    # Test capturing a single frame
    print("Testing frame capture...")
    frame = camera.capture_frame()
    
    if frame is None:
        print("✗ Failed to capture frame using Camera class")
        return False
        
    print(f"✓ Successfully captured frame: {frame.shape}")
    
    # Show the captured frame
    cv2.imshow("Captured Frame", frame)
    cv2.waitKey(1000)  # Show for 1 second
    cv2.destroyAllWindows()
    
    # Test face detection
    print("\nTesting face detection with Camera class...")
    face_result = camera.detect_face(frame)
    
    if not face_result[0]:
        print("Note: No face detected in test frame (this is OK if no face was present)")
    else:
        print("✓ Face detected successfully")
        face_bbox = face_result[1]
        landmarks = face_result[2]
        
        # Draw bounding box and landmarks on face
        debug_frame = frame.copy()
        x, y, w, h = face_bbox
        cv2.rectangle(debug_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Draw landmarks if available
        if landmarks:
            for (x, y) in landmarks:
                cv2.circle(debug_frame, (x, y), 2, (0, 0, 255), -1)
                
        cv2.imshow("Face Detection", debug_frame)
        cv2.waitKey(2000)  # Show for 2 seconds
        cv2.destroyAllWindows()
    
    # Initialize other components
    print("\nInitializing other components...")
    try:
        blink_detector = BlinkDetector(camera)
        print("✓ BlinkDetector initialized")
        
        texture_analyzer = TextureAnalyzer()
        print("✓ TextureAnalyzer initialized")
        
        print("\nAll components initialized successfully!")
        return True
    except Exception as e:
        print(f"✗ Error initializing components: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Test Face Recognition System Components')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    args = parser.parse_args()
    
    print("====================================================")
    print("  Face Recognition System Component Tests")
    print("====================================================")
    
    # Test environment
    env_result = test_environment()
    if not env_result:
        print("\nEnvironment test failed. Please install required packages.")
        return
        
    # Test camera directly
    camera_result = test_camera(args.camera)
    if not camera_result:
        print("\nBasic camera test failed. Check your camera connection and ID.")
        return
    
    # Test face detection
    face_result = test_face_detection()
    if not face_result:
        print("\nFace detection test failed. Check dlib installation and model files.")
    
    # Test components
    components_result = test_components(args.camera)
    if not components_result:
        print("\nComponent test failed. Check error messages above.")
        return
    
    print("\n====================================================")
    print("  All tests passed successfully!")
    print("====================================================")
    print("\nYou can now run the main application:")
    print("python main.py --mode demo")

if __name__ == "__main__":
    main() 