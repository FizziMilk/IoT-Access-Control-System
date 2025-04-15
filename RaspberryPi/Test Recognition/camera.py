import cv2
import time
import face_recognition
import numpy as np
from collections import deque
import random
import dlib
import scipy.signal

class CameraSystem:
    def __init__(self, camera_id=0, resolution=(640, 480)):
        self.camera_id = camera_id
        self.resolution = resolution
        
        # Initialize eye tracking variables for blink detection
        self.eye_history = []  # For tracking eye aspect ratio over time
    
    def capture_face(self):
        """Capture an image from the camera with face detection"""
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
        # Show camera feed until space is pressed
        print("Position face in frame and press SPACE to capture or ESC to cancel")
        face_found = False
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image")
                break
            
            # Show the frame
            display_frame = frame.copy()
            
            # Try to find faces - ensure correct format
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Make sure the image is uint8 (0-255)
            if rgb_small_frame.dtype != np.uint8:
                rgb_small_frame = rgb_small_frame.astype(np.uint8)
                
            # Detect faces
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            # Draw rectangle around faces
            for (top, right, bottom, left) in face_locations:
                # Scale back up
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                face_found = True
            
            # Show instructions on the frame
            text = "SPACE: Capture, ESC: Cancel"
            if face_found:
                status = "Face detected"
                color = (0, 255, 0)
            else:
                status = "No face detected"
                color = (0, 0, 255)
            
            cv2.putText(display_frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, status, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            cv2.imshow("Capture Face", display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                cv2.destroyAllWindows()
                cap.release()
                return None
            elif key == 32:  # SPACE
                if face_found:
                    break
        
        # Get final frame and clean up
        ret, frame = cap.read()
        cap.release()
        cv2.destroyAllWindows()
        
        if not ret:
            return None
            
        return frame
    
    def detect_texture(self, face_image, test_type="all", debug=True):
        """
        Advanced texture analysis to detect printed/digital photos vs real faces.
        This function implements multiple texture detection methods to identify spoofing attacks.
        
        Args:
            face_image: Image containing only the face region
            test_type: Which test to run ("laplacian", "fft", "color", "gradient", "lbp", "moire", "reflection", or "all")
            debug: Whether to print debug information and return visualizations
            
        Returns:
            tuple: (is_real, debug_image) where is_real is a boolean and debug_image shows analysis results
        """
        if face_image is None or face_image.size == 0:
            return False, None
            
        # Resize for consistent analysis (maintain aspect ratio)
        height, width = face_image.shape[:2]
        max_dim = 300
        scale = min(max_dim / width, max_dim / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        resized = cv2.resize(face_image, (new_width, new_height))
        
        # Convert to grayscale for texture analysis
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Keep track of test results
        test_results = {}
        
        # Create debug visualization if running all tests or debugging specific test
        if debug:
            if test_type == "all":
                # Make sure the debug image is tall enough for all visualizations
                debug_img_height = new_height * 2 + 250
            else:
                # Smaller debug image for single test
                debug_img_height = new_height + 100
                
            debug_img = np.zeros((debug_img_height, new_width * 2, 3), dtype=np.uint8)
            
            # Place the original image - ensure dimensions match
            if resized.shape[0] != new_height or resized.shape[1] != new_width:
                resized_for_debug = cv2.resize(resized, (new_width, new_height))
                debug_img[0:new_height, 0:new_width] = resized_for_debug
            else:
                debug_img[0:new_height, 0:new_width] = resized
                
            # Add title
            cv2.putText(debug_img, "Original", (10, new_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Run tests based on test_type parameter
        if test_type in ["laplacian", "all"]:
            # ===================== TEST 1: TEXTURE DETAIL ANALYSIS =====================
            # Apply Laplacian filter to detect edges/details (2nd order derivative)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            laplacian_variance = np.var(laplacian)
            
            # Normalize for visualization
            if debug:
                laplacian_norm = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                laplacian_color = cv2.applyColorMap(laplacian_norm, cv2.COLORMAP_JET)
                # Ensure dimensions match
                if laplacian_color.shape[:2] != (new_height, new_width):
                    laplacian_color = cv2.resize(laplacian_color, (new_width, new_height))
                debug_img[0:new_height, new_width:new_width*2] = laplacian_color
                cv2.putText(debug_img, "Laplacian (Edge Detail)", (new_width + 10, new_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Real faces typically have higher Laplacian variance (more texture detail)
            texture_threshold = 150.0
            test_results["laplacian"] = laplacian_variance > texture_threshold
            
            if test_type != "all":
                return test_results["laplacian"], debug_img if debug else None
        
        if test_type in ["fft", "all"]:
            # ===================== TEST 2: HIGH FREQUENCY ANALYSIS =====================
            # Apply FFT to analyze frequency content
            f_transform = np.fft.fft2(gray)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
            
            # Calculate high-frequency energy (outer portions of FFT)
            h, w = magnitude_spectrum.shape
            center_y, center_x = h // 2, w // 2
            radius = min(h, w) // 4  # Inner circle radius (low frequencies)
            
            # Create a mask for high-frequency regions
            y, x = np.ogrid[:h, :w]
            mask = ((x - center_x)**2 + (y - center_y)**2) > radius**2
            
            # Calculate energy in high-frequency regions
            high_freq_energy = np.mean(magnitude_spectrum[mask])
            
            # Visualize FFT with high-frequency regions highlighted
            if debug:
                # Normalize spectrum for visualization
                magnitude_norm = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                magnitude_color = cv2.applyColorMap(magnitude_norm, cv2.COLORMAP_JET)
                
                # Highlight high-frequency regions
                highlight_mask = np.zeros((h, w, 3), dtype=np.uint8)
                highlight_mask[mask] = [0, 255, 0]
                magnitude_color = cv2.addWeighted(magnitude_color, 0.7, highlight_mask, 0.3, 0)
                
                # Place in debug image
                fft_display = cv2.resize(magnitude_color, (new_width, new_height))
                
                if test_type == "all":
                    debug_img[new_height+50:new_height*2+50, 0:new_width] = fft_display
                    cv2.putText(debug_img, "FFT (High Freq in Green)", (10, new_height*2 + 70), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                else:
                    debug_img[0:new_height, new_width:new_width*2] = fft_display
                    cv2.putText(debug_img, "FFT (High Freq in Green)", (new_width + 10, new_height + 20), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Real faces have significant high-frequency components
            high_freq_threshold = 18.0
            test_results["fft"] = high_freq_energy > high_freq_threshold
            
            if test_type != "all":
                return test_results["fft"], debug_img if debug else None
        
        if test_type in ["color", "all"]:
            # ===================== TEST 3: COLOR VARIANCE ANALYSIS =====================
            # Calculate color standard deviation in each channel
            b, g, r = cv2.split(resized)
            b_std = np.std(b)
            g_std = np.std(g)
            r_std = np.std(r)
            color_std_avg = (b_std + g_std + r_std) / 3
            
            # Also check for unnatural channel correlation (phone screens have more correlated RGB)
            b_g_corr = np.corrcoef(b.flatten(), g.flatten())[0,1]
            b_r_corr = np.corrcoef(b.flatten(), r.flatten())[0,1]
            g_r_corr = np.corrcoef(g.flatten(), r.flatten())[0,1]
            avg_color_corr = (abs(b_g_corr) + abs(b_r_corr) + abs(g_r_corr)) / 3
            
            # Printed/digital images often have less color variance within skin tones
            color_threshold = 45.0
            corr_penalty = 10.0 * avg_color_corr  # Reduce score for high correlation
            adjusted_color_std = color_std_avg - corr_penalty
            test_results["color"] = adjusted_color_std > color_threshold
            
            # Visualize color channels for debugging
            if debug and test_type != "all":
                # Create color channel visualization
                color_vis = np.zeros((new_height, new_width*2, 3), dtype=np.uint8)
                # Show each channel
                for i, (channel, name, color) in enumerate(zip([b, g, r], ["Blue", "Green", "Red"], [(255,0,0), (0,255,0), (0,0,255)])):
                    vis_channel = cv2.cvtColor(channel, cv2.COLOR_GRAY2BGR)
                    # Add a color tint
                    color_tint = np.zeros_like(vis_channel)
                    color_tint[:] = color
                    vis_channel = cv2.addWeighted(vis_channel, 0.7, color_tint, 0.3, 0)
                    
                    pos_y = i * (new_height // 3)
                    h_segment = new_height // 3
                    if pos_y + h_segment <= new_height:
                        debug_img[pos_y:pos_y+h_segment, new_width:new_width*2] = vis_channel[0:h_segment, 0:new_width]
                        # Map color names to actual variable names
                        std_value = b_std if name == "Blue" else (g_std if name == "Green" else r_std)
                        cv2.putText(debug_img, f"{name}: std={std_value:.1f}", 
                                   (new_width + 10, pos_y + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            if test_type != "all":
                return test_results["color"], debug_img if debug else None
        
        if test_type in ["gradient", "all"]:
            # ===================== TEST 4: GRADIENT ANALYSIS =====================
            # Calculate image gradients (1st order derivatives)
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Calculate gradient complexity (variance of gradient directions)
            gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
            direction_variance = np.var(gradient_direction)
            
            # Digital displays often show more uniform gradient patterns
            gradient_threshold = 1200.0
            test_results["gradient"] = direction_variance > gradient_threshold
            
            # Visualize gradient for debugging
            if debug and test_type != "all":
                # Create gradient visualization
                sobel_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                sobel_color = cv2.applyColorMap(sobel_norm, cv2.COLORMAP_JET)
                if sobel_color.shape[:2] != (new_height, new_width):
                    sobel_color = cv2.resize(sobel_color, (new_width, new_height))
                    
                debug_img[0:new_height, new_width:new_width*2] = sobel_color
                cv2.putText(debug_img, "Gradient Magnitude", (new_width + 10, new_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if test_type != "all":
                return test_results["gradient"], debug_img if debug else None
        
        if test_type in ["lbp", "all"]:
            # ===================== TEST 5: LOCAL BINARY PATTERN ANALYSIS =====================
            # Calculate simple LBP (Local Binary Pattern)
            def local_binary_pattern(image):
                h, w = image.shape
                lbp = np.zeros_like(image)
                for i in range(1, h-1):
                    for j in range(1, w-1):
                        center = image[i, j]
                        binary = (image[i-1:i+2, j-1:j+2] >= center).flatten()
                        # Skip the center pixel (which is always 0)
                        binary = np.delete(binary, 4)
                        # Convert binary to decimal
                        lbp_value = np.sum(binary * (2**np.arange(8)))
                        lbp[i, j] = lbp_value
                return lbp
            
            # Calculate LBP and its histogram
            lbp = local_binary_pattern(gray)
            hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
            hist = hist / np.sum(hist)  # Normalize
            
            # Calculate LBP uniformity (real faces have more uniform LBP distribution)
            lbp_uniformity = 1.0 - np.std(hist)
            
            # Digital displays often show less uniform LBP patterns
            lbp_threshold = 0.93
            test_results["lbp"] = lbp_uniformity < lbp_threshold
            
            # Visualize LBP for debugging
            if debug and test_type != "all":
                # Normalize LBP for visualization
                lbp_norm = cv2.normalize(lbp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                lbp_color = cv2.applyColorMap(lbp_norm, cv2.COLORMAP_JET)
                if lbp_color.shape[:2] != (new_height, new_width):
                    lbp_color = cv2.resize(lbp_color, (new_width, new_height))
                    
                debug_img[0:new_height, new_width:new_width*2] = lbp_color
                cv2.putText(debug_img, "Local Binary Pattern", (new_width + 10, new_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if test_type != "all":
                return test_results["lbp"], debug_img if debug else None
        
        if test_type in ["moire", "all"]:
            # ===================== TEST 6: MOIRÉ PATTERN DETECTION =====================
            # Digital displays often show moiré interference patterns at certain scales
            # Apply bandpass filter to detect moiré patterns
            
            # Create bandpass filter (in frequency domain)
            rows, cols = gray.shape
            crow, ccol = rows // 2, cols // 2
            
            # Create a mask with a ring to isolate frequencies where moiré patterns appear
            mask = np.ones((rows, cols), np.uint8)
            r_outer = min(crow, ccol) // 2
            r_inner = r_outer // 2
            
            center = [crow, ccol]
            y, x = np.ogrid[:rows, :cols]
            mask_area = ((x - center[1])**2 + (y - center[0])**2 >= r_inner**2) & \
                        ((x - center[1])**2 + (y - center[0])**2 <= r_outer**2)
            mask[mask_area] = 0
            
            # Apply FFT, mask frequencies, then IFFT
            dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)
            
            # Apply mask to isolate frequency band
            dft_shift[:,:,0] = dft_shift[:,:,0] * mask
            dft_shift[:,:,1] = dft_shift[:,:,1] * mask
            
            # Inverse FFT to get filtered image
            f_ishift = np.fft.ifftshift(dft_shift)
            img_back = cv2.idft(f_ishift)
            filtered = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
            
            # Normalize filtered image
            cv2.normalize(filtered, filtered, 0, 255, cv2.NORM_MINMAX)
            filtered = filtered.astype(np.uint8)
            
            # Calculate energy in the filtered image (high energy indicates moiré patterns)
            moire_energy = np.mean(filtered)
            
            # Digital displays often show moiré patterns in this frequency band
            moire_threshold = 15.0
            test_results["moire"] = moire_energy < moire_threshold
            
            # Add filtered images for moiré pattern visualization
            if debug:
                if new_width > 0 and new_height > 0:
                    moire_display = cv2.applyColorMap(filtered, cv2.COLORMAP_JET)
                    if moire_display.shape[:2] != (new_height, new_width):
                        moire_display = cv2.resize(moire_display, (new_width, new_height))
                    
                    if test_type == "all":
                        debug_img[new_height+50:new_height*2+50, new_width:new_width*2] = moire_display
                        cv2.putText(debug_img, "Moiré Pattern Filter", (new_width + 10, new_height*2 + 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    else:
                        debug_img[0:new_height, new_width:new_width*2] = moire_display
                        cv2.putText(debug_img, "Moiré Pattern Filter", (new_width + 10, new_height + 20), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if test_type != "all":
                return test_results["moire"], debug_img if debug else None
        
        if test_type in ["reflection", "all"]:
            # ===================== TEST 7: REFLECTION DETECTION =====================
            # Digital displays often show reflections with different characteristics
            
            # Check for unnatural brightness patterns
            hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
            v_channel = hsv[:,:,2]
            
            # Calculate brightness standard deviation and max regions
            brightness_std = np.std(v_channel)
            bright_pixels = (v_channel > 240).sum() / v_channel.size
            
            # Also check for abnormal brightness distribution (screens are more uniform)
            brightness_hist, _ = np.histogram(v_channel, bins=64, range=(0, 256))
            brightness_hist = brightness_hist / np.sum(brightness_hist)
            brightness_entropy = -np.sum(brightness_hist * np.log2(brightness_hist + 1e-10))
            
            # Phone screens often have unnaturally bright regions or reflections
            reflection_threshold = 0.02
            # Reduce the threshold if entropy is low (uniform brightness, typical of screens)
            if brightness_entropy < 3.0:
                reflection_threshold *= 0.8
            test_results["reflection"] = bright_pixels < reflection_threshold
            
            # Visualize brightness for debugging
            if debug and test_type != "all":
                # Create brightness visualization
                v_norm = cv2.normalize(v_channel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                v_color = cv2.applyColorMap(v_norm, cv2.COLORMAP_JET)
                if v_color.shape[:2] != (new_height, new_width):
                    v_color = cv2.resize(v_color, (new_width, new_height))
                    
                debug_img[0:new_height, new_width:new_width*2] = v_color
                cv2.putText(debug_img, "Brightness Analysis", (new_width + 10, new_height + 20), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if test_type != "all":
                return test_results["reflection"], debug_img if debug else None
        
        # ===================== COMBINE RESULTS FOR ALL TESTS =====================
        if test_type == "all":
            # Visualize test results
            if debug:
                result_img = np.zeros((200, new_width * 2, 3), dtype=np.uint8)
                texts = [
                    f"1. Texture Detail: {'PASS' if test_results.get('laplacian', False) else 'FAIL'} ({laplacian_variance:.1f} > {texture_threshold})",
                    f"2. High Frequency: {'PASS' if test_results.get('fft', False) else 'FAIL'} ({high_freq_energy:.1f} > {high_freq_threshold})",
                    f"3. Color Variance: {'PASS' if test_results.get('color', False) else 'FAIL'} ({adjusted_color_std:.1f} > {color_threshold})",
                    f"4. Gradient Complexity: {'PASS' if test_results.get('gradient', False) else 'FAIL'} ({direction_variance:.1f} > {gradient_threshold})",
                    f"5. LBP Uniformity: {'PASS' if test_results.get('lbp', False) else 'FAIL'} ({lbp_uniformity:.3f} < {lbp_threshold})",
                    f"6. Moiré Pattern: {'PASS' if test_results.get('moire', False) else 'FAIL'} ({moire_energy:.1f} < {moire_threshold})",
                    f"7. Reflection: {'PASS' if test_results.get('reflection', False) else 'FAIL'} ({bright_pixels*100:.2f}% < {reflection_threshold*100}%)"
                ]
                
                for i, text in enumerate(texts):
                    color = (0, 255, 0) if "PASS" in text else (0, 0, 255)
                    cv2.putText(result_img, text, (10, 25 + i * 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Place in debug image
                debug_img[new_height*2+50:new_height*2+250, 0:new_width*2] = result_img
            
            # Count how many tests pass
            passing_tests = sum([
                test_results.get("laplacian", False), 
                test_results.get("fft", False), 
                test_results.get("color", False), 
                test_results.get("gradient", False),
                test_results.get("lbp", False),
                test_results.get("moire", False),
                test_results.get("reflection", False)
            ])
            
            # Require at least 5 out of 7 tests to pass for the image to be classified as real
            tests_required = 5
            is_real = passing_tests >= tests_required
            
            # Display final result
            if debug:
                result_text = f"REAL FACE ({passing_tests}/{tests_required} tests passed)" if is_real else f"FAKE FACE ({passing_tests}/{tests_required} tests passed)"
                result_color = (0, 255, 0) if is_real else (0, 0, 255)
                cv2.putText(debug_img, result_text, (new_width - 150, new_height*2 + 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, result_color, 2)
                
                print(f"Texture Analysis: {'REAL' if is_real else 'FAKE'} - {passing_tests}/{tests_required} tests passed")
                if "laplacian" in test_results:
                    print(f"  - Laplacian variance: {laplacian_variance:.1f} ({'PASS' if test_results['laplacian'] else 'FAIL'})")
                if "fft" in test_results:
                    print(f"  - High frequency energy: {high_freq_energy:.1f} ({'PASS' if test_results['fft'] else 'FAIL'})")
                if "color" in test_results:
                    print(f"  - Color variance: {adjusted_color_std:.1f} ({'PASS' if test_results['color'] else 'FAIL'})")
                if "gradient" in test_results:
                    print(f"  - Gradient complexity: {direction_variance:.1f} ({'PASS' if test_results['gradient'] else 'FAIL'})")
                if "lbp" in test_results:
                    print(f"  - LBP uniformity: {lbp_uniformity:.3f} ({'PASS' if test_results['lbp'] else 'FAIL'})")
                if "moire" in test_results:
                    print(f"  - Moiré pattern: {moire_energy:.1f} ({'PASS' if test_results['moire'] else 'FAIL'})")
                if "reflection" in test_results:
                    print(f"  - Reflection detection: {bright_pixels*100:.2f}% ({'PASS' if test_results['reflection'] else 'FAIL'})")
            
            return is_real, debug_img if debug else None
        
        # If test_type is not recognized, return False
        return False, debug_img if debug else None
    
    def detect_blink_with_movement_rejection(self, frame, face_bbox=None, landmarks=None, visualize=True):
        """
        Enhanced blink detection that filters out false positives caused by rapid camera movement.
        
        This method analyzes the eye aspect ratio (EAR) while monitoring face movement to reject
        cases where the entire face moves in a way that could be mistaken for a blink.
        
        Args:
            frame: Input video frame
            face_bbox: Optional face bounding box (x, y, w, h) if already detected
            landmarks: Optional facial landmarks if already detected
            visualize: Whether to draw visualization on the frame
            
        Returns:
            tuple: (is_blink_detected, visualization_frame, debug_data)
        """
        if frame is None:
            return False, None, {}
        
        debug_data = {
            "ear_values": [],
            "movement_values": [],
            "time_history": []
        }
        
        # Detect face and landmarks if not provided
        if face_bbox is None or landmarks is None:
            face_results = self.detect_face(frame)
            if not face_results[0]:  # No face detected
                return False, frame, debug_data
            face_bbox = face_results[1]
            landmarks = face_results[2]
        
        # Create a copy for visualization
        vis_frame = frame.copy() if visualize else None
        
        # Extract eye landmarks
        left_eye_landmarks = landmarks[36:42]  # Left eye points 
        right_eye_landmarks = landmarks[42:48]  # Right eye points
        
        # Calculate Eye Aspect Ratio (EAR)
        left_ear = self._calculate_ear(left_eye_landmarks)
        right_ear = self._calculate_ear(right_eye_landmarks)
        ear = (left_ear + right_ear) / 2.0
        
        # Get eye contours for later comparison
        left_eye_contour = self._get_eye_contour(left_eye_landmarks)
        right_eye_contour = self._get_eye_contour(right_eye_landmarks)
        
        # Store current timestamp, EAR value
        current_time = time.time()
        
        # Track face position for movement analysis - use nose tip for stability
        nose_tip = landmarks[30]
        
        # Update history
        self.eye_history.append((current_time, ear, nose_tip, left_eye_contour, right_eye_contour))
        
        # Keep only recent history (last 2 seconds)
        while self.eye_history and current_time - self.eye_history[0][0] > 2.0:
            self.eye_history.pop(0)
        
        # Need at least 5 frames of history to detect blinks reliably
        if len(self.eye_history) < 5:
            return False, vis_frame, debug_data
        
        # Calculate EAR stats 
        ear_values = [entry[1] for entry in self.eye_history]
        mean_ear = np.mean(ear_values)
        min_ear = min(ear_values)
        ear_std = np.std(ear_values)
        
        # Create sliding window for recent values (last ~0.5 second)
        recent_window = self.eye_history[-min(15, len(self.eye_history)):]
        recent_ear_values = [entry[1] for entry in recent_window]
        
        # Get timestamps for debug visualization
        timestamps = [entry[0] - current_time for entry in self.eye_history]
        debug_data["ear_values"] = ear_values
        debug_data["time_history"] = timestamps
        
        # =============== MOVEMENT ANALYSIS ===============
        # Calculate face movement between frames
        movement_values = []
        contour_changes = []
        avg_movement_vector = np.zeros(2)
        
        for i in range(1, len(self.eye_history)):
            prev_nose = self.eye_history[i-1][2]
            curr_nose = self.eye_history[i][2]
            
            # Calculate nose movement vector
            movement_vector = np.array([curr_nose[0] - prev_nose[0], curr_nose[1] - prev_nose[1]])
            movement_magnitude = np.linalg.norm(movement_vector)
            
            # Add to movement history
            movement_values.append(movement_magnitude)
            
            # Track overall movement direction
            if movement_magnitude > 0:
                normalized_vector = movement_vector / movement_magnitude
                avg_movement_vector += normalized_vector
            
            # Calculate eye contour change
            prev_left_contour = self.eye_history[i-1][3]
            curr_left_contour = self.eye_history[i][3]
            prev_right_contour = self.eye_history[i-1][4]
            curr_right_contour = self.eye_history[i][4]
            
            # Compare contour change with movement direction
            left_contour_change = self._calculate_contour_change(prev_left_contour, curr_left_contour)
            right_contour_change = self._calculate_contour_change(prev_right_contour, curr_right_contour)
            
            contour_changes.append((left_contour_change, right_contour_change))
            
        if movement_values:
            avg_movement_vector = avg_movement_vector / len(movement_values)
            avg_movement = np.mean(movement_values)
            max_movement = max(movement_values) if movement_values else 0
            movement_std = np.std(movement_values) if len(movement_values) > 1 else 0
            debug_data["movement_values"] = movement_values
        else:
            avg_movement = 0
            max_movement = 0
            movement_std = 0
            debug_data["movement_values"] = [0]
        
        # =============== BLINK DETECTION LOGIC ===============
        # A real blink should show:
        # 1. A significant drop in EAR (eye closing) followed by a return to normal (eye opening)
        # 2. Minimal vertical face movement during the EAR drop (not caused by camera motion)
        # 3. Consistent movement patterns between both eyes
        
        # Look for potential blink patterns in the recent window
        blink_detected = False
        blink_confidence = 0
        
        # Check for significant EAR dip
        ear_threshold = 0.2  # Threshold for eye closure
        ear_dip_detected = min_ear < mean_ear - 0.05 and min_ear < ear_threshold
        
        # Analyze EAR trajectory for blink-like pattern
        # In a real blink, EAR drops quickly then rises back
        ear_trajectory_score = 0
        ear_recovery = False
        
        # Need at least 10 frames to analyze trajectory
        if len(recent_ear_values) >= 10:
            # Check if EAR dropped then recovered
            min_idx = np.argmin(recent_ear_values)
            
            # Blink typically happens in the middle of the window
            if 2 < min_idx < len(recent_ear_values) - 3:
                pre_min = recent_ear_values[:min_idx]
                post_min = recent_ear_values[min_idx+1:]
                
                # In a real blink, EAR should decrease before minimum and increase after
                if (np.mean(pre_min[:3]) > np.mean(pre_min[-3:]) and 
                    np.mean(post_min[-3:]) > np.mean(post_min[:3])):
                    ear_trajectory_score += 1
                    
                # Check if EAR recovered to at least 80% of pre-blink value
                if len(post_min) > 0 and len(pre_min) > 0:
                    pre_blink_ear = np.mean(pre_min[:3])
                    post_blink_ear = np.max(post_min)
                    if post_blink_ear > 0.8 * pre_blink_ear:
                        ear_recovery = True
                        ear_trajectory_score += 1
        
        # =============== MOVEMENT REJECTION LOGIC ===============
        # For spoofing detection, analyze if the eye movements are consistent with:
        # 1. Natural blink physics (both eyes closing/opening simultaneously)
        # 2. Not caused by whole-face movement (especially vertical shaking)
        
        # Movement rejection score (higher = more likely to be rejected as fake)
        movement_rejection_score = 0
        
        # Check if movement coincides with EAR change
        if movement_values and recent_ear_values:
            # Find when minimum EAR occurred (potential blink)
            min_ear_idx = np.argmin(recent_ear_values)
            
            # Check if there was significant movement right at the EAR dip
            if min_ear_idx < len(movement_values):
                # Calculate movement around the minimum EAR point
                blink_moment_movement = movement_values[max(0, min_ear_idx-2):min(len(movement_values), min_ear_idx+3)]
                if blink_moment_movement and np.mean(blink_moment_movement) > 2.0:
                    # High movement during EAR dip is suspicious
                    movement_rejection_score += 2
                    
            # Check if movement is primarily vertical (common in spoofing attempts)
            vertical_ratio = abs(avg_movement_vector[1]) / (abs(avg_movement_vector[0]) + 0.001)
            if vertical_ratio > 1.5 and max_movement > 3.0:  # Primarily vertical movement
                movement_rejection_score += 1
                
            # Check for periodic movement pattern (shaking)
            if len(movement_values) > 10:
                # Calculate autocorrelation to detect rhythmic movement
                autocorr = np.correlate(movement_values, movement_values, mode='full')
                autocorr = autocorr[len(autocorr)//2:]  # Use only second half
                
                # Normalize autocorrelation
                if autocorr[0] != 0:
                    autocorr = autocorr / autocorr[0]
                    
                # Look for peaks in autocorrelation (indicates periodic movement)
                peaks, _ = scipy.signal.find_peaks(autocorr, height=0.5)
                if len(peaks) > 1:  # Multiple peaks indicate periodic movement
                    movement_rejection_score += 1
                
        # Look at contour changes to see if eyes are changing in a natural way
        if contour_changes:
            # In a real blink, both eyes should close/open similarly
            left_changes = [lc for lc, _ in contour_changes]
            right_changes = [rc for _, rc in contour_changes]
            
            # Calculate correlation between left and right eye contour changes
            if len(left_changes) > 3 and len(right_changes) > 3:
                eye_correlation = np.corrcoef(left_changes, right_changes)[0, 1]
                
                # In a real blink, eye movements are highly correlated
                if eye_correlation < 0.7:  # Low correlation is suspicious
                    movement_rejection_score += 1
        
        # =============== FINAL DECISION LOGIC ===============
        # Calculate overall blink confidence
        blink_score = 0
        
        # Strong EAR dip is the primary indicator
        if ear_dip_detected:
            blink_score += 2
            
        # Proper EAR trajectory adds confidence
        blink_score += ear_trajectory_score
        
        # EAR recovery is important
        if ear_recovery:
            blink_score += 1
            
        # Subtract movement rejection score
        blink_score -= movement_rejection_score
        
        # Final decision
        blink_detected = blink_score >= 3
        blink_confidence = blink_score
        
        # Add debugging information
        debug_data["blink_score"] = blink_score
        debug_data["movement_rejection_score"] = movement_rejection_score
        debug_data["ear_dip_detected"] = ear_dip_detected
        debug_data["ear_trajectory_score"] = ear_trajectory_score
        debug_data["ear_recovery"] = ear_recovery
        
        # Visualize if requested
        if visualize:
            # Draw eye contours
            for eye_points in [left_eye_landmarks, right_eye_landmarks]:
                pts = np.array(eye_points, dtype=np.int32)
                cv2.polylines(vis_frame, [pts], True, (0, 255, 255), 1)
            
            # Display current EAR
            ear_text = f"EAR: {ear:.2f}"
            cv2.putText(vis_frame, ear_text, (face_bbox[0], face_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display movement information
            movement_text = f"Movement: {avg_movement:.1f}"
            cv2.putText(vis_frame, movement_text, (face_bbox[0], face_bbox[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Display blink detection result
            result_color = (0, 255, 0) if blink_detected else (0, 0, 255)
            result_text = "BLINK DETECTED" if blink_detected else "No Blink"
            score_text = f"Score: {blink_score}"
            
            cv2.putText(vis_frame, result_text, (face_bbox[0] + face_bbox[2] - 160, face_bbox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, result_color, 2)
            cv2.putText(vis_frame, score_text, (face_bbox[0] + face_bbox[2] - 80, face_bbox[1] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add plots of EAR and movement if there's enough history
            if len(ear_values) > 3:
                self._add_tracking_plots(vis_frame, timestamps, ear_values, movement_values, blink_detected)
        
        return blink_detected, vis_frame, debug_data
        
    def _calculate_contour_change(self, prev_contour, curr_contour):
        """
        Calculate the change in eye contour shape between frames.
        
        Args:
            prev_contour: Previous eye contour points
            curr_contour: Current eye contour points
            
        Returns:
            float: Contour change metric
        """
        if not prev_contour or not curr_contour:
            return 0
            
        # Calculate average height difference
        prev_heights = [p[1] for p in prev_contour]
        curr_heights = [p[1] for p in curr_contour]
        
        return np.mean(np.abs(np.array(curr_heights) - np.array(prev_heights)))
        
    def _get_eye_contour(self, eye_landmarks):
        """
        Extract eye contour points for analysis.
        
        Args:
            eye_landmarks: List of eye landmark points
            
        Returns:
            list: Contour points
        """
        return [(int(p[0]), int(p[1])) for p in eye_landmarks]
    
    def _add_tracking_plots(self, frame, timestamps, ear_values, movement_values, blink_detected):
        """
        Add plots of EAR and movement to the visualization frame.
        
        Args:
            frame: Frame to draw on
            timestamps: List of relative timestamps
            ear_values: List of EAR values
            movement_values: List of movement values
            blink_detected: Whether a blink was detected
        """
        h, w = frame.shape[:2]
        plot_h, plot_w = 120, 300
        plot_x, plot_y = w - plot_w - 10, 10
        
        # Create plot area
        plot_area = np.ones((plot_h, plot_w, 3), dtype=np.uint8) * 50
        
        # Normalize values for plotting
        if ear_values:
            norm_ear = np.array(ear_values)
            norm_ear = (norm_ear - min(0.15, min(norm_ear))) / (max(0.4, max(norm_ear)) - min(0.15, min(norm_ear)))
            norm_ear = 1.0 - norm_ear  # Invert so blinks go up
            
            # Plot EAR values
            for i in range(1, len(norm_ear)):
                p1 = (int(plot_w * (i-1) / len(norm_ear)), int(plot_h * 0.8 * norm_ear[i-1]))
                p2 = (int(plot_w * i / len(norm_ear)), int(plot_h * 0.8 * norm_ear[i]))
                cv2.line(plot_area, p1, p2, (0, 255, 255), 1)
        
        # Plot movement values if available
        if movement_values:
            # Normalize movement values
            max_move = max(10.0, max(movement_values))
            norm_move = np.array(movement_values) / max_move
            
            # Plot movement values
            for i in range(1, len(norm_move)):
                p1 = (int(plot_w * (i-1) / len(norm_move)), int(plot_h * (0.9 - 0.4 * norm_move[i-1])))
                p2 = (int(plot_w * i / len(norm_move)), int(plot_h * (0.9 - 0.4 * norm_move[i])))
                cv2.line(plot_area, p1, p2, (0, 180, 0), 1)
        
        # Add labels
        cv2.putText(plot_area, "EAR", (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(plot_area, "Movement", (5, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 180, 0), 1)
        
        # Add threshold line for EAR
        threshold_y = int(plot_h * 0.8 * (1.0 - (0.2 - 0.15) / (0.4 - 0.15)))
        cv2.line(plot_area, (0, threshold_y), (plot_w, threshold_y), (0, 0, 255), 1, cv2.LINE_DASH)
        
        # Add blink detection indicator
        if blink_detected:
            cv2.putText(plot_area, "BLINK", (plot_w - 60, 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Overlay plot on frame
        frame[plot_y:plot_y+plot_h, plot_x:plot_x+plot_w] = plot_area
    
    def detect_liveness_multi(self, timeout=30):  # Increased timeout from 20 to 30 seconds
        """
        Advanced liveness detection using multiple techniques:
        1. Blink detection with higher thresholds
        2. Random head movement challenges
        3. Texture analysis to detect printed photos
        
        Args:
            timeout: Maximum time in seconds for the entire verification
            
        Returns:
            tuple: (is_live, face_image)
        """
        print("Enhanced liveness detection started...")
        
        # Enable skipping for troubleshooting - set these to True to skip specific tests
        # In a production environment, these should all be False
        SKIP_TEXTURE = False    # For troubleshooting only
        SKIP_BLINK = False      # For troubleshooting only
        SKIP_MOVEMENT = False   # For troubleshooting only
        
        # Allow user to toggle which tests to run with keyboard during detection
        # These will be updated during the detection loop if user presses keys
        skip_texture = SKIP_TEXTURE
        skip_blink = SKIP_BLINK
        skip_movement = SKIP_MOVEMENT
        
        # Initialize stage tracking
        stage_passed = {
            "texture": skip_texture,
            "blink": skip_blink,
            "head_movement": skip_movement
        }
        
        # Initialize camera
        cap = cv2.VideoCapture(self.camera_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        # Wait for camera to initialize
        time.sleep(0.5)
        
        # Parameters for blink detection - more strict thresholds
        EYE_AR_THRESH = 0.3  # Increased from 0.25 for lower framerates
        EYE_AR_CONSEC_FRAMES = 2
        
        # Store eye aspect ratios for blink detection
        ear_history = deque(maxlen=8)  # Longer history for better analysis
        
        # For head movement challenge
        movement_type = random.choice(["left", "right", "up", "down"])
        movement_start_time = None
        movement_completed = False
        movement_positions = []  # To track nose positions
        
        # Track progress and time
        start_time = time.time()
        current_stage = "init"
        face_image = None
        
        # Function to calculate eye aspect ratio
        def eye_aspect_ratio(eye):
            # Compute euclidean distance between eye landmarks
            A = np.linalg.norm(eye[1] - eye[5])
            B = np.linalg.norm(eye[2] - eye[4])
            C = np.linalg.norm(eye[0] - eye[3])
            # Calculate ratio
            ear = (A + B) / (2.0 * C)
            return ear
        
        # Begin with texture analysis - this should happen first with a static face
        print("Stage 1: Looking for a real face (not a printout)...")
        current_stage = "texture"
        texture_frames = 0
        texture_passed_count = 0
        texture_failed_count = 0  # Track failures too
        face_found = False
        
        # Use these to control when current_stage and other globals are updated
        # to avoid race conditions between updating state and using new state
        should_transition_to_blink = False
        should_transition_to_movement = False
        
        # For texture test rotation - run one test at a time in sequence
        texture_tests = ["laplacian", "fft", "color", "gradient", "lbp", "moire", "reflection"]
        current_texture_test_idx = 0
        texture_test_results = {test: 0 for test in texture_tests}  # Track passes for each test (0 = not tested yet)

        # Function to safely transition between stages
        def transition_to_stage(new_stage):
            nonlocal current_stage, ear_history, movement_positions, movement_start_time
            
            # Clear previous stage data
            ear_history.clear()
            movement_positions = []
            
            # Set new stage
            current_stage = new_stage
            
            # Initialize stage-specific variables
            if new_stage == "blink":
                print("Stage 2: Please blink naturally...")
            elif new_stage == "head_movement":
                print(f"Stage 3: Please move your head {movement_type.upper()}")
                movement_start_time = time.time()
                
        while True:
            # Check for overall timeout
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout:
                print("Liveness detection timeout")
                break
                
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Handle any pending stage transitions before processing the frame
            if should_transition_to_blink:
                transition_to_stage("blink")
                should_transition_to_blink = False
            elif should_transition_to_movement:
                transition_to_stage("head_movement")
                should_transition_to_movement = False
            
            # Convert to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process at a smaller size for speed
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=0.5, fy=0.5)
            
            # Display frame with overlays
            display_frame = frame.copy()
            
            # Calculate time remaining
            time_left = max(0, timeout - elapsed_time)
            
            # Add keyboard shortcuts for toggling tests (for troubleshooting)
            cv2.putText(display_frame, "Press T to toggle texture test", 
                      (10, display_frame.shape[0]-60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press B to toggle blink test", 
                      (10, display_frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(display_frame, "Press M to toggle movement test", 
                      (10, display_frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(small_frame)
            
            if face_locations:
                # Scale back face location to full size
                for (top, right, bottom, left) in face_locations:
                    # Scale back up
                    top *= 2
                    right *= 2
                    bottom *= 2
                    left *= 2
                    
                    # Draw face rectangle
                    cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    
                    # Extract face region for texture analysis
                    if current_stage == "texture" and texture_frames % 5 == 0:  # Check every 5 frames (more frequent)
                        face_region = frame[top:bottom, left:right]
                        if face_region.size > 0 and face_region.shape[0] > 20 and face_region.shape[1] > 20:  # Ensure the region is valid and large enough
                            try:
                                # Get the current test to run
                                current_test = texture_tests[current_texture_test_idx]
                                
                                # Run only one texture test at a time
                                is_test_passed, debug_img = self.detect_texture(face_region, test_type=current_test, debug=True)
                                
                                # Track test result
                                if is_test_passed:
                                    texture_test_results[current_test] += 1
                                    cv2.putText(display_frame, f"{current_test.upper()} TEST PASSED", (left, top-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                                else:
                                    cv2.putText(display_frame, f"{current_test.upper()} TEST FAILED", (left, top-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                                
                                # Show test results on screen
                                for i, (test, passes) in enumerate(texture_test_results.items()):
                                    status = "N/A" if passes == 0 else f"PASS ({passes})" if passes >= 2 else f"FAIL"
                                    color = (255, 255, 255) if passes == 0 else (0, 255, 0) if passes >= 2 else (0, 0, 255)
                                    cv2.putText(display_frame, f"{test}: {status}", 
                                              (10, 110 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                                
                                # Show debug image if available
                                if debug_img is not None:
                                    # Only show if the debug image is valid
                                    if debug_img.shape[0] > 0 and debug_img.shape[1] > 0:
                                        # Resize to a reasonable size 
                                        debug_img_resized = cv2.resize(debug_img, (320, 240))
                                        # Place in top right corner
                                        h, w = debug_img_resized.shape[:2]
                                        display_frame[10:10+h, display_frame.shape[1]-w-10:display_frame.shape[1]-10] = debug_img_resized
                                
                                # Rotate to next test after a few frames with the same test
                                if texture_frames % 15 == 0:
                                    current_texture_test_idx = (current_texture_test_idx + 1) % len(texture_tests)
                                
                                # Check if we've collected enough passes across tests
                                passing_tests = sum(1 for passes in texture_test_results.values() if passes >= 2)
                                if passing_tests >= 4:  # Need 4 out of 7 tests to pass
                                    stage_passed["texture"] = True
                                    print("✓ Texture check passed! It appears to be a real face.")
                                    print(f"Test results: {texture_test_results}")
                                    # Schedule transition to blink detection
                                    should_transition_to_blink = True
                                
                            except Exception as e:
                                print(f"Error in texture analysis: {str(e)}")
                    texture_frames += 1
                
                # Get facial landmarks for further analysis
                face_landmarks = face_recognition.face_landmarks(small_frame, face_locations)
                
                for landmarks in face_landmarks:
                    # Check if eye landmarks exist for blink detection
                    if current_stage in ["blink", "head_movement"] and 'left_eye' in landmarks and 'right_eye' in landmarks:
                        try:
                            # Convert landmarks to numpy arrays
                            left_eye = np.array(landmarks['left_eye'])
                            right_eye = np.array(landmarks['right_eye'])
                            
                            # Scale back up to full size
                            left_eye = left_eye * 2
                            right_eye = right_eye * 2
                            
                            # Draw eyes
                            if current_stage == "blink":
                                for eye_point in left_eye:
                                    cv2.circle(display_frame, tuple(eye_point), 2, (0, 255, 255), -1)
                                for eye_point in right_eye:
                                    cv2.circle(display_frame, tuple(eye_point), 2, (0, 255, 255), -1)
                                
                                # Calculate eye aspect ratio
                                left_ear = eye_aspect_ratio(left_eye)
                                right_ear = eye_aspect_ratio(right_eye)
                                
                                # Average the eye aspect ratio
                                ear = (left_ear + right_ear) / 2.0
                                ear_history.append(ear)
                                
                                # Display EAR value
                                cv2.putText(display_frame, f"EAR: {ear:.2f}", 
                                          (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                
                                # Check for blink using our improved function with movement rejection
                                if self.detect_blink_with_movement_rejection(frame, face_locations, landmarks, True):
                                    stage_passed["blink"] = True
                                    print("✓ Blink detected! Moving to head movement test.")
                                    # Schedule transition to head movement
                                    should_transition_to_movement = True
                        except Exception as e:
                            print(f"Error in blink detection: {str(e)}")
                    
                    # Track nose position for head movement challenge
                    if current_stage == "head_movement" and 'nose_tip' in landmarks:
                        try:
                            # Get nose point - convert to numpy array and handle it safely
                            nose_points = np.array(landmarks['nose_tip'])
                            if len(nose_points) > 0:
                                # Use the mean position of all nose points
                                nose_point = np.mean(nose_points, axis=0) * 2  # Scale back up
                                
                                # Make sure it's a valid coordinate
                                if not np.isnan(nose_point).any():
                                    # Convert to integers for drawing
                                    nose_x, nose_y = int(nose_point[0]), int(nose_point[1])
                                    
                                    # Draw nose point
                                    cv2.circle(display_frame, (nose_x, nose_y), 5, (0, 0, 255), -1)
                                    
                                    # Track nose positions
                                    movement_positions.append((nose_x, nose_y))
                                    
                                    # Only analyze after collecting enough positions
                                    if len(movement_positions) > 10:
                                        # Use tuple unpacking to avoid slice issues
                                        # Get first few positions
                                        first_positions = movement_positions[:5]
                                        # Get recent positions
                                        recent_positions = movement_positions[-5:]
                                        
                                        # Calculate mean positions
                                        start_x = sum(p[0] for p in first_positions) / len(first_positions)
                                        start_y = sum(p[1] for p in first_positions) / len(first_positions)
                                        current_x = sum(p[0] for p in recent_positions) / len(recent_positions)
                                        current_y = sum(p[1] for p in recent_positions) / len(recent_positions)
                                        
                                        # Convert to integers for drawing
                                        start_point = (int(start_x), int(start_y))
                                        current_point = (int(current_x), int(current_y))
                                        
                                        # Draw movement vector if points are valid
                                        if (0 <= start_point[0] < display_frame.shape[1] and 
                                            0 <= start_point[1] < display_frame.shape[0] and
                                            0 <= current_point[0] < display_frame.shape[1] and
                                            0 <= current_point[1] < display_frame.shape[0]):
                                            cv2.arrowedLine(display_frame, 
                                                          start_point, 
                                                          current_point,
                                                          (255, 0, 0), 2)
                                        
                                        # Calculate differences
                                        x_diff = current_x - start_x
                                        y_diff = current_y - start_y
                                        
                                        # Minimum pixel movement required
                                        min_movement = 20  # Reduced for low frame rates
                                        
                                        # Display movement values for debugging
                                        cv2.putText(display_frame, f"Movement: x={x_diff:.1f}, y={y_diff:.1f}", 
                                                  (10, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                                        
                                        # Check if movement matches the required direction
                                        if movement_type == "left" and x_diff < -min_movement:
                                            movement_completed = True
                                        elif movement_type == "right" and x_diff > min_movement:
                                            movement_completed = True
                                        elif movement_type == "up" and y_diff < -min_movement:
                                            movement_completed = True
                                        elif movement_type == "down" and y_diff > min_movement:
                                            movement_completed = True
                                        
                                        if movement_completed:
                                            stage_passed["head_movement"] = True
                                            face_image = frame
                                            print(f"✓ Head movement {movement_type} detected! Liveness confirmed.")
                        except Exception as e:
                            # Just log the error and continue - don't crash
                            print(f"Error tracking nose position: {str(e)}")
                            # This allows the system to continue even if there's an issue with movement tracking
            
            # Draw stage information
            stages_info = [
                f"1. Texture Analysis: {'✓' if stage_passed['texture'] else '...'}",
                f"2. Blink Detection: {'✓' if stage_passed['blink'] else '...' if stage_passed['texture'] else 'Waiting'}",
                f"3. Head Movement ({movement_type}): {'✓' if stage_passed['head_movement'] else '...' if stage_passed['blink'] else 'Waiting'}"
            ]
            
            for i, info in enumerate(stages_info):
                cv2.putText(display_frame, info, 
                          (10, 280 + i*30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                          (0, 255, 0) if "✓" in info else (255, 255, 255), 1)
            
            # Display time remaining
            cv2.putText(display_frame, f"Time left: {int(time_left)}s", 
                      (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display current instruction
            instruction = ""
            if current_stage == "texture":
                instruction = f"Hold still while we analyze your face (Test: {texture_tests[current_texture_test_idx]})"
            elif current_stage == "blink":
                instruction = "Please blink naturally"
            elif current_stage == "head_movement":
                instruction = f"Please move your head {movement_type.upper()}"
                
            cv2.putText(display_frame, instruction, 
                      (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Check if all stages passed
            all_passed = all(stage_passed.values())
            if all_passed:
                cv2.putText(display_frame, "LIVENESS CONFIRMED!", 
                          (display_frame.shape[1]//2 - 150, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
            # Show the frame
            cv2.imshow("Liveness Detection", display_frame)
            
            # Check for keyboard input to toggle tests
            key = cv2.waitKey(1) & 0xFF
            if key == ord('t'):  # Toggle texture test
                skip_texture = not skip_texture
                if skip_texture:
                    stage_passed["texture"] = True
                    print("Texture test skipped")
                else:
                    stage_passed["texture"] = False
                    print("Texture test enabled")
            elif key == ord('b'):  # Toggle blink test
                skip_blink = not skip_blink
                if skip_blink:
                    stage_passed["blink"] = True
                    print("Blink test skipped")
                else:
                    stage_passed["blink"] = False
                    print("Blink test enabled")
            elif key == ord('m'):  # Toggle movement test
                skip_movement = not skip_movement
                if skip_movement:
                    stage_passed["head_movement"] = True
                    print("Movement test skipped")
                else:
                    stage_passed["head_movement"] = False
                    print("Movement test enabled")
            elif key == 27:  # ESC key
                break
            elif all_passed:
                # Wait a moment to show the success message
                time.sleep(2)
                break
                
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        
        # Return the results
        return all_passed, face_image
    
    def detect_liveness(self, timeout=10):
        """
        Legacy liveness detection method - now redirects to the more secure version
        """
        print("Using enhanced liveness detection for better security...")
        return self.detect_liveness_multi(timeout)
        
    def capture_face_with_liveness(self):
        """Capture a face image with liveness verification"""
        is_live, face_image = self.detect_liveness_multi()
        
        if not is_live:
            print("Liveness check failed. Please try again.")
            return None
            
        print("Liveness check passed!")
        
        # If we got a face during liveness detection, use it
        if face_image is not None:
            return face_image
            
        # Otherwise capture a new face
        return self.capture_face()

    def detect_face(self, frame):
        """
        Detect a face and its landmarks in the given frame.
        
        Args:
            frame: Input video frame
            
        Returns:
            tuple: (success, face_bbox, landmarks)
        """
        try:
            # Resize for faster detection
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Detect face locations
            face_locations = face_recognition.face_locations(rgb_small_frame)
            
            if not face_locations:
                return False, None, None
                
            # Get first face
            top, right, bottom, left = face_locations[0]
            
            # Scale back to full size
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2
            
            face_bbox = (left, top, right-left, bottom-top)
            
            # Get facial landmarks
            landmarks_list = face_recognition.face_landmarks(rgb_small_frame, [face_locations[0]])
            
            if not landmarks_list:
                return False, face_bbox, None
                
            # Convert face_recognition landmarks to dlib format (flat list of points)
            landmarks = []
            for feature in ['chin', 'left_eyebrow', 'right_eyebrow', 'nose_bridge', 'nose_tip', 
                           'left_eye', 'right_eye', 'top_lip', 'bottom_lip']:
                if feature in landmarks_list[0]:
                    for point in landmarks_list[0][feature]:
                        # Scale back to full size
                        landmarks.append((point[0] * 2, point[1] * 2))
            
            return True, face_bbox, landmarks
        except Exception as e:
            print(f"Error in face detection: {str(e)}")
            return False, None, None
            
    def _calculate_ear(self, eye_landmarks):
        """
        Calculate the Eye Aspect Ratio (EAR) from eye landmarks.
        
        Args:
            eye_landmarks: List of eye landmark points
            
        Returns:
            float: Eye Aspect Ratio
        """
        # Ensure landmarks are in correct format
        if not eye_landmarks or len(eye_landmarks) < 6:
            return 0.0
            
        try:
            # Convert to numpy array if not already
            points = np.array(eye_landmarks)
            
            # Calculate horizontal distance (eye width)
            horizontal_dist = np.linalg.norm(points[0] - points[3])
            
            # Calculate vertical distances
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            
            # Calculate EAR
            ear = (v1 + v2) / (2.0 * horizontal_dist)
            
            return ear
        except Exception as e:
            print(f"Error calculating EAR: {str(e)}")
            return 0.0 