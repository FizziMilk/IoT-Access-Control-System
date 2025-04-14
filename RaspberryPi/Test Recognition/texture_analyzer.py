import cv2
import numpy as np
import time
import random
from collections import deque

class TextureAnalyzer:
    """
    Analyzes image textures for liveness detection to prevent spoofing.
    Uses multiple methods to distinguish real faces from photos or digital displays.
    """
    def __init__(self):
        """Initialize texture analyzer with default parameters"""
        pass
        
    def analyze_texture(self, face_image, debug=False):
        """
        Analyzes image texture to detect if an image is from a real face
        or a display/printout (anti-spoofing).
        
        Uses multiple texture analysis techniques:
        1. Laplacian variance (detail preservation)
        2. FFT analysis (high frequency components)
        3. Color variance analysis
        4. Gradient complexity
        5. LBP pattern uniformity
        6. Moiré pattern detection 
        7. Reflection detection
        
        Args:
            face_image: Image of face to analyze
            debug: Whether to return visualization
            
        Returns:
            tuple: (is_real, debug_image)
        """
        if face_image is None:
            return False, None
            
        # Resize for consistent analysis
        height, width = face_image.shape[:2]
        new_width, new_height = 300, 300
        if width > 0 and height > 0:
            resized = cv2.resize(face_image, (new_width, new_height))
        else:
            return False, None
            
        # Convert to grayscale
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Create debug image if requested
        if debug:
            debug_img = np.zeros((new_height*2 + 300, new_width*2, 3), dtype=np.uint8)
            # Place original image
            debug_img[0:new_height, 0:new_width] = resized
            cv2.putText(debug_img, "Original", (10, new_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ===================== TEST 1: TEXTURE DETAIL ANALYSIS =====================
        # Real faces have more texture detail which is preserved in the Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_variance = np.var(laplacian)
        
        # Screen displays often have less texture detail
        texture_threshold = 250.0  # Higher threshold for more strict checking
        texture_test_pass = laplacian_variance > texture_threshold
        
        # Visualize Laplacian if debugging
        if debug:
            lap_vis = cv2.normalize(laplacian, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            lap_color = cv2.applyColorMap(lap_vis, cv2.COLORMAP_JET)
            lap_display = cv2.resize(lap_color, (new_width, new_height))
            debug_img[0:new_height, new_width:new_width*2] = lap_display
            cv2.putText(debug_img, "Laplacian", (new_width + 10, new_height + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # ===================== TEST 2: FFT ANALYSIS =====================
        # Real faces have more high-frequency components
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
        
        # Get dimensions
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
            debug_img[new_height+50:new_height*2+50, 0:new_width] = fft_display
            cv2.putText(debug_img, "FFT (High Freq in Green)", (10, new_height*2 + 70), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Real faces have significant high-frequency components
        high_freq_threshold = 18.0  # Threshold for high frequency energy
        high_freq_test_pass = high_freq_energy > high_freq_threshold
        
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
        color_threshold = 45.0  # Threshold for color standard deviation
        corr_penalty = 10.0 * avg_color_corr  # Reduce score for high correlation
        adjusted_color_std = color_std_avg - corr_penalty
        color_test_pass = adjusted_color_std > color_threshold
        
        # ===================== TEST 4: GRADIENT ANALYSIS =====================
        # Calculate image gradients (1st order derivatives)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        # Calculate gradient complexity (variance of gradient directions)
        gradient_direction = np.arctan2(sobely, sobelx) * 180 / np.pi
        direction_variance = np.var(gradient_direction)
        
        # Digital displays often show more uniform gradient patterns
        gradient_threshold = 1200.0  # Threshold for gradient direction variance
        gradient_test_pass = direction_variance > gradient_threshold
        
        # ===================== TEST 5: LOCAL BINARY PATTERN ANALYSIS =====================
        # Calculate LBP and its histogram
        lbp = self._local_binary_pattern(gray)
        hist, _ = np.histogram(lbp, bins=256, range=(0, 256))
        hist = hist / np.sum(hist)  # Normalize
        
        # Calculate LBP uniformity (real faces have more uniform LBP distribution)
        lbp_uniformity = 1.0 - np.std(hist)
        
        # Digital displays often show less uniform LBP patterns
        lbp_threshold = 0.93  # Threshold for LBP uniformity
        lbp_test_pass = lbp_uniformity < lbp_threshold
        
        # ===================== TEST 6: MOIRÉ PATTERN DETECTION =====================
        moire_energy, filtered = self._detect_moire_patterns(gray)
        
        # Digital displays often show moiré patterns in this frequency band
        moire_threshold = 15.0  # Threshold for moiré pattern energy
        moire_test_pass = moire_energy < moire_threshold
        
        # ===================== TEST 7: REFLECTION DETECTION =====================
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
        reflection_threshold = 0.02  # Threshold for bright pixel ratio
        # Reduce the threshold if entropy is low (uniform brightness, typical of screens)
        if brightness_entropy < 3.0:
            reflection_threshold *= 0.8
        reflection_test_pass = bright_pixels < reflection_threshold
        
        # ===================== COMBINE RESULTS =====================
        # Visualize test results
        if debug:
            result_img = np.zeros((200, new_width * 2, 3), dtype=np.uint8)
            texts = [
                f"1. Texture Detail: {'PASS' if texture_test_pass else 'FAIL'} ({laplacian_variance:.1f} > {texture_threshold})",
                f"2. High Frequency: {'PASS' if high_freq_test_pass else 'FAIL'} ({high_freq_energy:.1f} > {high_freq_threshold})",
                f"3. Color Variance: {'PASS' if color_test_pass else 'FAIL'} ({adjusted_color_std:.1f} > {color_threshold})",
                f"4. Gradient Complexity: {'PASS' if gradient_test_pass else 'FAIL'} ({direction_variance:.1f} > {gradient_threshold})",
                f"5. LBP Uniformity: {'PASS' if lbp_test_pass else 'FAIL'} ({lbp_uniformity:.3f} < {lbp_threshold})",
                f"6. Moiré Pattern: {'PASS' if moire_test_pass else 'FAIL'} ({moire_energy:.1f} < {moire_threshold})",
                f"7. Reflection: {'PASS' if reflection_test_pass else 'FAIL'} ({bright_pixels*100:.2f}% < {reflection_threshold*100}%)"
            ]
            
            for i, text in enumerate(texts):
                color = (0, 255, 0) if "PASS" in text else (0, 0, 255)
                cv2.putText(result_img, text, (10, 25 + i * 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
            # Place in debug image
            debug_img[new_height*2+50:new_height*2+250, 0:new_width*2] = result_img
            
            # Add filtered images for moiré pattern and reflection detection
            if new_width > 0 and new_height > 0:
                moire_display = cv2.applyColorMap(filtered, cv2.COLORMAP_JET)
                moire_display = cv2.resize(moire_display, (new_width, new_height))
                debug_img[new_height+50:new_height*2+50, new_width:new_width*2] = moire_display
                cv2.putText(debug_img, "Moiré Pattern Filter", (new_width + 10, new_height*2 + 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Count how many tests pass
        passing_tests = sum([
            texture_test_pass, 
            high_freq_test_pass, 
            color_test_pass, 
            gradient_test_pass,
            lbp_test_pass,
            moire_test_pass,
            reflection_test_pass
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
        
        return is_real, debug_img if debug else None
        
    def _local_binary_pattern(self, image):
        """
        Calculate simple LBP (Local Binary Pattern).
        
        Args:
            image: Grayscale image
            
        Returns:
            numpy.ndarray: LBP image
        """
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
        
    def _detect_moire_patterns(self, gray):
        """
        Apply bandpass filter to detect moiré patterns in screens.
        
        Args:
            gray: Grayscale image
            
        Returns:
            tuple: (moiré_energy, filtered_image)
        """
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
        
        return moire_energy, filtered 