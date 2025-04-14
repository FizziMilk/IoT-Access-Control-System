import cv2
import numpy as np
import time
import scipy.signal
from collections import deque

class BlinkDetector:
    """
    Enhanced blink detection with movement analysis to reject spoofing attempts.
    """
    def __init__(self, camera):
        """
        Initialize with a camera object
        
        Args:
            camera: Camera object that provides face detection and EAR calculation
        """
        self.camera = camera
        self.eye_history = []
    
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
            face_results = self.camera.detect_face(frame)
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
        left_ear = self.camera._calculate_ear(left_eye_landmarks)
        right_ear = self.camera._calculate_ear(right_eye_landmarks)
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
        movement_values, contour_changes, avg_movement, max_movement, movement_std, avg_movement_vector = \
            self._analyze_movement()
        
        debug_data["movement_values"] = movement_values if movement_values else [0]
        
        # =============== BLINK DETECTION LOGIC ===============
        ear_dip_detected, ear_trajectory_score, ear_recovery = self._analyze_ear_trajectory(
            recent_ear_values, mean_ear, min_ear)
        
        # =============== MOVEMENT REJECTION LOGIC ===============
        movement_rejection_score = self._calculate_movement_rejection(
            movement_values, recent_ear_values, contour_changes, avg_movement_vector)
        
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
            vis_frame = self._visualize_results(vis_frame, face_bbox, left_eye_landmarks, 
                                           right_eye_landmarks, ear, avg_movement, 
                                           blink_detected, blink_score, timestamps, 
                                           ear_values, movement_values)
        
        return blink_detected, vis_frame, debug_data
    
    def _analyze_movement(self):
        """Analyze movement patterns between frames"""
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
            max_movement = max(movement_values)
            movement_std = np.std(movement_values) if len(movement_values) > 1 else 0
        else:
            avg_movement = 0
            max_movement = 0
            movement_std = 0
        
        return movement_values, contour_changes, avg_movement, max_movement, movement_std, avg_movement_vector
    
    def _analyze_ear_trajectory(self, recent_ear_values, mean_ear, min_ear):
        """Analyze EAR trajectory for blink-like pattern"""
        # Check for significant EAR dip
        ear_threshold = 0.2  # Threshold for eye closure
        ear_dip_detected = min_ear < mean_ear - 0.05 and min_ear < ear_threshold
        
        # Analyze EAR trajectory for blink-like pattern
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
        
        return ear_dip_detected, ear_trajectory_score, ear_recovery

    def _calculate_movement_rejection(self, movement_values, recent_ear_values, contour_changes, avg_movement_vector):
        """Calculate movement rejection score for spoofing detection"""
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
            if vertical_ratio > 1.5 and max(movement_values) > 3.0:  # Primarily vertical movement
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
        
        return movement_rejection_score
    
    def _visualize_results(self, frame, face_bbox, left_eye_landmarks, right_eye_landmarks, 
                         ear, avg_movement, blink_detected, blink_score, 
                         timestamps, ear_values, movement_values):
        """Visualize blink detection results on the frame"""
        # Draw eye contours
        for eye_points in [left_eye_landmarks, right_eye_landmarks]:
            pts = np.array(eye_points, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (0, 255, 255), 1)
        
        # Display current EAR
        ear_text = f"EAR: {ear:.2f}"
        cv2.putText(frame, ear_text, (face_bbox[0], face_bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display movement information
        movement_text = f"Movement: {avg_movement:.1f}"
        cv2.putText(frame, movement_text, (face_bbox[0], face_bbox[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Display blink detection result
        result_color = (0, 255, 0) if blink_detected else (0, 0, 255)
        result_text = "BLINK DETECTED" if blink_detected else "No Blink"
        score_text = f"Score: {blink_score}"
        
        cv2.putText(frame, result_text, (face_bbox[0] + face_bbox[2] - 160, face_bbox[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, result_color, 2)
        cv2.putText(frame, score_text, (face_bbox[0] + face_bbox[2] - 80, face_bbox[1] - 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add plots of EAR and movement if there's enough history
        if len(ear_values) > 3:
            self._add_tracking_plots(frame, timestamps, ear_values, movement_values, blink_detected)
            
        return frame
    
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