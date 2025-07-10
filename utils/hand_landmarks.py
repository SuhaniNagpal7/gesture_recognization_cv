import cv2
import mediapipe as mp
import numpy as np
from typing import List, Tuple, Optional

class HandLandmarkDetector:
    """
    Hand landmark detection using MediaPipe
    """
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.hands = self.mp_hands.Hands(
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        
        # Hand landmark indices
        self.LANDMARK_INDICES = {
            'wrist': 0,
            'thumb_tip': 4,
            'index_tip': 8,
            'middle_tip': 12,
            'ring_tip': 16,
            'pinky_tip': 20,
            'thumb_ip': 3,
            'index_pip': 6,
            'middle_pip': 10,
            'ring_pip': 14,
            'pinky_pip': 18
        }
    
    def detect_hands(self, image: np.ndarray) -> Tuple[np.ndarray, List]:
        """
        Detect hands in the image and return processed image with landmarks
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (processed_image, hand_landmarks_list)
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image
        results = self.hands.process(image_rgb)
        
        # Convert back to BGR for OpenCV
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        
        hand_landmarks_list = []
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_drawing.draw_landmarks(
                    image_bgr,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Extract landmark coordinates
                landmarks = self._extract_landmarks(hand_landmarks, image_bgr.shape)
                hand_landmarks_list.append(landmarks)
        
        return image_bgr, hand_landmarks_list
    
    def _extract_landmarks(self, hand_landmarks, image_shape: Tuple[int, int, int]) -> dict:
        """
        Extract landmark coordinates from MediaPipe hand landmarks
        
        Args:
            hand_landmarks: MediaPipe hand landmarks object
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            Dictionary containing landmark coordinates
        """
        height, width, _ = image_shape
        landmarks = {}
        
        for name, idx in self.LANDMARK_INDICES.items():
            landmark = hand_landmarks.landmark[idx]
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            z = landmark.z
            landmarks[name] = (x, y, z)
        
        return landmarks
    
    def get_finger_states(self, landmarks: dict) -> dict:
        """
        Determine which fingers are extended based on landmark positions
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Dictionary with finger states (True = extended, False = closed)
        """
        finger_states = {}
        
        # Thumb (compare tip to IP joint)
        thumb_tip = landmarks['thumb_tip']
        thumb_ip = landmarks['thumb_ip']
        finger_states['thumb'] = thumb_tip[1] < thumb_ip[1]
        
        # Index finger (compare tip to PIP joint)
        index_tip = landmarks['index_tip']
        index_pip = landmarks['index_pip']
        finger_states['index'] = index_tip[1] < index_pip[1]
        
        # Middle finger
        middle_tip = landmarks['middle_tip']
        middle_pip = landmarks['middle_pip']
        finger_states['middle'] = middle_tip[1] < middle_pip[1]
        
        # Ring finger
        ring_tip = landmarks['ring_tip']
        ring_pip = landmarks['ring_pip']
        finger_states['ring'] = ring_tip[1] < ring_pip[1]
        
        # Pinky finger
        pinky_tip = landmarks['pinky_tip']
        pinky_pip = landmarks['pinky_pip']
        finger_states['pinky'] = pinky_tip[1] < pinky_pip[1]
        
        return finger_states
    
    def calculate_gesture_features(self, landmarks: dict) -> np.ndarray:
        """
        Calculate features for gesture classification
        
        Args:
            landmarks: Dictionary of landmark coordinates
            
        Returns:
            Feature vector for gesture classification
        """
        features = []
        
        # Get finger states
        finger_states = self.get_finger_states(landmarks)
        features.extend([finger_states['thumb'], finger_states['index'], 
                       finger_states['middle'], finger_states['ring'], 
                       finger_states['pinky']])
        
        # Calculate distances between key points
        wrist = landmarks['wrist']
        index_tip = landmarks['index_tip']
        middle_tip = landmarks['middle_tip']
        
        # Distance from wrist to index tip
        dist_wrist_index = np.sqrt((wrist[0] - index_tip[0])**2 + (wrist[1] - index_tip[1])**2)
        features.append(dist_wrist_index)
        
        # Distance from wrist to middle tip
        dist_wrist_middle = np.sqrt((wrist[0] - middle_tip[0])**2 + (wrist[1] - middle_tip[1])**2)
        features.append(dist_wrist_middle)
        
        # Distance between index and middle tips
        dist_index_middle = np.sqrt((index_tip[0] - middle_tip[0])**2 + (index_tip[1] - middle_tip[1])**2)
        features.append(dist_index_middle)
        
        return np.array(features)
    
    def release(self):
        """Release resources"""
        self.hands.close() 