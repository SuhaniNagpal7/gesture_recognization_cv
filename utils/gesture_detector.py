import numpy as np
from typing import Dict, List, Tuple, Optional
from .hand_landmarks import HandLandmarkDetector

class GestureDetector:
    """
    Hand gesture recognition based on finger positions and patterns
    """
    
    def __init__(self):
        self.landmark_detector = HandLandmarkDetector()
        
        # Define gesture patterns
        self.GESTURE_PATTERNS = {
            'pointing': {
                'fingers': {'thumb': False, 'index': True, 'middle': False, 'ring': False, 'pinky': False},
                'description': 'Index finger pointing',
                'action': 'draw'
            },
            'fist': {
                'fingers': {'thumb': False, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
                'description': 'Closed fist',
                'action': 'erase'
            },
            'peace': {
                'fingers': {'thumb': False, 'index': True, 'middle': True, 'ring': False, 'pinky': False},
                'description': 'Peace sign (index and middle finger)',
                'action': 'select'
            },
            'thumbs_up': {
                'fingers': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
                'description': 'Thumbs up',
                'action': 'confirm'
            },
            'thumbs_down': {
                'fingers': {'thumb': True, 'index': False, 'middle': False, 'ring': False, 'pinky': False},
                'description': 'Thumbs down',
                'action': 'cancel'
            },
            'open_palm': {
                'fingers': {'thumb': True, 'index': True, 'middle': True, 'ring': True, 'pinky': True},
                'description': 'Open palm (all fingers extended)',
                'action': 'clear'
            },
            'three_fingers': {
                'fingers': {'thumb': False, 'index': True, 'middle': True, 'ring': True, 'pinky': False},
                'description': 'Three fingers (index, middle, ring)',
                'action': 'zoom'
            },
            'ok_sign': {
                'fingers': {'thumb': True, 'index': True, 'middle': False, 'ring': False, 'pinky': False},
                'description': 'OK sign (thumb and index forming circle)',
                'action': 'ok'
            }
        }
        
        # Gesture confidence thresholds
        self.CONFIDENCE_THRESHOLD = 0.8
        self.GESTURE_HISTORY_SIZE = 5
        self.gesture_history = []
    
    def detect_gesture(self, landmarks: Dict) -> Tuple[str, float, Dict]:
        """
        Detect gesture from hand landmarks
        
        Args:
            landmarks: Dictionary of hand landmark coordinates
            
        Returns:
            Tuple of (gesture_name, confidence, gesture_info)
        """
        if not landmarks:
            return None, 0.0, {}
        
        # Get finger states
        finger_states = self.landmark_detector.get_finger_states(landmarks)
        
        # Calculate gesture features
        features = self.landmark_detector.calculate_gesture_features(landmarks)
        
        # Match against known patterns
        best_match = None
        best_confidence = 0.0
        
        for gesture_name, pattern in self.GESTURE_PATTERNS.items():
            confidence = self._calculate_gesture_confidence(finger_states, pattern['fingers'])
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = gesture_name
        
        # Apply confidence threshold
        if best_confidence >= self.CONFIDENCE_THRESHOLD:
            gesture_info = self.GESTURE_PATTERNS[best_match].copy()
            gesture_info['confidence'] = best_confidence
            gesture_info['finger_states'] = finger_states
            gesture_info['features'] = features
            
            # Update gesture history
            self._update_gesture_history(best_match)
            
            return best_match, best_confidence, gesture_info
        
        return None, 0.0, {}
    
    def _calculate_gesture_confidence(self, finger_states: Dict, pattern: Dict) -> float:
        """
        Calculate confidence score for a gesture pattern match
        
        Args:
            finger_states: Current finger states
            pattern: Expected finger pattern
            
        Returns:
            Confidence score (0.0 to 1.0)
        """
        matches = 0
        total_fingers = len(pattern)
        
        for finger, expected_state in pattern.items():
            if finger in finger_states and finger_states[finger] == expected_state:
                matches += 1
        
        return matches / total_fingers
    
    def _update_gesture_history(self, gesture: str):
        """Update gesture history for stability"""
        self.gesture_history.append(gesture)
        if len(self.gesture_history) > self.GESTURE_HISTORY_SIZE:
            self.gesture_history.pop(0)
    
    def get_stable_gesture(self) -> Optional[str]:
        """
        Get the most stable gesture from history
        
        Returns:
            Most frequent gesture in history, or None if no stable gesture
        """
        if not self.gesture_history:
            return None
        
        # Count gesture occurrences
        gesture_counts = {}
        for gesture in self.gesture_history:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        # Find most frequent gesture
        most_frequent = max(gesture_counts.items(), key=lambda x: x[1])
        
        # Return gesture if it appears in majority of recent frames
        if most_frequent[1] >= len(self.gesture_history) * 0.6:
            return most_frequent[0]
        
        return None
    
    def get_gesture_action(self, gesture_name: str) -> Optional[str]:
        """
        Get the action associated with a gesture
        
        Args:
            gesture_name: Name of the gesture
            
        Returns:
            Associated action, or None if gesture not found
        """
        if gesture_name in self.GESTURE_PATTERNS:
            return self.GESTURE_PATTERNS[gesture_name]['action']
        return None
    
    def add_custom_gesture(self, name: str, finger_pattern: Dict, action: str, description: str = ""):
        """
        Add a custom gesture pattern
        
        Args:
            name: Name of the gesture
            finger_pattern: Dictionary of finger states
            action: Action associated with the gesture
            description: Description of the gesture
        """
        self.GESTURE_PATTERNS[name] = {
            'fingers': finger_pattern,
            'action': action,
            'description': description
        }
    
    def get_available_gestures(self) -> List[str]:
        """Get list of available gesture names"""
        return list(self.GESTURE_PATTERNS.keys())
    
    def get_gesture_info(self, gesture_name: str) -> Optional[Dict]:
        """
        Get information about a specific gesture
        
        Args:
            gesture_name: Name of the gesture
            
        Returns:
            Gesture information dictionary, or None if not found
        """
        return self.GESTURE_PATTERNS.get(gesture_name)
    
    def reset_history(self):
        """Reset gesture history"""
        self.gesture_history.clear() 