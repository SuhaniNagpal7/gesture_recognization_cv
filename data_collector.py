import cv2
import numpy as np
import json
import os
import time
from datetime import datetime
from utils.hand_landmarks import HandLandmarkDetector
from utils.gesture_detector import GestureDetector

class GestureDataCollector:
    """
    Tool for collecting gesture data for training custom models
    """
    
    def __init__(self):
        self.landmark_detector = HandLandmarkDetector()
        self.gesture_detector = GestureDetector()
        self.cap = None
        self.running = False
        
        # Data collection settings
        self.data_dir = "data/raw"
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Collection state
        self.current_gesture = None
        self.samples_collected = 0
        self.target_samples = 100
        self.collection_active = False
        
        # Data storage
        self.collected_data = []
    
    def start_collection(self, gesture_name: str, samples: int = 100):
        """
        Start data collection for a specific gesture
        
        Args:
            gesture_name: Name of the gesture to collect
            samples: Number of samples to collect
        """
        self.current_gesture = gesture_name
        self.target_samples = samples
        self.samples_collected = 0
        self.collected_data = []
        
        print(f"Starting data collection for gesture: {gesture_name}")
        print(f"Target samples: {samples}")
        print("Press 's' to start/stop collection, 'q' to quit")
        
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = self._process_frame(frame)
            
            # Display frame
            cv2.imshow('Gesture Data Collection', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.collection_active = not self.collection_active
                if self.collection_active:
                    print("Collection started!")
                else:
                    print("Collection paused!")
            elif key == ord('r'):
                self._reset_collection()
                print("Collection reset!")
        
        self._cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process frame for data collection"""
        # Resize frame
        display_frame = cv2.resize(frame, (800, 600))
        
        # Detect hands and landmarks
        processed_frame, hand_landmarks_list = self.landmark_detector.detect_hands(display_frame)
        
        # Process each detected hand
        for landmarks in hand_landmarks_list:
            # Calculate features
            features = self.landmark_detector.calculate_gesture_features(landmarks)
            finger_states = self.landmark_detector.get_finger_states(landmarks)
            
            # Collect data if active
            if self.collection_active and self.samples_collected < self.target_samples:
                self._collect_sample(landmarks, features, finger_states)
            
            # Draw hand landmarks
            self._draw_landmarks(processed_frame, landmarks)
        
        # Draw collection info
        self._draw_collection_info(processed_frame)
        
        return processed_frame
    
    def _collect_sample(self, landmarks: dict, features: np.ndarray, finger_states: dict):
        """Collect a single sample"""
        sample = {
            'gesture': self.current_gesture,
            'timestamp': time.time(),
            'features': features.tolist(),
            'finger_states': finger_states,
            'landmarks': landmarks
        }
        
        self.collected_data.append(sample)
        self.samples_collected += 1
        
        print(f"Collected sample {self.samples_collected}/{self.target_samples}")
        
        # Save data if collection is complete
        if self.samples_collected >= self.target_samples:
            self._save_data()
            self.collection_active = False
            print("Data collection completed!")
    
    def _draw_landmarks(self, frame: np.ndarray, landmarks: dict):
        """Draw hand landmarks on frame"""
        # Draw key landmarks
        key_points = ['wrist', 'thumb_tip', 'index_tip', 'middle_tip', 'ring_tip', 'pinky_tip']
        
        for point_name in key_points:
            if point_name in landmarks:
                point = landmarks[point_name]
                cv2.circle(frame, (point[0], point[1]), 5, (0, 255, 0), -1)
                cv2.putText(frame, point_name, (point[0] + 10, point[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def _draw_collection_info(self, frame: np.ndarray):
        """Draw collection information on frame"""
        # Background rectangle
        cv2.rectangle(frame, (10, 10), (400, 150), (0, 0, 0), -1)
        cv2.rectangle(frame, (10, 10), (400, 150), (255, 255, 255), 2)
        
        # Collection status
        status_color = (0, 255, 0) if self.collection_active else (0, 0, 255)
        status_text = "ACTIVE" if self.collection_active else "PAUSED"
        cv2.putText(frame, f"Collection: {status_text}", 
                   (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # Gesture info
        cv2.putText(frame, f"Gesture: {self.current_gesture}", 
                   (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress
        progress = self.samples_collected / self.target_samples
        cv2.putText(frame, f"Progress: {self.samples_collected}/{self.target_samples}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Progress bar
        bar_width = 300
        bar_height = 20
        bar_x, bar_y = 20, 120
        
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), 
                     (100, 100, 100), -1)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + int(bar_width * progress), bar_y + bar_height), 
                     (0, 255, 0), -1)
        
        # Instructions
        instructions = [
            "Press 's' to start/stop collection",
            "Press 'r' to reset",
            "Press 'q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, 
                       (10, frame.shape[0] - 80 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _save_data(self):
        """Save collected data to file"""
        if not self.collected_data:
            return
        
        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.data_dir}/{self.current_gesture}_{timestamp}.json"
        
        # Save data
        with open(filename, 'w') as f:
            json.dump(self.collected_data, f, indent=2)
        
        print(f"Data saved to: {filename}")
        print(f"Collected {len(self.collected_data)} samples")
    
    def _reset_collection(self):
        """Reset collection state"""
        self.samples_collected = 0
        self.collected_data = []
        self.collection_active = False
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_detector.release()
        print("Data collection finished.")

def main():
    """Main function for data collection"""
    collector = GestureDataCollector()
    
    print("Gesture Data Collection Tool")
    print("=" * 40)
    
    # Get gesture name
    gesture_name = input("Enter gesture name: ").strip()
    if not gesture_name:
        print("Invalid gesture name")
        return
    
    # Get number of samples
    try:
        samples = int(input("Enter number of samples (default 100): ") or "100")
    except ValueError:
        samples = 100
    
    print(f"\nStarting data collection for '{gesture_name}' with {samples} samples")
    print("Make sure your hand is visible in the camera")
    print("Press 's' to start/stop collection")
    
    try:
        collector.start_collection(gesture_name, samples)
    except KeyboardInterrupt:
        print("\nCollection interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        collector._cleanup()

if __name__ == "__main__":
    main() 