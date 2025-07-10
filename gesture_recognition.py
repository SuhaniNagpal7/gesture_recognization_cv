import cv2
import numpy as np
import time
from utils.hand_landmarks import HandLandmarkDetector
from utils.gesture_detector import GestureDetector

class GestureRecognitionApp:
    """
    Main application for real-time gesture recognition
    """
    
    def __init__(self):
        self.landmark_detector = HandLandmarkDetector()
        self.gesture_detector = GestureDetector()
        self.cap = None
        self.running = False
        
        # Display settings
        self.display_width = 1280
        self.display_height = 720
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
    
    def start(self):
        """Start the gesture recognition application"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        print("Gesture Recognition Started!")
        print("Press 'q' to quit, 'h' for help")
        self._show_help()
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = self._process_frame(frame)
            
            # Display frame
            cv2.imshow('Gesture Recognition', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                self._show_help()
            elif key == ord('r'):
                self.gesture_detector.reset_history()
                print("Gesture history reset")
        
        self._cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for gesture recognition"""
        # Resize frame for display
        display_frame = cv2.resize(frame, (self.display_width, self.display_height))
        
        # Detect hands and landmarks
        processed_frame, hand_landmarks_list = self.landmark_detector.detect_hands(display_frame)
        
        # Process each detected hand
        for landmarks in hand_landmarks_list:
            # Detect gesture
            gesture, confidence, gesture_info = self.gesture_detector.detect_gesture(landmarks)
            
            if gesture:
                # Draw gesture information
                self._draw_gesture_info(processed_frame, gesture, confidence, gesture_info, landmarks)
        
        # Update FPS
        self._update_fps()
        
        # Draw FPS and instructions
        self._draw_overlay(processed_frame)
        
        return processed_frame
    
    def _draw_gesture_info(self, frame: np.ndarray, gesture: str, confidence: float, 
                          gesture_info: dict, landmarks: dict):
        """Draw gesture information on frame"""
        # Get hand position (wrist)
        wrist_pos = landmarks.get('wrist', (0, 0))
        
        # Draw gesture name
        cv2.putText(frame, f"Gesture: {gesture.upper()}", 
                   (wrist_pos[0] - 100, wrist_pos[1] - 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Draw confidence
        cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                   (wrist_pos[0] - 100, wrist_pos[1] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw action
        if 'action' in gesture_info:
            action = gesture_info['action']
            cv2.putText(frame, f"Action: {action}", 
                       (wrist_pos[0] - 100, wrist_pos[1] + 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw finger states
        if 'finger_states' in gesture_info:
            self._draw_finger_states(frame, gesture_info['finger_states'], wrist_pos)
    
    def _draw_finger_states(self, frame: np.ndarray, finger_states: dict, base_pos: tuple):
        """Draw finger state indicators"""
        y_offset = 50
        for i, (finger, state) in enumerate(finger_states.items()):
            color = (0, 255, 0) if state else (0, 0, 255)
            status = "✓" if state else "✗"
            cv2.putText(frame, f"{finger}: {status}", 
                       (base_pos[0] + 150, base_pos[1] + y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def _update_fps(self):
        """Update FPS calculation"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def _draw_overlay(self, frame: np.ndarray):
        """Draw FPS and instructions overlay"""
        # Draw FPS
        cv2.putText(frame, f"FPS: {self.current_fps}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw instructions
        instructions = [
            "Press 'q' to quit",
            "Press 'h' for help",
            "Press 'r' to reset gesture history"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, 
                       (10, frame.shape[0] - 80 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _show_help(self):
        """Show help information"""
        print("\n" + "="*50)
        print("GESTURE RECOGNITION HELP")
        print("="*50)
        print("Supported Gestures:")
        print("1. Pointing (index finger) - Drawing mode")
        print("2. Fist (closed hand) - Erase mode")
        print("3. Peace sign (index + middle) - Select mode")
        print("4. Thumbs up - Confirm action")
        print("5. Thumbs down - Cancel action")
        print("6. Open palm - Clear canvas")
        print("7. Three fingers - Zoom mode")
        print("8. OK sign - OK action")
        print("\nControls:")
        print("- 'q': Quit application")
        print("- 'h': Show this help")
        print("- 'r': Reset gesture history")
        print("="*50 + "\n")
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_detector.release()
        print("Application closed.")

def main():
    """Main function to run the gesture recognition application"""
    app = GestureRecognitionApp()
    try:
        app.start()
    except KeyboardInterrupt:
        print("\nApplication interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        app._cleanup()

if __name__ == "__main__":
    main() 