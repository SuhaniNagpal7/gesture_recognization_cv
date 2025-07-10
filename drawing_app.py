import cv2
import numpy as np
import time
from utils.hand_landmarks import HandLandmarkDetector
from utils.gesture_detector import GestureDetector
from utils.drawing_utils import DrawingCanvas, GestureDrawingController, DrawingVisualizer

class InteractiveDrawingApp:
    """
    Interactive drawing application controlled by hand gestures
    """
    
    def __init__(self):
        self.landmark_detector = HandLandmarkDetector()
        self.gesture_detector = GestureDetector()
        
        # Canvas settings
        self.canvas_width = 800
        self.canvas_height = 600
        
        # Initialize drawing components
        self.canvas = DrawingCanvas(self.canvas_width, self.canvas_height)
        self.drawing_controller = GestureDrawingController(self.canvas)
        self.visualizer = DrawingVisualizer(self.canvas_width, self.canvas_height)
        
        # Camera settings
        self.cap = None
        self.running = False
        
        # Display settings
        self.display_width = 1280
        self.display_height = 720
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Gesture tracking
        self.current_gesture = None
        self.gesture_info = {}
        self.hand_position = None
    
    def start(self):
        """Start the interactive drawing application"""
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        self.running = True
        print("Interactive Drawing App Started!")
        print("Use gestures to control drawing:")
        print("- Pointing: Draw")
        print("- Fist: Eraser")
        print("- Peace: Change color")
        print("- Thumbs up: Save")
        print("- Thumbs down: Clear")
        print("- Open palm: Undo")
        print("- Three fingers: Redo")
        print("Press 'q' to quit")
        
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Could not read frame")
                break
            
            # Process frame
            processed_frame = self._process_frame(frame)
            
            # Create display frame
            display_frame = self._create_display_frame(processed_frame)
            
            # Display frame
            cv2.imshow('Interactive Drawing App', display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas.clear_canvas()
                print("Canvas cleared!")
            elif key == ord('s'):
                self.canvas.save_drawing("drawing_saved.png")
                print("Drawing saved!")
            elif key == ord('z'):
                self.canvas.undo()
                print("Undo performed")
            elif key == ord('y'):
                self.canvas.redo()
                print("Redo performed")
        
        self._cleanup()
    
    def _process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Process a single frame for gesture recognition"""
        # Resize frame for display
        display_frame = cv2.resize(frame, (self.display_width, self.display_height))
        
        # Detect hands and landmarks
        processed_frame, hand_landmarks_list = self.landmark_detector.detect_hands(display_frame)
        
        # Process each detected hand
        for landmarks in hand_landmarks_list:
            # Get hand position (wrist)
            self.hand_position = landmarks.get('wrist', (0, 0))
            
            # Detect gesture
            gesture, confidence, gesture_info = self.gesture_detector.detect_gesture(landmarks)
            
            if gesture and confidence > 0.8:
                self.current_gesture = gesture
                self.gesture_info = gesture_info
                
                # Process gesture for drawing
                self.drawing_controller.process_gesture(gesture, self.hand_position)
                
                # Draw gesture information
                self._draw_gesture_info(processed_frame, gesture, confidence, gesture_info)
            else:
                # Stop drawing if no gesture detected
                self.drawing_controller.stop_drawing()
                self.current_gesture = None
        
        # Update FPS
        self._update_fps()
        
        return processed_frame
    
    def _create_display_frame(self, camera_frame: np.ndarray) -> np.ndarray:
        """Create the main display frame with camera and canvas"""
        # Get canvas image
        canvas_image = self.canvas.get_canvas()
        
        # Create display frame with gesture info
        display_frame = self.visualizer.create_display_frame(
            canvas_image, self.current_gesture, self.gesture_info
        )
        
        # Resize camera frame to fit in display
        camera_height = int(display_frame.shape[0] * 0.4)
        camera_width = int(camera_height * 4/3)  # 4:3 aspect ratio
        camera_frame_resized = cv2.resize(camera_frame, (camera_width, camera_height))
        
        # Create combined frame
        combined_height = display_frame.shape[0] + camera_height
        combined_width = max(display_frame.shape[1], camera_width)
        
        combined_frame = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 50
        
        # Place canvas on top
        combined_frame[:display_frame.shape[0], :display_frame.shape[1]] = display_frame
        
        # Place camera frame below
        y_offset = display_frame.shape[0]
        x_offset = (combined_width - camera_width) // 2
        combined_frame[y_offset:y_offset+camera_height, x_offset:x_offset+camera_width] = camera_frame_resized
        
        # Add separator line
        cv2.line(combined_frame, (0, display_frame.shape[0]), 
                (combined_width, display_frame.shape[0]), (255, 255, 255), 2)
        
        # Add FPS and instructions
        self._draw_overlay(combined_frame)
        
        return combined_frame
    
    def _draw_gesture_info(self, frame: np.ndarray, gesture: str, confidence: float, gesture_info: dict):
        """Draw gesture information on camera frame"""
        if self.hand_position:
            # Draw gesture name
            cv2.putText(frame, f"Gesture: {gesture.upper()}", 
                       (self.hand_position[0] - 100, self.hand_position[1] - 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Draw confidence
            cv2.putText(frame, f"Confidence: {confidence:.2f}", 
                       (self.hand_position[0] - 100, self.hand_position[1] - 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Draw action
            if 'action' in gesture_info:
                action = gesture_info['action']
                cv2.putText(frame, f"Action: {action}", 
                           (self.hand_position[0] - 100, self.hand_position[1] + 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    
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
        
        # Draw current drawing mode
        mode_text = "Eraser" if self.canvas.eraser_mode else "Draw"
        cv2.putText(frame, f"Mode: {mode_text}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # Draw current color
        color_name = "Unknown"
        for name, color in self.canvas.colors.items():
            if color == self.canvas.current_color:
                color_name = name
                break
        cv2.putText(frame, f"Color: {color_name}", 
                   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw instructions
        instructions = [
            "Controls:",
            "q - Quit",
            "c - Clear canvas",
            "s - Save drawing",
            "z - Undo",
            "y - Redo"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, 
                       (10, frame.shape[0] - 120 + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def _cleanup(self):
        """Clean up resources"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        self.landmark_detector.release()
        print("Application closed.")

def main():
    """Main function to run the interactive drawing application"""
    app = InteractiveDrawingApp()
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