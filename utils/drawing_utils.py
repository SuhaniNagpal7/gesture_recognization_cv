import cv2
import numpy as np
from typing import Tuple, List, Optional
import json
import os

class DrawingCanvas:
    """
    Interactive drawing canvas with gesture-based controls
    """
    
    def __init__(self, width: int = 800, height: int = 600):
        self.width = width
        self.height = height
        self.canvas = np.ones((height, width, 3), dtype=np.uint8) * 255  # White background
        self.drawing = False
        self.last_point = None
        self.current_color = (0, 0, 0)  # Black
        self.current_thickness = 3
        self.eraser_mode = False
        self.undo_stack = []
        self.redo_stack = []
        self.max_undo_steps = 50
        
        # Drawing settings
        self.colors = {
            'black': (0, 0, 0),
            'red': (0, 0, 255),
            'green': (0, 255, 0),
            'blue': (255, 0, 0),
            'yellow': (0, 255, 255),
            'purple': (255, 0, 255),
            'orange': (0, 165, 255)
        }
        
        # Save initial state
        self._save_state()
    
    def clear_canvas(self):
        """Clear the entire canvas"""
        self.canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
        self._save_state()
    
    def set_color(self, color_name: str):
        """Set drawing color"""
        if color_name in self.colors:
            self.current_color = self.colors[color_name]
    
    def set_thickness(self, thickness: int):
        """Set line thickness"""
        self.current_thickness = max(1, min(20, thickness))
    
    def toggle_eraser(self):
        """Toggle eraser mode"""
        self.eraser_mode = not self.eraser_mode
        if self.eraser_mode:
            self.current_color = (255, 255, 255)  # White for eraser
        else:
            self.current_color = (0, 0, 0)  # Black
    
    def start_drawing(self, point: Tuple[int, int]):
        """Start drawing at the given point"""
        self.drawing = True
        self.last_point = point
    
    def draw(self, point: Tuple[int, int]):
        """Draw line from last point to current point"""
        if self.drawing and self.last_point:
            cv2.line(self.canvas, self.last_point, point, self.current_color, self.current_thickness)
            self.last_point = point
    
    def stop_drawing(self):
        """Stop drawing and save state"""
        self.drawing = False
        self.last_point = None
        self._save_state()
    
    def _save_state(self):
        """Save current canvas state for undo"""
        if len(self.undo_stack) >= self.max_undo_steps:
            self.undo_stack.pop(0)
        self.undo_stack.append(self.canvas.copy())
        self.redo_stack.clear()  # Clear redo stack when new action is performed
    
    def undo(self):
        """Undo last drawing action"""
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.canvas = self.undo_stack[-1].copy()
    
    def redo(self):
        """Redo last undone action"""
        if self.redo_stack:
            self.undo_stack.append(self.redo_stack.pop())
            self.canvas = self.undo_stack[-1].copy()
    
    def get_canvas(self) -> np.ndarray:
        """Get current canvas image"""
        return self.canvas.copy()
    
    def save_drawing(self, filename: str):
        """Save drawing to file"""
        cv2.imwrite(filename, self.canvas)
    
    def load_drawing(self, filename: str):
        """Load drawing from file"""
        if os.path.exists(filename):
            loaded_canvas = cv2.imread(filename)
            if loaded_canvas is not None:
                # Resize to fit canvas
                loaded_canvas = cv2.resize(loaded_canvas, (self.width, self.height))
                self.canvas = loaded_canvas
                self._save_state()

class GestureDrawingController:
    """
    Controller that maps gestures to drawing actions
    """
    
    def __init__(self, canvas: DrawingCanvas):
        self.canvas = canvas
        self.last_gesture = None
        self.gesture_cooldown = 0
        self.COOLDOWN_FRAMES = 10
        
        # Gesture to action mapping
        self.gesture_actions = {
            'pointing': self._handle_pointing,
            'fist': self._handle_fist,
            'peace': self._handle_peace,
            'thumbs_up': self._handle_thumbs_up,
            'thumbs_down': self._handle_thumbs_down,
            'open_palm': self._handle_open_palm,
            'three_fingers': self._handle_three_fingers
        }
    
    def process_gesture(self, gesture: str, hand_position: Tuple[int, int]):
        """
        Process gesture and perform corresponding drawing action
        
        Args:
            gesture: Detected gesture name
            hand_position: Position of hand in image coordinates
        """
        if self.gesture_cooldown > 0:
            self.gesture_cooldown -= 1
            return
        
        if gesture != self.last_gesture:
            self.last_gesture = gesture
            
            if gesture in self.gesture_actions:
                self.gesture_actions[gesture](hand_position)
                self.gesture_cooldown = self.COOLDOWN_FRAMES
    
    def _handle_pointing(self, hand_position: Tuple[int, int]):
        """Handle pointing gesture - start/continue drawing"""
        if not self.canvas.drawing:
            self.canvas.start_drawing(hand_position)
        else:
            self.canvas.draw(hand_position)
    
    def _handle_fist(self, hand_position: Tuple[int, int]):
        """Handle fist gesture - toggle eraser"""
        self.canvas.toggle_eraser()
        print("Eraser mode toggled")
    
    def _handle_peace(self, hand_position: Tuple[int, int]):
        """Handle peace sign - select color"""
        # Cycle through colors
        color_names = list(self.canvas.colors.keys())
        current_color_name = None
        for name, color in self.canvas.colors.items():
            if color == self.canvas.current_color:
                current_color_name = name
                break
        
        if current_color_name:
            current_index = color_names.index(current_color_name)
            next_index = (current_index + 1) % len(color_names)
            self.canvas.set_color(color_names[next_index])
            print(f"Color changed to: {color_names[next_index]}")
    
    def _handle_thumbs_up(self, hand_position: Tuple[int, int]):
        """Handle thumbs up - confirm/save"""
        self.canvas.save_drawing("drawing_saved.png")
        print("Drawing saved!")
    
    def _handle_thumbs_down(self, hand_position: Tuple[int, int]):
        """Handle thumbs down - cancel/clear"""
        self.canvas.clear_canvas()
        print("Canvas cleared!")
    
    def _handle_open_palm(self, hand_position: Tuple[int, int]):
        """Handle open palm - undo"""
        self.canvas.undo()
        print("Undo performed")
    
    def _handle_three_fingers(self, hand_position: Tuple[int, int]):
        """Handle three fingers - redo"""
        self.canvas.redo()
        print("Redo performed")
    
    def stop_drawing(self):
        """Stop current drawing action"""
        self.canvas.stop_drawing()
        self.last_gesture = None

class DrawingVisualizer:
    """
    Visualizer for drawing application with gesture feedback
    """
    
    def __init__(self, canvas_width: int, canvas_height: int):
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.info_panel_width = 300
        self.total_width = canvas_width + self.info_panel_width
        
    def create_display_frame(self, canvas_image: np.ndarray, current_gesture: str = None, 
                           gesture_info: dict = None) -> np.ndarray:
        """
        Create display frame with canvas and information panel
        
        Args:
            canvas_image: Canvas image
            current_gesture: Current detected gesture
            gesture_info: Additional gesture information
            
        Returns:
            Combined display frame
        """
        # Create info panel
        info_panel = np.ones((self.canvas_height, self.info_panel_width, 3), dtype=np.uint8) * 240
        
        # Add gesture information
        if current_gesture and gesture_info:
            self._add_gesture_info(info_panel, current_gesture, gesture_info)
        
        # Add instructions
        self._add_instructions(info_panel)
        
        # Combine canvas and info panel
        display_frame = np.hstack([canvas_image, info_panel])
        
        return display_frame
    
    def _add_gesture_info(self, info_panel: np.ndarray, gesture: str, gesture_info: dict):
        """Add gesture information to info panel"""
        y_offset = 30
        
        # Gesture name
        cv2.putText(info_panel, f"Gesture: {gesture.upper()}", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Confidence
        if 'confidence' in gesture_info:
            confidence = gesture_info['confidence']
            cv2.putText(info_panel, f"Confidence: {confidence:.2f}", 
                       (10, y_offset + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Action
        if 'action' in gesture_info:
            action = gesture_info['action']
            cv2.putText(info_panel, f"Action: {action}", 
                       (10, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        # Finger states
        if 'finger_states' in gesture_info:
            y_offset += 100
            cv2.putText(info_panel, "Finger States:", 
                       (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            
            finger_states = gesture_info['finger_states']
            for i, (finger, state) in enumerate(finger_states.items()):
                color = (0, 255, 0) if state else (0, 0, 255)
                status = "Extended" if state else "Closed"
                cv2.putText(info_panel, f"{finger}: {status}", 
                           (10, y_offset + 25 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    def _add_instructions(self, info_panel: np.ndarray):
        """Add drawing instructions to info panel"""
        y_offset = 300
        
        cv2.putText(info_panel, "GESTURE CONTROLS:", 
                   (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        instructions = [
            "Pointing: Draw",
            "Fist: Eraser",
            "Peace: Change Color",
            "Thumbs Up: Save",
            "Thumbs Down: Clear",
            "Open Palm: Undo",
            "Three Fingers: Redo"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(info_panel, instruction, 
                       (10, y_offset + 30 + i * 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1) 