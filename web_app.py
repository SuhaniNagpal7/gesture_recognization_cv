import streamlit as st
import cv2
import numpy as np
import time
import json
import os
from PIL import Image
import plotly.express as px
import plotly.graph_objects as go
from utils.hand_landmarks import HandLandmarkDetector
from utils.gesture_detector import GestureDetector
from utils.drawing_utils import DrawingCanvas, GestureDrawingController

class StreamlitGestureApp:
    """
    Streamlit web application for gesture recognition
    """
    
    def __init__(self):
        self.landmark_detector = HandLandmarkDetector()
        self.gesture_detector = GestureDetector()
        self.canvas = DrawingCanvas(600, 400)
        self.drawing_controller = GestureDrawingController(self.canvas)
        
        # Session state
        if 'gesture_history' not in st.session_state:
            st.session_state.gesture_history = []
        if 'current_gesture' not in st.session_state:
            st.session_state.current_gesture = None
        if 'drawing_mode' not in st.session_state:
            st.session_state.drawing_mode = False
    
    def run(self):
        """Run the Streamlit application"""
        st.set_page_config(
            page_title="Gesture Recognition System",
            page_icon="🤚",
            layout="wide"
        )
        
        st.title("🤚 Gesture Recognition System")
        st.markdown("---")
        
        # Sidebar
        self._create_sidebar()
        
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self._create_main_content()
        
        with col2:
            self._create_info_panel()
    
    def _create_sidebar(self):
        """Create sidebar with controls"""
        st.sidebar.title("🎛️ Controls")
        
        # Mode selection
        mode = st.sidebar.selectbox(
            "Select Mode",
            ["Gesture Recognition", "Interactive Drawing", "Data Collection", "Analysis"]
        )
        
        st.session_state.mode = mode
        
        # Gesture settings
        st.sidebar.subheader("⚙️ Gesture Settings")
        confidence_threshold = st.sidebar.slider(
            "Confidence Threshold", 0.0, 1.0, 0.8, 0.05
        )
        self.gesture_detector.CONFIDENCE_THRESHOLD = confidence_threshold
        
        # Drawing settings
        if mode == "Interactive Drawing":
            st.sidebar.subheader("🎨 Drawing Settings")
            color = st.sidebar.selectbox(
                "Color",
                ["black", "red", "green", "blue", "yellow", "purple", "orange"]
            )
            self.canvas.set_color(color)
            
            thickness = st.sidebar.slider("Line Thickness", 1, 20, 3)
            self.canvas.set_thickness(thickness)
            
            col1, col2 = st.sidebar.columns(2)
            with col1:
                if st.button("Clear Canvas"):
                    self.canvas.clear_canvas()
                    st.success("Canvas cleared!")
            
            with col2:
                if st.button("Save Drawing"):
                    self.canvas.save_drawing("web_drawing.png")
                    st.success("Drawing saved!")
        
        # Available gestures
        st.sidebar.subheader("📋 Available Gestures")
        gestures = self.gesture_detector.get_available_gestures()
        for gesture in gestures:
            info = self.gesture_detector.get_gesture_info(gesture)
            if info:
                st.sidebar.markdown(f"**{gesture.title()}**: {info['description']}")
    
    def _create_main_content(self):
        """Create main content area"""
        if st.session_state.mode == "Gesture Recognition":
            self._gesture_recognition_mode()
        elif st.session_state.mode == "Interactive Drawing":
            self._interactive_drawing_mode()
        elif st.session_state.mode == "Data Collection":
            self._data_collection_mode()
        elif st.session_state.mode == "Analysis":
            self._analysis_mode()
    
    def _gesture_recognition_mode(self):
        """Gesture recognition mode"""
        st.header("🎯 Real-time Gesture Recognition")
        
        # Camera feed
        camera_placeholder = st.empty()
        
        # Start camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open camera")
            return
        
        # Process frames
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hand_landmarks_list = self.landmark_detector.detect_hands(frame)
            
            # Detect gestures
            for landmarks in hand_landmarks_list:
                gesture, confidence, gesture_info = self.gesture_detector.detect_gesture(landmarks)
                
                if gesture:
                    # Update session state
                    st.session_state.current_gesture = gesture
                    st.session_state.gesture_history.append({
                        'gesture': gesture,
                        'confidence': confidence,
                        'timestamp': time.time()
                    })
                    
                    # Draw gesture info on frame
                    self._draw_gesture_info_on_frame(processed_frame, gesture, confidence, gesture_info)
            
            # Convert frame for display
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            frame_count += 1
            if frame_count % 30 == 0:  # Update every 30 frames
                time.sleep(0.1)  # Small delay to prevent overwhelming
    
    def _interactive_drawing_mode(self):
        """Interactive drawing mode"""
        st.header("🎨 Interactive Drawing")
        
        # Create two columns for camera and canvas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📹 Camera Feed")
            camera_placeholder = st.empty()
        
        with col2:
            st.subheader("🎨 Drawing Canvas")
            canvas_placeholder = st.empty()
        
        # Start camera
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            st.error("Could not open camera")
            return
        
        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, hand_landmarks_list = self.landmark_detector.detect_hands(frame)
            
            # Process gestures for drawing
            for landmarks in hand_landmarks_list:
                gesture, confidence, gesture_info = self.gesture_detector.detect_gesture(landmarks)
                
                if gesture and confidence > 0.8:
                    hand_position = landmarks.get('wrist', (0, 0))
                    self.drawing_controller.process_gesture(gesture, hand_position)
                    
                    # Draw gesture info
                    self._draw_gesture_info_on_frame(processed_frame, gesture, confidence, gesture_info)
            
            # Display camera feed
            frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            camera_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Display canvas
            canvas_image = self.canvas.get_canvas()
            canvas_rgb = cv2.cvtColor(canvas_image, cv2.COLOR_BGR2RGB)
            canvas_placeholder.image(canvas_rgb, channels="RGB", use_column_width=True)
            
            time.sleep(0.1)
    
    def _data_collection_mode(self):
        """Data collection mode"""
        st.header("📊 Data Collection")
        
        st.info("This mode allows you to collect gesture data for training custom models.")
        
        # Gesture selection
        gesture_name = st.selectbox(
            "Select Gesture to Collect",
            self.gesture_detector.get_available_gestures()
        )
        
        # Collection settings
        samples_per_gesture = st.number_input("Samples per gesture", 10, 1000, 100)
        
        if st.button("Start Data Collection"):
            st.info(f"Collecting {samples_per_gesture} samples for '{gesture_name}' gesture...")
            
            # Data collection logic here
            collected_data = []
            
            cap = cv2.VideoCapture(0)
            sample_count = 0
            
            while sample_count < samples_per_gesture:
                ret, frame = cap.read()
                if not ret:
                    continue
                
                processed_frame, hand_landmarks_list = self.landmark_detector.detect_hands(frame)
                
                for landmarks in hand_landmarks_list:
                    features = self.landmark_detector.calculate_gesture_features(landmarks)
                    collected_data.append({
                        'gesture': gesture_name,
                        'features': features.tolist(),
                        'timestamp': time.time()
                    })
                    sample_count += 1
                
                # Display progress
                progress = sample_count / samples_per_gesture
                st.progress(progress)
                st.write(f"Collected {sample_count}/{samples_per_gesture} samples")
            
            cap.release()
            
            # Save collected data
            if collected_data:
                os.makedirs("data/raw", exist_ok=True)
                filename = f"data/raw/{gesture_name}_{int(time.time())}.json"
                with open(filename, 'w') as f:
                    json.dump(collected_data, f)
                
                st.success(f"Data saved to {filename}")
    
    def _analysis_mode(self):
        """Analysis mode"""
        st.header("📈 Gesture Analysis")
        
        # Gesture history analysis
        if st.session_state.gesture_history:
            st.subheader("Gesture History")
            
            # Create DataFrame for analysis
            import pandas as pd
            df = pd.DataFrame(st.session_state.gesture_history)
            
            # Gesture frequency
            gesture_counts = df['gesture'].value_counts()
            
            # Plot gesture frequency
            fig = px.bar(
                x=gesture_counts.index,
                y=gesture_counts.values,
                title="Gesture Frequency",
                labels={'x': 'Gesture', 'y': 'Count'}
            )
            st.plotly_chart(fig)
            
            # Confidence analysis
            fig2 = px.box(
                df,
                x='gesture',
                y='confidence',
                title="Gesture Confidence Distribution"
            )
            st.plotly_chart(fig2)
            
            # Timeline
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            fig3 = px.line(
                df,
                x='timestamp',
                y='confidence',
                color='gesture',
                title="Gesture Confidence Over Time"
            )
            st.plotly_chart(fig3)
        else:
            st.info("No gesture data available. Start gesture recognition to collect data.")
    
    def _create_info_panel(self):
        """Create information panel"""
        st.header("ℹ️ Information")
        
        # Current gesture
        if st.session_state.current_gesture:
            st.success(f"Current Gesture: {st.session_state.current_gesture.title()}")
        else:
            st.info("No gesture detected")
        
        # Gesture statistics
        if st.session_state.gesture_history:
            st.subheader("📊 Statistics")
            total_gestures = len(st.session_state.gesture_history)
            unique_gestures = len(set([g['gesture'] for g in st.session_state.gesture_history]))
            avg_confidence = np.mean([g['confidence'] for g in st.session_state.gesture_history])
            
            st.metric("Total Gestures", total_gestures)
            st.metric("Unique Gestures", unique_gestures)
            st.metric("Avg Confidence", f"{avg_confidence:.2f}")
        
        # Instructions
        st.subheader("📖 Instructions")
        st.markdown("""
        **Gesture Recognition:**
        - Show your hand to the camera
        - Make different gestures
        - View real-time recognition
        
        **Interactive Drawing:**
        - Pointing: Draw
        - Fist: Eraser
        - Peace: Change color
        - Thumbs up: Save
        - Thumbs down: Clear
        - Open palm: Undo
        - Three fingers: Redo
        """)
    
    def _draw_gesture_info_on_frame(self, frame, gesture, confidence, gesture_info):
        """Draw gesture information on frame"""
        # This is a simplified version for web display
        cv2.putText(frame, f"{gesture.upper()}: {confidence:.2f}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

def main():
    """Main function"""
    app = StreamlitGestureApp()
    app.run()

if __name__ == "__main__":
    main() 