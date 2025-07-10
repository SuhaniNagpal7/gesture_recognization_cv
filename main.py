#!/usr/bin/env python3
"""
Main entry point for the Gesture Recognition System
"""

import sys
import os
import subprocess
import time

def print_banner():
    """Print application banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                    GESTURE RECOGNITION SYSTEM                ║
    ║                                                              ║
    ║  🤚 Real-time hand gesture recognition and control          ║
    ║  🎨 Interactive drawing with gesture controls               ║
    ║  📊 Data collection and model training                     ║
    ║  🌐 Web interface for easy interaction                     ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def print_menu():
    """Print main menu options"""
    menu = """
    📋 Available Applications:
    
    1. 🎯 Basic Gesture Recognition
       - Real-time hand gesture detection
       - Visual feedback and gesture information
    
    2. 🎨 Interactive Drawing App
       - Draw with hand gestures
       - Gesture-controlled drawing tools
    
    3. 🌐 Web Interface
       - Streamlit-based web application
       - Multiple modes in one interface
    
    4. 📊 Data Collection Tool
       - Collect custom gesture datasets
       - Prepare data for model training
    
    5. 🤖 Model Training
       - Train custom gesture recognition models
       - Compare different algorithms
    
    6. 📖 Help & Documentation
       - View project documentation
       - Learn about supported gestures
    
    7. 🛠️ Install Dependencies
       - Install required packages
       - Setup development environment
    
    0. 🚪 Exit
    
    """
    print(menu)

def run_gesture_recognition():
    """Run basic gesture recognition"""
    print("\n🎯 Starting Basic Gesture Recognition...")
    print("Press 'q' to quit, 'h' for help")
    time.sleep(2)
    
    try:
        from gesture_recognition import main
        main()
    except ImportError as e:
        print(f"Error: Could not import gesture recognition module - {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"Error running gesture recognition: {e}")

def run_drawing_app():
    """Run interactive drawing application"""
    print("\n🎨 Starting Interactive Drawing App...")
    print("Use gestures to control drawing:")
    print("- Pointing: Draw")
    print("- Fist: Eraser")
    print("- Peace: Change color")
    print("- Thumbs up: Save")
    print("- Thumbs down: Clear")
    print("Press 'q' to quit")
    time.sleep(2)
    
    try:
        from drawing_app import main
        main()
    except ImportError as e:
        print(f"Error: Could not import drawing app module - {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"Error running drawing app: {e}")

def run_web_interface():
    """Run Streamlit web interface"""
    print("\n🌐 Starting Web Interface...")
    print("The web interface will open in your browser.")
    print("Press Ctrl+C to stop the server.")
    time.sleep(2)
    
    try:
        # Check if streamlit is available
        import streamlit
        print("Starting Streamlit server...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "web_app.py"])
    except ImportError:
        print("Error: Streamlit not found. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "streamlit"])
        print("Starting Streamlit server...")
        subprocess.run([sys.executable, "-m", "streamlit", "run", "web_app.py"])
    except Exception as e:
        print(f"Error running web interface: {e}")

def run_data_collection():
    """Run data collection tool"""
    print("\n📊 Starting Data Collection Tool...")
    print("This tool helps you collect custom gesture datasets.")
    time.sleep(2)
    
    try:
        from data_collector import main
        main()
    except ImportError as e:
        print(f"Error: Could not import data collector module - {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"Error running data collection: {e}")

def run_model_training():
    """Run model training"""
    print("\n🤖 Starting Model Training...")
    print("This will train custom gesture recognition models.")
    print("Make sure you have collected some data first.")
    time.sleep(2)
    
    try:
        from model_trainer import main
        main()
    except ImportError as e:
        print(f"Error: Could not import model trainer module - {e}")
        print("Make sure all dependencies are installed.")
    except Exception as e:
        print(f"Error running model training: {e}")

def show_help():
    """Show help and documentation"""
    help_text = """
    📖 GESTURE RECOGNITION SYSTEM - HELP
    
    🎯 Supported Gestures:
    
    1. Pointing (Index finger extended)
       - Action: Draw
       - Use for drawing on canvas
    
    2. Fist (Closed hand)
       - Action: Erase
       - Toggle eraser mode
    
    3. Peace Sign (Index + Middle finger)
       - Action: Select/Change color
       - Cycle through colors
    
    4. Thumbs Up
       - Action: Confirm/Save
       - Save current drawing
    
    5. Thumbs Down
       - Action: Cancel/Clear
       - Clear canvas
    
    6. Open Palm (All fingers extended)
       - Action: Clear/Undo
       - Undo last action
    
    7. Three Fingers (Index + Middle + Ring)
       - Action: Zoom/Redo
       - Redo last undone action
    
    8. OK Sign (Thumb + Index forming circle)
       - Action: OK
       - Confirm action
    
    🛠️ System Requirements:
    - Python 3.7+
    - Webcam
    - Good lighting
    - Clear hand visibility
    
    📁 Project Structure:
    - gesture_recognition.py: Basic gesture detection
    - drawing_app.py: Interactive drawing application
    - web_app.py: Streamlit web interface
    - data_collector.py: Data collection tool
    - model_trainer.py: Model training
    - utils/: Core modules
    - config/: Configuration files
    - data/: Data storage
    - models/: Trained models
    
    🔧 Troubleshooting:
    - Make sure webcam is connected and accessible
    - Ensure good lighting conditions
    - Install all dependencies: pip install -r requirements.txt
    - Check camera permissions
    
    📞 For more help, check the README.md file.
    """
    print(help_text)

def install_dependencies():
    """Install required dependencies"""
    print("\n🛠️ Installing Dependencies...")
    
    try:
        # Check if requirements.txt exists
        if not os.path.exists("requirements.txt"):
            print("Error: requirements.txt not found!")
            return
        
        print("Installing packages from requirements.txt...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully!")
        
    except Exception as e:
        print(f"Error installing dependencies: {e}")
        print("Please install manually: pip install -r requirements.txt")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        'opencv-python',
        'mediapipe',
        'numpy',
        'tensorflow',
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'pandas',
        'Pillow',
        'streamlit',
        'plotly'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Run option 7 to install dependencies.")
        return False
    
    print("✅ All dependencies are installed!")
    return True

def main():
    """Main function"""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        print("\nWould you like to install missing dependencies? (y/n)")
        response = input().lower().strip()
        if response == 'y':
            install_dependencies()
    
    while True:
        print_menu()
        
        try:
            choice = input("Enter your choice (0-7): ").strip()
            
            if choice == '0':
                print("\n👋 Goodbye! Thanks for using Gesture Recognition System!")
                break
            elif choice == '1':
                run_gesture_recognition()
            elif choice == '2':
                run_drawing_app()
            elif choice == '3':
                run_web_interface()
            elif choice == '4':
                run_data_collection()
            elif choice == '5':
                run_model_training()
            elif choice == '6':
                show_help()
            elif choice == '7':
                install_dependencies()
            else:
                print("❌ Invalid choice. Please enter a number between 0-7.")
            
            if choice in ['1', '2', '3', '4', '5']:
                input("\nPress Enter to return to main menu...")
                
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using Gesture Recognition System!")
            break
        except Exception as e:
            print(f"Error: {e}")
            input("\nPress Enter to continue...")

if __name__ == "__main__":
    main() 