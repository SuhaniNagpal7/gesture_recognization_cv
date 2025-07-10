#!/usr/bin/env python3
"""
Installation script for Gesture Recognition System
"""

import subprocess
import sys
import os
import platform

def print_banner():
    """Print installation banner"""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║              GESTURE RECOGNITION SYSTEM INSTALLER            ║
    ║                                                              ║
    ║  🤚 Installing dependencies and setting up the system       ║
    ║  🎨 Creating necessary directories and files                ║
    ║  📊 Preparing data collection tools                         ║
    ║  🤖 Setting up model training environment                   ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    print("🔍 Checking Python version...")
    
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print(f"❌ Python {version.major}.{version.minor} detected")
        print("This system requires Python 3.7 or higher")
        return False
    
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} detected")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("\n📦 Installing dependencies...")
    
    try:
        # Upgrade pip first
        print("Upgrading pip...")
        subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], 
                      check=True, capture_output=True)
        
        # Install requirements
        print("Installing packages from requirements.txt...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], 
                              check=True, capture_output=True, text=True)
        
        print("✅ Dependencies installed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        print("Error output:", e.stderr)
        return False
    except FileNotFoundError:
        print("❌ requirements.txt not found!")
        return False

def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    
    directories = [
        'models',
        'data',
        'data/raw',
        'data/processed',
        'config'
    ]
    
    for directory in directories:
        try:
            os.makedirs(directory, exist_ok=True)
            print(f"✅ Created {directory}/")
        except Exception as e:
            print(f"❌ Error creating {directory}/: {e}")
            return False
    
    return True

def test_installation():
    """Test if installation was successful"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import cv2
        import mediapipe
        import numpy
        import tensorflow
        import sklearn
        import matplotlib
        import streamlit
        
        print("✅ All core dependencies imported successfully!")
        
        # Test camera access
        print("Testing camera access...")
        cap = cv2.VideoCapture(0)
        if cap.isOpened():
            print("✅ Camera access successful!")
            cap.release()
        else:
            print("⚠️  Camera not accessible (this is normal if no camera is connected)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Test error: {e}")
        return False

def show_next_steps():
    """Show next steps after installation"""
    print("\n🎉 Installation completed successfully!")
    print("\n📋 Next Steps:")
    print("1. Run the system: python main.py")
    print("2. Test basic gesture recognition: python gesture_recognition.py")
    print("3. Try interactive drawing: python drawing_app.py")
    print("4. Launch web interface: streamlit run web_app.py")
    print("5. Collect custom data: python data_collector.py")
    print("6. Train custom models: python model_trainer.py")
    
    print("\n📖 Documentation:")
    print("- README.md: Project overview and usage")
    print("- config/gestures.json: Gesture configurations")
    print("- Run 'python test_system.py' to verify everything works")
    
    print("\n💡 Tips:")
    print("- Ensure good lighting for best gesture recognition")
    print("- Keep your hand clearly visible to the camera")
    print("- Start with basic gestures like pointing and fist")
    print("- Use the web interface for easy interaction")

def main():
    """Main installation function"""
    print_banner()
    
    # Check Python version
    if not check_python_version():
        print("\n❌ Installation failed: Incompatible Python version")
        return False
    
    # Create directories
    if not create_directories():
        print("\n❌ Installation failed: Could not create directories")
        return False
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Installation failed: Could not install dependencies")
        print("Try installing manually: pip install -r requirements.txt")
        return False
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation failed: Tests failed")
        return False
    
    # Show next steps
    show_next_steps()
    
    return True

if __name__ == "__main__":
    try:
        success = main()
        if success:
            print("\n✅ Installation completed successfully!")
        else:
            print("\n❌ Installation failed!")
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n⚠️  Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error during installation: {e}")
        sys.exit(1) 