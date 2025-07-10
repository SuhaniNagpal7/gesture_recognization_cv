#!/usr/bin/env python3
"""
Test script for the Gesture Recognition System
"""

import sys
import os
import importlib
import traceback

def test_imports():
    """Test if all modules can be imported"""
    print("🔍 Testing module imports...")
    
    modules_to_test = [
        'utils.hand_landmarks',
        'utils.gesture_detector', 
        'utils.drawing_utils',
        'gesture_recognition',
        'drawing_app',
        'data_collector',
        'model_trainer'
    ]
    
    failed_imports = []
    
    for module in modules_to_test:
        try:
            importlib.import_module(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ Failed to import {len(failed_imports)} modules")
        return False
    else:
        print("\n✅ All modules imported successfully!")
        return True

def test_dependencies():
    """Test if required dependencies are available"""
    print("\n🔍 Testing dependencies...")
    
    dependencies = [
        ('cv2', 'opencv-python'),
        ('mediapipe', 'mediapipe'),
        ('numpy', 'numpy'),
        ('tensorflow', 'tensorflow'),
        ('sklearn', 'scikit-learn'),
        ('matplotlib', 'matplotlib'),
        ('seaborn', 'seaborn'),
        ('pandas', 'pandas'),
        ('PIL', 'Pillow'),
        ('streamlit', 'streamlit'),
        ('plotly', 'plotly')
    ]
    
    missing_deps = []
    
    for module, package in dependencies:
        try:
            importlib.import_module(module)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n❌ Missing dependencies: {', '.join(missing_deps)}")
        print("Run: pip install -r requirements.txt")
        return False
    else:
        print("\n✅ All dependencies available!")
        return True

def test_gesture_detection():
    """Test gesture detection functionality"""
    print("\n🔍 Testing gesture detection...")
    
    try:
        from utils.hand_landmarks import HandLandmarkDetector
        from utils.gesture_detector import GestureDetector
        
        # Create instances
        landmark_detector = HandLandmarkDetector()
        gesture_detector = GestureDetector()
        
        # Test gesture patterns
        gestures = gesture_detector.get_available_gestures()
        print(f"✅ Available gestures: {len(gestures)}")
        for gesture in gestures:
            info = gesture_detector.get_gesture_info(gesture)
            if info:
                print(f"  - {gesture}: {info['description']}")
        
        # Test feature calculation
        dummy_landmarks = {
            'wrist': (100, 100, 0),
            'thumb_tip': (110, 90, 0),
            'index_tip': (120, 80, 0),
            'middle_tip': (130, 85, 0),
            'ring_tip': (125, 95, 0),
            'pinky_tip': (115, 105, 0),
            'thumb_ip': (105, 95, 0),
            'index_pip': (115, 85, 0),
            'middle_pip': (125, 90, 0),
            'ring_pip': (120, 100, 0),
            'pinky_pip': (110, 110, 0)
        }
        
        features = landmark_detector.calculate_gesture_features(dummy_landmarks)
        print(f"✅ Feature calculation: {len(features)} features")
        
        # Test gesture detection
        gesture, confidence, info = gesture_detector.detect_gesture(dummy_landmarks)
        print(f"✅ Gesture detection test completed")
        
        landmark_detector.release()
        return True
        
    except Exception as e:
        print(f"❌ Gesture detection test failed: {e}")
        traceback.print_exc()
        return False

def test_drawing_canvas():
    """Test drawing canvas functionality"""
    print("\n🔍 Testing drawing canvas...")
    
    try:
        from utils.drawing_utils import DrawingCanvas, GestureDrawingController
        
        # Create canvas
        canvas = DrawingCanvas(400, 300)
        print("✅ Canvas created")
        
        # Test drawing
        canvas.start_drawing((100, 100))
        canvas.draw((150, 150))
        canvas.stop_drawing()
        print("✅ Drawing test completed")
        
        # Test color change
        canvas.set_color('red')
        print("✅ Color change test completed")
        
        # Test eraser
        canvas.toggle_eraser()
        print("✅ Eraser test completed")
        
        # Test undo/redo
        canvas.undo()
        canvas.redo()
        print("✅ Undo/Redo test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Drawing canvas test failed: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    print("\n🔍 Testing configuration...")
    
    try:
        import json
        
        config_path = "config/gestures.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            gestures = config.get('gestures', {})
            settings = config.get('settings', {})
            
            print(f"✅ Configuration loaded: {len(gestures)} gestures")
            print(f"✅ Settings: {len(settings)} parameters")
            
            return True
        else:
            print("❌ Configuration file not found")
            return False
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_directory_structure():
    """Test if required directories exist"""
    print("\n🔍 Testing directory structure...")
    
    required_dirs = [
        'utils',
        'models',
        'data',
        'data/raw',
        'data/processed',
        'config'
    ]
    
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory):
            print(f"✅ {directory}/")
        else:
            print(f"❌ {directory}/ (missing)")
            missing_dirs.append(directory)
    
    if missing_dirs:
        print(f"\n❌ Missing directories: {', '.join(missing_dirs)}")
        return False
    else:
        print("\n✅ All directories exist!")
        return True

def run_all_tests():
    """Run all tests"""
    print("🧪 GESTURE RECOGNITION SYSTEM - TEST SUITE")
    print("=" * 50)
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Dependencies", test_dependencies),
        ("Module Imports", test_imports),
        ("Configuration", test_configuration),
        ("Gesture Detection", test_gesture_detection),
        ("Drawing Canvas", test_drawing_canvas)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
            traceback.print_exc()
    
    print(f"\n{'='*50}")
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready to use.")
        print("\n🚀 You can now run:")
        print("  python main.py")
        print("  python gesture_recognition.py")
        print("  python drawing_app.py")
        print("  streamlit run web_app.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        print("\n💡 Try installing dependencies:")
        print("  pip install -r requirements.txt")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1) 