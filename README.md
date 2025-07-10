# Gesture Recognition System

A comprehensive computer vision project that recognizes hand gestures in real-time and maps them to various actions, including drawing applications.

## Features

- **Real-time Hand Detection**: Uses MediaPipe for accurate hand landmark detection
- **Multiple Gesture Recognition**: 
  - Pointing gesture (index finger extended)
  - Fist gesture (closed hand)
  - Peace sign (index and middle finger)
  - Thumbs up/down
  - Open palm
  - Custom gestures
- **Interactive Drawing Application**: Gestures control drawing functions
- **Gesture-to-Action Mapping**: Configurable mapping system
- **Data Collection Tool**: Built-in tool for collecting custom gesture datasets
- **Model Training**: Support for custom gesture model training
- **Modern Web Interface**: Streamlit-based UI for easy interaction

## Supported Gestures

1. **Pointing** - Index finger extended (Drawing mode)
2. **Fist** - Closed hand (Erase mode)
3. **Peace Sign** - Index and middle finger (Select mode)
4. **Thumbs Up** - Thumb pointing up (Confirm action)
5. **Thumbs Down** - Thumb pointing down (Cancel action)
6. **Open Palm** - All fingers extended (Clear canvas)
7. **Custom Gestures** - Train your own gestures

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gesture-recognition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python main.py
```

## Usage

### Basic Gesture Recognition
```bash
python gesture_recognition.py
```

### Interactive Drawing Application
```bash
python drawing_app.py
```

### Web Interface
```bash
streamlit run web_app.py
```

### Data Collection
```bash
python data_collector.py
```

## Project Structure

```
gesture-recognition/
├── main.py                 # Main application entry point
├── gesture_recognition.py  # Core gesture recognition module
├── drawing_app.py         # Interactive drawing application
├── web_app.py            # Streamlit web interface
├── data_collector.py     # Data collection tool
├── model_trainer.py      # Custom model training
├── utils/
│   ├── __init__.py
│   ├── gesture_detector.py
│   ├── hand_landmarks.py
│   └── drawing_utils.py
├── models/
│   └── gesture_model.h5
├── data/
│   ├── raw/
│   └── processed/
├── config/
│   └── gestures.json
└── requirements.txt
```

## Configuration

Edit `config/gestures.json` to customize gesture mappings:

```json
{
  "pointing": {
    "action": "draw",
    "description": "Index finger extended for drawing"
  },
  "fist": {
    "action": "erase",
    "description": "Closed hand for erasing"
  }
}
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

MIT License - see LICENSE file for details 