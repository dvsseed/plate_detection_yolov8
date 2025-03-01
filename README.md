# License Plate Detection and Recognition

This project provides a Python script for detecting and recognizing license plates from images using the YOLO object detection model and Tesseract OCR for text recognition.

## Features

- **License Plate Detection**: A pre-trained YOLO model detects license plates in images.
- **Text Recognition**: Employs Tesseract OCR to extract text from detected plates.
- **Visualization**: Draws bounding boxes and recognized text on the image.
- **Flexible Output**: Option to display results or save them to a file.
- **Command-Line Interface**: Supports customizable parameters via command-line arguments.

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy (`numpy`)
- Pytesseract (`pytesseract`)
- Ultralytics YOLO (`ultralytics`)
- Tesseract OCR installed on your system

## Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/dvsseed/plate_detection_yolov8.git
   cd plate_detection_yolov8
