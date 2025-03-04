# License Plate Detection and Recognition

This project provides a Python script for detecting and recognizing license plates from images using the YOLO object detection model and Tesseract OCR for text recognition.

## Features

- **License Plate Detection**: A pre-trained YOLO model detects license plates in images.
- **Text Recognition**: Employs Tesseract OCR to extract text from detected plates.
- **Visualization**: Draws bounding boxes and recognized text on the image.
- **Flexible Output**: Option to display results or save them to a file.
- **Command-Line Interface**: Supports customizable parameters via command-line arguments.

## Requirements

- **Platform**: Windows 11
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
   ```
2. **Install Python Dependencies**:
   ```bash
   pip install opencv-python numpy pytesseract ultralytics
   ```
3. **Install Tesseract OCR**:
   - Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).
   - Update the `pytesseract.pytesseract.tesseract_cmd` variable in the script to point to your Tesseract executable. For example:
   ```python
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```
4. **Prepare the YOLO Model**:
   - Place your pre-trained YOLO model (e.g., `best.pt`) in the project directory. If your model is located elsewhere, specify its path when initializing the `LicensePlateDetector` class.

## Usage

- Run the script with default settings:
  ```bash
  python license_plate_detection.py
   ```

### Command-Line Options

- `--image`: Path to the input image (default: `./cars0.jpg`).
- `--scale`: Display scale percentage for the output image (default: `50`).
- `--save`: Save the output image instead of displaying it (flag).

### Example:

- Ex:
  ```bash
  python license_plate_detection.py --image "./cars0.jpg" --scale 75 --save
  ```
- This command will process the image `cars0.jpg`, scale the display to 75%, and save the result as `output_cars0.jpg`.

## Customization

- Model Path: Modify the model_path parameter in LicensePlateDetector.__init__() to use a different YOLO model.
- OCR Settings: Adjust the custom_config in _recognize_text_from_plate() to match your license plate format. For example, to allow specific characters:
  ```python
  custom_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789·'
  ```
- Display Scale: Use the `--scale` argument to adjust the size of the displayed or saved image.

## Example Output

- Input: An image containing one or more vehicles with license plates.
- Output: The image with green bounding boxes around detected license plates and the recognized text displayed above each box.
- Sample Console Output:
  ```text
  Detected license plate 1: AM7A043
  Detected license plate 2: AH14623
  ```

## Project Structure
 <!-- TREEVIEW START -->
 ```bash
 license-plate-detection/
 │
 ├── license_plate_detection.py  # Main script for detection and recognition
 ├── best.pt                     # Pre-trained YOLO model (example)
 ├── cars0.jpg                   # Sample input image
 └── README.md                   # Project documentation
 ```
 <!-- TREEVIEW END -->

> **💡 TIP 1**: You can download the `best.pt` file from [here](https://mahaljsp.ddns.net/files/yolov8/car.pt)
>
> **💡 TIP 2**: You can download car plate license images from Kaggle: [Car License Plate Detection](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection?resource=download).

## Troubleshooting

- Model Loading Error: If you encounter an error like $\color{#ff5733}{Error\ loading\ YOLO\ model:\ pickle\ data\ was\ truncated}$, ensure that the model file is not corrupted and the path is correct. You may need to re-download the model or check the file integrity.
- Tesseract Not Found: Make sure Tesseract OCR is installed and the path in the script is correctly set.
- Image Not Found: Verify that the input image path is correct and the file exists.

## License

This project is licensed under the MIT License.

## Acknowledgements

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for the object detection framework.
- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) for text recognition capabilities.

## Contributing

Feel free to submit issues or pull requests if you have suggestions for improvements or new features. Contributions are welcome!
