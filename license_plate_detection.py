import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO
import argparse

# Set the path to the Tesseract OCR executable
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

class LicensePlateDetector:
    """A class to detect and recognize license plates from images using YOLO and Tesseract OCR."""

    def __init__(self, model_path="best.pt"):
        """
        Initialize the LicensePlateDetector with a YOLO model.

        Args:
            model_path (str): Path to the YOLO model file (default: 'best.pt').
        """
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Load the YOLO model from the specified path.

        Args:
            model_path (str): Path to the YOLO model file.

        Returns:
            YOLO: Loaded YOLO model object.

        Raises:
            SystemExit: If model loading fails.
        """
        try:
            return YOLO(model_path)
        except Exception as e:
            print(f"Error loading YOLO model: {e}")
            exit(1)

    def detect_and_recognize(self, image_path, display_scale=50, save_output=False):
        """
        Detect and recognize license plates in an image.

        Args:
            image_path (str): Path to the input image.
            display_scale (int): Percentage to scale the display image (default: 50).
            save_output (bool): Whether to save the output image (default: False).
        """
        # Load the image
        image = self._load_image(image_path)
        if image is None:
            print("Image loading failed. Exiting.")
            return

        # Calculate resized dimensions for display
        width = int(image.shape[1] * display_scale / 100)
        height = int(image.shape[0] * display_scale / 100)

        # Perform detection using YOLO
        results = self.model(image)

        # Initialize the license plate counter
        plate_count = 0

        # Process each detected license plate
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cropped_plate = image[y1:y2, x1:x2]

                # Recognize text from the cropped plate
                plate_text = self._recognize_text_from_plate(cropped_plate)

                # Counter increments
                plate_count += 1

                # Print the license plate results with serial number
                print(f"Detected license plate {plate_count}: {plate_text}")

                # Draw bounding box and text on the image
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, f"p{plate_count}: {plate_text}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

        # Display or save the result
        if save_output:
            output_path = f"output_{image_path.split('/')[-1]}"
            cv2.imwrite(output_path, image)
            print(f"Output saved to {output_path}")
        else:
            resized_result = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            cv2.imshow("License Plate Detection", resized_result)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    def _load_image(self, image_path):
        """
        Load an image from the specified path.

        Args:
            image_path (str): Path to the image file.

        Returns:
            numpy.ndarray: Loaded image, or None if loading fails.
        """
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to load image from {image_path}")
        return image

    def _recognize_text_from_plate(self, plate_image):
        """
        Recognize text from a license plate image using Tesseract OCR.

        Args:
            plate_image (numpy.ndarray): Cropped image of the license plate.

        Returns:
            str: Recognized text from the plate.
        """
        if plate_image is None or plate_image.size == 0:
            return "No plate detected"

        # Convert to grayscale
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

        # Apply OTSU thresholding for binarization
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Configure Tesseract OCR to recognize alphanumeric characters and '·'
        custom_config = r'--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789·'
        text = pytesseract.image_to_string(binary, config=custom_config)

        return text.strip()

def main():
    """Parse command-line arguments and run the license plate detector."""
    parser = argparse.ArgumentParser(description="Detect and recognize license plates in images.")
    parser.add_argument("--image", type=str, default="./cars0.jpg", help="Path to the input image")
    parser.add_argument("--scale", type=int, default=50, help="Display scale percentage")
    parser.add_argument("--save", action="store_true", help="Save the output image")
    args = parser.parse_args()

    # Instantiate and run the detector
    detector = LicensePlateDetector()
    detector.detect_and_recognize(args.image, display_scale=args.scale, save_output=args.save)

if __name__ == "__main__":
    main()