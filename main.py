import cv2

from processors.region_detector import RegionDetector
from processors.hologram import HologramScanner
from processors.Extract_Text import ExtractText

class OCRScanner:

    def __init__(self):

        self.region_detector = RegionDetector()

        self.hologram_scanner = HologramScanner()

        self.text_extractor = ExtractText()
    # --------------------------------
    # MAIN PROCESS
    # --------------------------------
    def process_image(self, image_path):

        # load image
        image = cv2.imread(image_path)

        if image is None:

            return {
                "success": False,
                "message": "Image not found",
                "original": None,
                "cropped": None,
                "serial": "Not Detected"
            }

        # --------------------------------
        # DETECT HOLOGRAM REGION
        # --------------------------------
        cropped_region = (
            self.region_detector
            .detect_hologram_region(image)
        )
        Part_number = (
            self.text_extractor
            .scan(image)
        )
        # --------------------------------
        # SCAN SERIAL
        # --------------------------------
        serial_number = (
            self.hologram_scanner
            .scan(cropped_region)
        )

        # --------------------------------
        # RETURN DATA
        # --------------------------------
        return {
            "success": True,
            "message": "Completed",

            "original": image,

            "cropped": cropped_region,

            "serial": serial_number,

            "part_no": Part_number["part_no"],

            "grade": Part_number["grade"],

            "lot": Part_number["lot"]
        }
