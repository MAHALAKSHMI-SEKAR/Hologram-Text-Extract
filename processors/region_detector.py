import cv2
import numpy as np


class RegionDetector:

    def __init__(self):
        pass

    # --------------------------------
    # RESIZE IMAGE
    # --------------------------------
    def resize_image(self, image, width=1400):

        h, w = image.shape[:2]

        scale = width / float(w)

        new_h = int(h * scale)

        resized = cv2.resize(
            image,
            (width, new_h)
        )

        return resized

    # --------------------------------
    # AUTO ROTATE IMAGE
    # --------------------------------
    def auto_rotate(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blur,
            50,
            150
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=2
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        largest = max(
            contours,
            key=cv2.contourArea
        )

        rect = cv2.minAreaRect(
            largest
        )

        angle = rect[-1]

        if angle < -45:
            angle = 90 + angle

        elif angle > 45:
            angle = angle - 90

        h, w = image.shape[:2]

        center = (w // 2, h // 2)

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    # --------------------------------
    # PREPROCESS
    # --------------------------------
    def preprocess(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        return thresh

    # --------------------------------
    # DETECT EXACT HOLOGRAM BOX
    # --------------------------------
    def detect_frame(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blur = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        edges = cv2.Canny(
            blur,
            80,
            200
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (5, 5)
        )

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=2
        )

        contours, hierarchy = cv2.findContours(
            edges,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_crop = None

        best_score = -999999

        h, w = image.shape[:2]

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 10000:
                continue

            peri = cv2.arcLength(
                cnt,
                True
            )

            approx = cv2.approxPolyDP(
                cnt,
                0.02 * peri,
                True
            )

            if len(approx) != 4:
                continue

            x, y, cw, ch = cv2.boundingRect(
                approx
            )

            aspect_ratio = cw / float(ch)

            if 0.7 < aspect_ratio < 2.2:

                rect_area = cw * ch

                fill_ratio = area / float(rect_area)

                if fill_ratio < 0.45:
                    continue

                center_x = x + cw // 2

                center_y = y + ch // 2

                dist_x = abs(
                    center_x - w // 2
                )

                dist_y = abs(
                    center_y - h // 2
                )

                score = (
                    rect_area
                    - dist_x * 2
                    - dist_y * 2
                )

                if score > best_score:

                    best_score = score

                    pad_x = int(cw * 0.08)

                    pad_y = int(ch * 0.08)

                    x1 = max(x - pad_x, 0)

                    y1 = max(y - pad_y, 0)

                    x2 = min(
                        x + cw + pad_x,
                        w
                    )

                    y2 = min(
                        y + ch + pad_y,
                        h
                    )

                    best_crop = image[
                        y1:y2,
                        x1:x2
                    ]

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        3
                    )

        return best_crop

    # --------------------------------
    # MAIN
    # --------------------------------
    def detect_hologram_region(self, image):

        resized = self.resize_image(
            image
        )

        # AUTO STRAIGHTEN
        resized = self.auto_rotate(
            resized
        )

        h, w = resized.shape[:2]

        # --------------------------------
        # SMART ROI
        # --------------------------------
        roi_x1 = int(w * 0.22)

        roi_y1 = int(h * 0.28)

        roi_x2 = int(w * 0.82)

        roi_y2 = int(h * 0.82)

        roi = resized[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        # --------------------------------
        # DETECT HOLOGRAM BOX
        # --------------------------------
        detected = self.detect_frame(
            roi
        )

        if detected is not None:

            return detected

        # --------------------------------
        # FALLBACK
        # --------------------------------
        fallback_x1 = int(w * 0.40)

        fallback_y1 = int(h * 0.38)

        fallback_x2 = int(w * 0.72)

        fallback_y2 = int(h * 0.75)

        fallback = resized[
            fallback_y1:fallback_y2,
            fallback_x1:fallback_x2
        ]

        return fallback


# --------------------------------
# TEST
# --------------------------------
if __name__ == "__main__":

    image_path = r"YOUR_IMAGE_PATH.jpg"

    image = cv2.imread(image_path)

    if image is None:
        print("Image not found")
        exit()

    detector = RegionDetector()

    result = detector.detect_hologram_region(
        image
    )

    cv2.imshow(
        "Detected Hologram",
        result
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()