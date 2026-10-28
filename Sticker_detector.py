import cv2
import numpy as np

VIDEO_PATH = r"C:\Users\Gayathri S Anushya\Downloads\ITK\ITK_FUll.mp4"

MIN_GOOD_MATCHES = 15

CONFIDENCE_DIVISOR = 40


class ProfessionalDetector:

    def __init__(self):

        self.sift = cv2.SIFT_create()

        index_params = dict(
            algorithm=1,
            trees=5
        )

        search_params = dict(
            checks=80
        )

        self.flann = cv2.FlannBasedMatcher(
            index_params,
            search_params
        )

        print("Loading video reference frames...")

        self.video_data = self.extract_video_features(
            VIDEO_PATH
        )

        print(
            f"Loaded {len(self.video_data)} frames."
        )

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
    # AUTO ROTATE
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
    # DETECT HOLOGRAM STICKER
    # --------------------------------
    def detect_hologram_region(self, image):

        resized = self.resize_image(
            image
        )

        resized = self.auto_rotate(
            resized
        )

        h, w = resized.shape[:2]

        # SMALLER SMART ROI
        roi_x1 = int(w * 0.38)

        roi_y1 = int(h * 0.40)

        roi_x2 = int(w * 0.75)

        roi_y2 = int(h * 0.75)

        roi = resized[
            roi_y1:roi_y2,
            roi_x1:roi_x2
        ]

        gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        # CLAHE improves hologram edges
        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(
            gray,
            (3, 3),
            0
        )

        edges = cv2.Canny(
            blur,
            100,
            220
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=1
        )

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        best_crop = None

        best_area = 0

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < 4000:
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

            # hologram box mostly rectangle
            if len(approx) != 4:
                continue

            x, y, cw, ch = cv2.boundingRect(
                approx
            )

            aspect_ratio = cw / float(ch)

            # tighter aspect ratio
            if not (0.8 <= aspect_ratio <= 1.8):
                continue

            rect_area = cw * ch

            fill_ratio = area / float(rect_area)

            # remove loose contours
            if fill_ratio < 0.65:
                continue

            if rect_area > best_area:

                best_area = rect_area

                # VERY SMALL PADDING
                crop_x = int(cw * 0.04)

                crop_y = int(ch * 0.04)

                x1 = max(x + crop_x, 0)

                y1 = max(y + crop_y, 0)

                x2 = min(
                    x + cw - crop_x,
                    roi.shape[1]
                )

                y2 = min(
                    y + ch - crop_y,
                    roi.shape[0]
                )

                best_crop = roi[
                    y1:y2,
                    x1:x2
                ]

                # DRAW ONLY EXACT BOX
                cv2.rectangle(
                    roi,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

        if best_crop is not None:

            return best_crop

        return roi

    # --------------------------------
    # VIDEO FEATURES
    # --------------------------------
    def extract_video_features(self, path):

        cap = cv2.VideoCapture(path)

        data = []

        frame_id = 0

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_id += 1

            if frame_id % 5 != 0:
                continue

            hologram = self.detect_hologram_region(
                frame
            )

            gray = cv2.cvtColor(
                hologram,
                cv2.COLOR_BGR2GRAY
            )

            kp, des = self.sift.detectAndCompute(
                gray,
                None
            )

            if des is not None and len(des) >= 10:

                data.append(
                    (hologram, des)
                )

        cap.release()

        return data

    # --------------------------------
    # MATCH IMAGE
    # --------------------------------
    def match(self, image):

        hologram = self.detect_hologram_region(
            image
        )

        gray = cv2.cvtColor(
            hologram,
            cv2.COLOR_BGR2GRAY
        )

        kp1, des1 = self.sift.detectAndCompute(
            gray,
            None
        )

        if des1 is None or len(des1) < 2:

            return None, 0

        best_good_count = 0

        best_frame = None

        for frame, des2 in self.video_data:

            if des2 is None or len(des2) < 2:
                continue

            try:

                matches = self.flann.knnMatch(
                    des1,
                    des2,
                    k=2
                )

            except cv2.error:
                continue

            good = []

            for pair in matches:

                if len(pair) == 2:

                    m, n = pair

                    if m.distance < 0.65 * n.distance:

                        good.append(m)

            if len(good) > best_good_count:

                best_good_count = len(good)

                best_frame = frame.copy()

        if best_frame is None:

            return None, 0

        confidence = min(
            100,
            int(
                (best_good_count / CONFIDENCE_DIVISOR) * 100
            )
        )

        # --------------------------------
        # SHOW TEST + MATCHED FRAME
        # --------------------------------


        return best_frame, confidence