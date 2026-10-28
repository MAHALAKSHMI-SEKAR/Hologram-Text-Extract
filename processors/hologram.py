import cv2
import easyocr
import numpy as np
import re


class HologramScanner:

    def __init__(self):

        print("\n[INFO] Loading EasyOCR...")

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

        print("[INFO] OCR READY\n")

    # --------------------------------
    # UPSCALE IMAGE
    # --------------------------------
    def upscale(self, image, scale=4):

        h, w = image.shape[:2]

        resized = cv2.resize(
            image,
            (w * scale, h * scale),
            interpolation=cv2.INTER_CUBIC
        )

        return resized

    # --------------------------------
    # ROTATE IMAGE
    # --------------------------------
    def rotate_image(self, image, angle):

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
    # ENHANCE WHITE TEXT
    # --------------------------------
    def enhance(self, image):

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        # upscale
        gray = self.upscale(
            gray,
            scale=5
        )

        # CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=4,
            tileGridSize=(8, 8)
        )

        enhanced = clahe.apply(gray)

        # brightness improve
        enhanced = cv2.convertScaleAbs(
            enhanced,
            alpha=2.0,
            beta=30
        )

        # sharpen
        kernel = np.array([
            [-1, -1, -1],
            [-1, 9, -1],
            [-1, -1, -1]
        ])

        sharpened = cv2.filter2D(
            enhanced,
            -1,
            kernel
        )

        return sharpened

    # --------------------------------
    # DETECT SERIAL AREA
    # --------------------------------
    def detect_serial_region(self, image):

        h, w = image.shape[:2]

        # bottom-middle focus
        x1 = int(w * 0.10)

        y1 = int(h * 0.45)

        x2 = int(w * 0.95)

        y2 = int(h * 0.95)

        roi = image[y1:y2, x1:x2]

        return roi

    # --------------------------------
    # CREATE OCR VARIANTS
    # --------------------------------
    def create_variants(self, image):

        variants = []

        enhanced = self.enhance(
            image
        )

        variants.append(
            ("enhanced", enhanced)
        )

        # multiple thresholds
        for thresh in [100, 120, 140, 160, 180]:

            _, binary = cv2.threshold(
                enhanced,
                thresh,
                255,
                cv2.THRESH_BINARY
            )

            variants.append(
                (f"binary_{thresh}", binary)
            )

            inverted = cv2.bitwise_not(
                binary
            )

            variants.append(
                (f"invert_{thresh}", inverted)
            )

        # adaptive threshold
        adaptive = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5
        )

        variants.append(
            ("adaptive", adaptive)
        )

        return variants

    # --------------------------------
    # CLEAN TEXT
    # --------------------------------
    def clean(self, text):

        text = text.upper()

        replacements = {
            'O': '0',
            'I': '1',
            'L': '1',
            'S': '5',
            'B': '8',
            'Z': '2'
        }

        cleaned = ""

        for ch in text:

            cleaned += replacements.get(
                ch,
                ch
            )

        return cleaned

    # --------------------------------
    # OCR ENGINE
    # --------------------------------
    def run_ocr(self, image):

        results = self.reader.readtext(
            image,
            detail=1,
            paragraph=False,
            allowlist='0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )

        print("\n=== OCR RESULTS ===")

        line_candidates = []

        # --------------------------------
        # STORE OCR BOXES
        # --------------------------------
        for result in results:

            bbox = result[0]

            text = result[1]

            conf = result[2]

            print(
                f"{text} ({conf:.2f})"
            )

            if conf < 0.08:
                continue

            x1 = int(bbox[0][0])

            y1 = int(bbox[0][1])

            x2 = int(bbox[2][0])

            y2 = int(bbox[2][1])

            line_candidates.append({

                "text": text,

                "x": x1,

                "y": y1,

                "w": x2 - x1,

                "h": y2 - y1
            })

        # --------------------------------
        # SORT
        # --------------------------------
        line_candidates = sorted(
            line_candidates,
            key=lambda k: (k["y"], k["x"])
        )

        # --------------------------------
        # MERGE TEXT
        # --------------------------------
        merged = ""

        prev_x = None

        prev_y = None

        for item in line_candidates:

            text = item["text"]

            x = item["x"]

            y = item["y"]

            if prev_x is not None:

                dx = abs(x - prev_x)

                dy = abs(y - prev_y)

                if dx < 120 and dy < 50:

                    merged += text

                else:

                    merged += " " + text

            else:

                merged += text

            prev_x = x

            prev_y = y

        print(f"\n[MERGED OCR] {merged}")

        return merged

    # --------------------------------
    # SERIAL EXTRACTION
    # --------------------------------
    def extract_serial(self, text):

        text = self.clean(text)

        print(f"\n[CLEANED] {text}")

        filtered = re.sub(
            r'[^A-Z0-9\-]',
            ' ',
            text
        )

        filtered = re.sub(
            r'\s+',
            ' ',
            filtered
        )

        print(f"[FILTERED] {filtered}")

        # --------------------------------
        # PRIORITY 1
        # STRICT SERIAL FORMAT
        # --------------------------------

        serial_patterns = [

            r'\b\d{3}-\d{6}\b',

            r'\b\d{3}-\d{5,8}\b',

            r'\b\d{2,4}-\d{4,8}\b'
        ]

        for pattern in serial_patterns:

            matches = re.findall(
                pattern,
                filtered
            )

            if matches:

                print("\n[STRICT MATCHES]")

                print(matches)

                return matches[0]

        # --------------------------------
        # FALLBACK
        # --------------------------------

        tokens = filtered.split()

        candidates = []

        for token in tokens:

            token = token.strip()

            if len(token) < 5:
                continue

            score = 0

            # PURE NUMERIC SERIALS
            if re.fullmatch(r'\d+-\d+', token):
                score += 10

            # contains digits
            if re.search(r'\d', token):
                score += 2

            # contains dash
            if '-' in token:
                score += 3

            # letters reduce confidence
            if re.search(r'[A-Z]', token):
                score -= 2

            # longer token
            if len(token) >= 7:
                score += 2

            candidates.append(
                (token, score)
            )

        if len(candidates) == 0:

            return "Not Detected"

        candidates = sorted(
            candidates,
            key=lambda x: x[1],
            reverse=True
        )

        print("\n[CANDIDATES]")

        for c in candidates:

            print(c)

        return candidates[0][0]

    # --------------------------------
    # MAIN SCAN
    # --------------------------------
    def scan(self, image):

        if image is None or image.size == 0:

            return "Not Detected"

        # --------------------------------
        # MULTI PREPROCESSING
        # --------------------------------

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = self.upscale(
            gray,
            scale=4
        )

        clahe = cv2.createCLAHE(
            clipLimit=3.0,
            tileGridSize=(8, 8)
        )

        gray_clahe = clahe.apply(gray)

        blur = cv2.GaussianBlur(
            gray_clahe,
            (3, 3),
            0
        )

        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            11,
            2
        )

        edges = cv2.Canny(
            gray_clahe,
            50,
            150
        )

        images_to_try = [

            ("GRAY", gray),

            ("CLAHE", gray_clahe),

            ("THRESH", thresh),

            ("EDGES", edges)
        ]

        # --------------------------------
        # ROTATIONS
        # --------------------------------

        rotations = [

            ("0°", None),

            ("90°", cv2.ROTATE_90_CLOCKWISE),

            ("180°", cv2.ROTATE_180),

            ("270°", cv2.ROTATE_90_COUNTERCLOCKWISE)
        ]

        # --------------------------------
        # OCR LOOP
        # --------------------------------

        for method_name, img in images_to_try:

            for angle_name, rotate_flag in rotations:

                if rotate_flag is None:

                    rotated = img

                else:

                    rotated = cv2.rotate(
                        img,
                        rotate_flag
                    )

                print("\n======================")

                print(
                    f"[METHOD] {method_name}"
                )

                print(
                    f"[ROTATION] {angle_name}"
                )

                print("======================")

                # --------------------------------
                # OCR
                # --------------------------------

                results = self.reader.readtext(
                    rotated,
                    detail=1,
                    paragraph=False,
                    allowlist='0123456789-ABCDEFGHIJKLMNOPQRSTUVWXYZ-'
                )

                words = []

                print("\n[OCR OUTPUT]")

                for result in results:

                    text = result[1]

                    conf = result[2]

                    print(
                        f"{text} ({conf:.2f})"
                    )

                    if conf > 0.30:

                        words.extend(
                            text.upper().split()
                        )

                full_text = " ".join(words)

                print(
                    f"\n[FULL TEXT] {full_text}"
                )

                # --------------------------------
                # SERIAL EXTRACTION
                # --------------------------------

                serial = self.extract_serial(
                    full_text
                )

                if (
                    serial != "Not Detected"
                    and serial != ""
                ):

                    print(
                        f"\n✅ FOUND: {serial}"
                    )

                    return serial

        return "Not Detected"