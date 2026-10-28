import cv2
import easyocr
import re


class ExtractText:

    def __init__(self):

        self.reader = easyocr.Reader(
            ['en'],
            gpu=False
        )

    # --------------------------------
    # EXTRACT SERIAL
    # --------------------------------
    def extract_serial(self, text):

        text = text.upper()

        
        match = re.search(r'(\d{2,4})\s*[-/]\s*([\d\.]{2,10})', text)
        if match:
            part1 = match.group(1)
            part2 = re.sub(r'\D', '', match.group(2))
            return f"{part1}-{part2}"

        match = re.search(r'([A-Z]{1,3})\s*-\s*(\d{4,10})', text)
        if match:
            return f"{match.group(1)}-{match.group(2)}"

        match = re.search(r'\b([A-Z]{1,3}\d{5,10})\b', text)
        if match:
            return match.group(1)

        match = re.search(r'\b(\d{5,10})\b', text)
        if match:
            return match.group(1)

        return "Not found"

    # --------------------------------
    # GET OCR TEXT
    # --------------------------------
    def get_full_text(self, img):

        results = self.reader.readtext(img)

        words = []

        for (_, text, prob) in results:

            if prob > 0.3:

                words.extend(
                    text.upper().split()
                )
                print(f"Detected text: {text} (confidence: {prob:.2f})")

        return " ".join(words)

    # --------------------------------
    # MAIN SCAN
    # --------------------------------
    def scan(self, image):

        if image is None:

            return {
                "part_no": "Not found",
                "grade": "Not found",
                "lot": "Not found",
            }

        # --------------------------------
        # PREPROCESS
        # --------------------------------
        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.convertScaleAbs(
            gray,
            alpha=1.5,
            beta=20
        )

        # --------------------------------
        # OCR
        # --------------------------------
        results = self.reader.readtext(gray)

        filtered_results = []

        for (_, text, prob) in results:

            if prob > 0.3:

                filtered_results.append(
                    text.upper()
                )
                print(f"Detected text: {text} (confidence: {prob:.2f})")
        full_text = " ".join(filtered_results)

        # --------------------------------
        # PART NO
        # --------------------------------
        part_no = "Not found"

        match = re.search(
            r'PART\s*NO[:\s]*([A-Z0-9/_-]+)',
            full_text
        )

        if match:
            part_no = match.group(1)
            part_no = part_no.replace("I","/")

        # --------------------------------
        # GRADE
        # --------------------------------
        grade = "Not found"

        words = []

        for item in filtered_results:

            split_words = item.upper().split()

            for w in split_words:

                clean = re.sub(r'[^A-Z0-9]', '', w)

                if clean:
                    words.append(clean)

        for i, word in enumerate(words):

            if word == "GRADE":

                collected = []

                # take next 4 words after GRADE
                next_words = words[i+1:i+5]

                for w in next_words:

                    # skip unwanted small OCR garbage
                    if len(w) <= 1:
                        continue

                    # stop if another field starts
                    if w in [
                        "LOT",
                        "NUMBER",
                        "QTY",
                        "PO",
                        "SCR",
                        "MFG"
                    ]:
                        break

                    collected.append(w)

                if collected:

                    grade = " ".join(collected)

                    break

        # --------------------------------
        # LOT NUMBER
        # --------------------------------
        lot = "Not found"

        # Flexible pattern for LOT NUMBER line
        match = re.search(
            r'LOT\s*NUMBER\s*[:;]?\s*([A-Z0-9\s]{4,10})',
            full_text,
            re.IGNORECASE
        )

        if match:
            raw = match.group(1).strip().upper()

            raw = re.sub(r'[^A-Z0-9]', '', raw)

            match2 = re.match(r'(\d{2})([A-Z]?)(\d{2})', raw)

            if match2:
                part1 = match2.group(1)
                middle = match2.group(2) if match2.group(2) else "?"
                part2 = match2.group(3)

                lot = f"{part1} {middle} {part2}"

        # --------------------------------
        # SERIAL
        # --------------------------------
        
        
        return {

            "part_no": part_no,

            "grade": grade,

            "lot": lot,
        }