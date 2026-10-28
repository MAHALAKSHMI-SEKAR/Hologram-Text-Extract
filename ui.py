import cv2
import tkinter as tk

from tkinter import filedialog
from PIL import Image, ImageTk

from main import OCRScanner
from Sticker_detector import ProfessionalDetector


class OCRScannerUI:

    def __init__(self, root):

        self.root = root

        self.root.title("OCR Hologram Scanner")

        self.root.geometry("1600x950")

        self.root.configure(bg="#edf1f7")

        self.scanner = OCRScanner()

        self.detector = ProfessionalDetector()

        # ====================================================
        # HEADER
        # ====================================================
        header = tk.Frame(
            root,
            bg="#0f172a",
            height=90
        )

        header.pack(
            fill="x"
        )

        title = tk.Label(
            header,
            text="OCR Hologram Scanner",
            font=("Arial", 24, "bold"),
            fg="white",
            bg="#0f172a"
        )

        title.pack(
            pady=22
        )

        # ====================================================
        # UPLOAD BUTTON  
        # ====================================================
        button_frame = tk.Frame(
            root,bg="#edf1f7"
        )

        button_frame.pack(
            pady=10
        )

        upload_btn = tk.Button(
            button_frame,
            text="UPLOAD IMAGE",
            command=self.upload_image,
            font=("Arial", 14, "bold"),
            bg="#2563eb",
            fg="white",
            padx=30,
            pady=12,
            relief="flat",
            cursor="hand2"
        )

        upload_btn.pack()

        # ====================================================
        # SCROLLABLE CANVAS
        # ====================================================
        self.canvas = tk.Canvas(
            root,
            bg="#edf1f7",
            highlightthickness=0
        )

        self.scrollbar = tk.Scrollbar(
            root,
            orient="vertical",
            command=self.canvas.yview
        )

        self.canvas.configure(
            yscrollcommand=self.scrollbar.set
        )

        self.scrollbar.pack(
            side="right",
            fill="y"
        )

        self.canvas.pack(
            side="left",
            fill="both",
            expand=True
        )

        # ====================================================
        # MAIN CONTAINER
        # ====================================================
        self.main_container = tk.Frame(
            self.canvas,
            bg="#edf1f7"
        )

        self.canvas.create_window(
            (0, 0),
            window=self.main_container,
            anchor="nw"
        )

        # ====================================================
        # AUTO SCROLL REGION
        # ====================================================
        self.main_container.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        # ====================================================
        # MOUSE WHEEL SCROLL
        # ====================================================
        self.canvas.bind_all(
            "<MouseWheel>",
            self._on_mousewheel
        )

        # ====================================================
        # LEFT SECTION
        # ====================================================
        left_section = tk.Frame(
            self.main_container,
            bg="#edf1f7"
        )

        left_section.grid(
            row=0,
            column=0,
            padx=15,
            pady=10,
            sticky="n"
        )

        # ====================================================
        # RIGHT SECTION
        # ====================================================
        right_section = tk.Frame(
            self.main_container,
            bg="#edf1f7"
        )

        right_section.grid(
            row=0,
            column=1,
            padx=15,
            pady=10,
            sticky="n"
        )

        # ====================================================
        # ORIGINAL IMAGE CARD
        # ====================================================
        original_card = tk.Frame(
            left_section,
            bg="white",
            bd=1,
            relief="solid"
        )

        original_card.pack(
            fill="x",
            pady=10
        )

        tk.Label(
            original_card,
            text="Uploaded Image",
            font=("Arial", 16, "bold"),
            bg="white",
            fg="#111827"
        ).pack(
            pady=12
        )

        self.original_label = tk.Label(
            original_card,
            bg="white"
        )

        self.original_label.pack(
            pady=10
        )

        # ====================================================
        # CROPPED REGION CARD
        # ====================================================
        crop_card = tk.Frame(
            left_section,
            bg="white",
            bd=1,
            relief="solid"
        )

        crop_card.pack(
            fill="x",
            pady=10
        )

        tk.Label(
            crop_card,
            text="Detected Hologram Region",
            font=("Arial", 16, "bold"),
            bg="white",
            fg="#111827"
        ).pack(
            pady=12
        )

        self.crop_label = tk.Label(
            crop_card,
            bg="white"
        )

        self.crop_label.pack(
            pady=10
        )

        # ====================================================
        # RESULT CARD
        # ====================================================
        # result_card = tk.Frame(
        #     right_section,
        #     bg="white",
        #     bd=1,
        #     relief="solid"
        # )

        # result_card.pack(
        #     fill="x",
        #     pady=10
        # )

        # tk.Label(
        #     result_card,
        #     text="Verification Result",
        #     font=("Arial", 16, "bold"),
        #     bg="white",
        #     fg="#111827"
        # ).pack(
        #     pady=12
        # )

        # self.result_label = tk.Label(
        #     result_card,
        #     text="Waiting For Upload...",
        #     font=("Arial", 22, "bold"),
        #     fg="#f59e0b",
        #     bg="white"
        # )

        # self.result_label.pack(
        #     pady=10
        # )

        # self.confidence_label = tk.Label(
        #     result_card,
        #     text="Confidence : 0%",
        #     font=("Arial", 15, "bold"),
        #     fg="#374151",
        #     bg="white"
        # )

        # self.confidence_label.pack(
        #     pady=10
        # )

        # ====================================================
        # VIDEO MATCH CARD
        # ====================================================
        video_card = tk.Frame(
            right_section,
            bg="white",
            bd=1,
            relief="solid"
        )

        video_card.pack(
            fill="x",
            pady=10
        )

        tk.Label(
            video_card,
            text="Best Match Video Frame",
            font=("Arial", 16, "bold"),
            bg="white",
            fg="#111827"
        ).pack(
            pady=12
        )

        self.video_label = tk.Label(
            video_card,
            bg="white"
        )

        self.video_label.pack(
            pady=10
        )

        # ====================================================
        # OCR DETAILS CARD
        # ====================================================
        details_card = tk.Frame(
            right_section,
            bg="white",
            bd=1,
            relief="solid"
        )

        details_card.pack(
            fill="x",
            pady=10
        )

        tk.Label(
            details_card,
            text="OCR Extracted Details",
            font=("Arial", 16, "bold"),
            bg="white",
            fg="#111827"
        ).pack(
            pady=15
        )

        # ====================================================
        # SERIAL
        # ====================================================
        self.serial_label = tk.Label(
            details_card,
            text="Serial Number :",
            font=("Arial", 15, "bold"),
            fg="#2563eb",
            bg="white",
            anchor="w"
        )

        self.serial_label.pack(
            fill="x",
            padx=20,
            pady=10
        )

        # ====================================================
        # PART NUMBER
        # ====================================================
        self.part_label = tk.Label(
            details_card,
            text="Part Number :",
            font=("Arial", 15, "bold"),
            fg="#059669",
            bg="white",
            anchor="w"
        )

        self.part_label.pack(
            fill="x",
            padx=20,
            pady=10
        )

        # ====================================================
        # GRADE
        # ====================================================
        self.grade_label = tk.Label(
            details_card,
            text="Grade :",
            font=("Arial", 15, "bold"),
            fg="#7c3aed",
            bg="white",
            anchor="w"
        )

        self.grade_label.pack(
            fill="x",
            padx=20,
            pady=10
        )

        # ====================================================
        # LOT NUMBER
        # ====================================================
        self.lot_label = tk.Label(
            details_card,
            text="Lot Number :",
            font=("Arial", 15, "bold"),
            fg="#ea580c",
            bg="white",
            anchor="w"
        )

        self.lot_label.pack(
            fill="x",
            padx=20,
            pady=10
        )

        self.logo_label = tk.Label(
            details_card,
            text="Logo :",
            font=("Arial", 15, "bold"),
            fg="#059669",
            bg="white",
            anchor="w"
        )

        self.logo_label.pack(
            fill="x",
            padx=20,
            pady=10
        )
    # ====================================================
    # MOUSE SCROLL
    # ====================================================
    def _on_mousewheel(self, event):

        self.canvas.yview_scroll(
            int(-1 * (event.delta / 120)),
            "units"
        )

    # ====================================================
    # DISPLAY IMAGE
    # ====================================================
    def display_image(
        self,
        image,
        label,
        width,
        height
    ):

        if image is None:
            return

        image_rgb = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        image_rgb = cv2.resize(
            image_rgb,
            (width, height)
        )

        pil_image = Image.fromarray(
            image_rgb
        )

        tk_image = ImageTk.PhotoImage(
            pil_image
        )

        label.config(
            image=tk_image
        )

        label.image = tk_image

    # ====================================================
    # UPLOAD IMAGE
    # ====================================================
    def upload_image(self):

        file_path = filedialog.askopenfilename(
            filetypes=[
                (
                    "Image Files",
                    "*.jpg *.jpeg *.png *.bmp"
                )
            ]
        )

        if not file_path:
            return

        # ====================================================
        # OCR PROCESS
        # ====================================================
        result = self.scanner.process_image(
            file_path
        )

        # ====================================================
        # SHOW IMAGES
        # ====================================================
        self.display_image(
            result["original"],
            self.original_label,
            650,
            350
        )

        self.display_image(
            result["cropped"],
            self.crop_label,
            320,
            250
        )

        # ====================================================
        # OCR TEXT
        # ====================================================
        self.serial_label.config(
            text=f"Serial Number : {result['serial']}"
        )

        self.part_label.config(
            text=f"Part Number : {result['part_no']}"
        )

        self.grade_label.config(
            text=f"Grade : {result['grade']}"
        )

        self.lot_label.config(
            text=f"Lot Number : {result['lot']}"
        )
        # image = cv2.imread(
        #     file_path
        # )
        cropped_region = result["cropped"]
        best_frame, confidence = (
            self.detector.match(cropped_region)
        )
        if confidence >= 35:
            self.logo_label.config(
                text=f"Logo : ITK {confidence}%",
                fg="#16a34a"
            )
        else:
            self.logo_label.config(
                text=f"Logo : Not Detected {confidence}%",
                fg="#dc2626"
            )
        # ====================================================
        # DETECTOR
        # ====================================================
        # image = cv2.imread(
        #     file_path
        # )

        # best_frame, confidence = (
        #     self.detector.match(image)
        # )

        # self.confidence_label.config(
        #     text=f"Confidence : {confidence}%"
        # )

        # # ====================================================
        # # RESULT
        # # ====================================================
        # if confidence >= 90:

        #     self.result_label.config(
        #         text="HOLOGRAM VERIFIED ✅",
        #         fg="#16a34a"
        #     )

        # else:

        #     self.result_label.config(
        #         text="NOT MATCHED ❌",
        #         fg="#dc2626"
        #     )

        # ====================================================
        # SHOW VIDEO FRAME
        # ====================================================
        if best_frame is not None:

            self.display_image(
                best_frame,
                self.video_label,
                500,
                300
            )


# ====================================================
# MAIN
# ====================================================
if __name__ == "__main__":

    root = tk.Tk()

    app = OCRScannerUI(root)

    root.mainloop()