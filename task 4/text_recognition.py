
import cv2
import pytesseract
from pytesseract import Output
from PIL import Image, ImageDraw, ImageFont
import os
import shutil


def configure_tesseract():
    executable = shutil.which("tesseract")
    if executable is None:
        standard_path = os.path.join(
            os.environ.get("ProgramFiles", r"C:\Program Files"),
            "Tesseract-OCR",
            "tesseract.exe",
        )
        if os.path.isfile(standard_path):
            executable = standard_path

    if executable is None:
        raise FileNotFoundError(
            "Tesseract OCR was not found. Install it from "
            "https://github.com/UB-Mannheim/tesseract/wiki and add it to PATH."
        )

    pytesseract.pytesseract.tesseract_cmd = executable


def create_sample_image(path="sample_input.png"):
    img = Image.new("RGB", (800, 300), color="white")
    draw = ImageDraw.Draw(img)

    try:
        font_large = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 40
        )
        font_small = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 26
        )
    except IOError:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    draw.text((40, 40), "Artificial Intelligence", fill="black", font=font_large)
    draw.text((40, 100), "Text Recognition Project 4", fill="black", font=font_large)
    draw.text((40, 170), "Hello World! OCR Test 2026", fill="black", font=font_small)
    draw.text((40, 210), "Model: Tesseract LSTM Engine", fill="black", font=font_small)

    img.save(path)
    return path
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    return image, thresh

def recognize_text(processed_image):
  
    recognized_text = pytesseract.image_to_string(processed_image)
    data = pytesseract.image_to_data(processed_image, output_type=Output.DICT)
    return recognized_text, data

def display_results(recognized_text, data):
    print("=" * 60)
    print(" RECOGNIZED TEXT")
    print("=" * 60)
    print(recognized_text.strip())
    print()

    print("=" * 60)
    print(" PER-WORD MODEL OUTPUT (word, confidence %)")
    print("=" * 60)
    n_boxes = len(data["text"])
    detected_words = 0
    total_conf = 0
    for i in range(n_boxes):
        word = data["text"][i].strip()
        conf = int(data["conf"][i])
        if word and conf > 0:
            detected_words += 1
            total_conf += conf
            print(f"  '{word}'  ->  confidence: {conf}%")

    print()
    print("=" * 60)
    print(" SUMMARY")
    print("=" * 60)
    print(f"  Words detected      : {detected_words}")
    if detected_words:
        print(f"  Average confidence  : {total_conf / detected_words:.1f}%")
    print("=" * 60)

def save_annotated_image(image, data, out_path="annotated_output.png"):
    annotated = image.copy()
    n_boxes = len(data["text"])
    for i in range(n_boxes):
        if int(data["conf"][i]) > 0 and data["text"][i].strip():
            (x, y, w, h) = (data["left"][i], data["top"][i],
                            data["width"][i], data["height"][i])
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 200, 0), 2)
            cv2.putText(annotated, f"{data['conf'][i]}%", (x, y - 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
    cv2.imwrite(out_path, annotated)
    return out_path

if __name__ == "__main__":
    configure_tesseract()
    sample_path = create_sample_image()
    print(f"[1/4] Sample input image created: {sample_path}")

    original, processed = preprocess_image(sample_path)
    print("[2/4] Image preprocessed (grayscale + threshold)")

    text, data = recognize_text(processed)
    print("[3/4] Pre-trained Tesseract OCR model ran successfully")

    print("[4/4] Displaying results...\n")
    display_results(text, data)

    annotated_path = save_annotated_image(original, data)
    print(f"\nAnnotated image with bounding boxes saved to: {annotated_path}")
