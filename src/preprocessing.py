#contains logic for preprocessing the image data
import cv2
import numpy as np
from PIL import Image, ImageOps

def preprocess(img: Image.Image) -> Image.Image:
    # grayscale
    img = ImageOps.grayscale(img)
    # normalize contrast
    img = ImageOps.autocontrast(img)
    # binary conversion (black text white background)
    img = binarize(img)
    # removing horizontal lines
    img = remove_horizontal_lines(img)
    return img

def binarize(img: Image.Image) -> Image.Image:
    gray = np.array(img.convert("L"))
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(bw)

def remove_horizontal_lines(img: Image.Image, min_line_frac: float = 0.25) -> Image.Image:
    bw = np.array(img.convert("L"))
    ink = cv2.bitwise_not(bw)

    # 1. find lines: keep only horizontal runs at least 25% of the page width
    width = max(1, int(ink.shape[1] * min_line_frac))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, 1))
    lines = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    lines = cv2.dilate(lines, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))   # cover the line's full thickness

    # 2. subtract lines, then patch small gaps they cut through letters
    cleaned = cv2.subtract(ink, lines)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3)))

    return Image.fromarray(cv2.bitwise_not(cleaned))   