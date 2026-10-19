import cv2
import numpy as np

def extract_color_histogram(img, bins=(32, 32, 32)):
    """Trích xuất Color Histogram trong không gian HSV và chuẩn hóa L2."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist