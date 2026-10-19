import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def extract_lbp(img, P=8, R=1):
    """Trích xuất đặc trưng kết cấu Local Binary Pattern (LBP) từ ảnh xám[cite: 30]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray, P, R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    norm = np.linalg.norm(hist)
    if norm > 0:
        hist /= norm
    return hist