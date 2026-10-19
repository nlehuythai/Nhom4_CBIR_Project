import cv2
import numpy as np
from scipy.stats import skew

def extract_color_moments(img):
    """Trích xuất 3 đại lượng thống kê (Mean, Std, Skewness) trên 3 kênh HSV[cite: 30]."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    moments = []
    for i in range(3):
        channel = hsv[:, :, i].astype(np.float32) / 255.0
        mean = np.mean(channel)
        std = np.std(channel)
        skewness = skew(channel.flatten())
        if np.isnan(skewness):
            skewness = 0.0
        moments.extend([mean, std, skewness])
    
    moments = np.array(moments, dtype=np.float32)
    norm = np.linalg.norm(moments)
    if norm > 0:
        moments /= norm
    return moments