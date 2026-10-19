import cv2
import numpy as np

def load_and_preprocess_image(image_path, target_size=(256, 256)):
    """Nạp và chuẩn hóa kích thước ảnh đầu vào[cite: 30]."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Không thể đọc ảnh từ đường dẫn: {image_path}")
    img_resized = cv2.resize(img, target_size)
    return img_resized