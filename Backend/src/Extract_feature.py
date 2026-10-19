import os
import cv2
import numpy as np
from Preprocess import load_and_preprocess_image
from color_histogram import extract_color_histogram
from color_moments import extract_color_moments
from lbp import extract_lbp

# Cấu hình đường dẫn (Điều chỉnh tên thư mục 'data' hoặc 'dataset' cho khớp với máy bạn)
DATASET_DIR = "data" if os.path.exists("data") else "dataset"
FEATURE_DIR = "features"

os.makedirs(FEATURE_DIR, exist_ok=True)

def build_feature_database():
    if not os.path.exists(DATASET_DIR):
        print(f"❌ Không tìm thấy thư mục chứa dữ liệu: {DATASET_DIR}. Vui lòng tạo thư mục và bỏ ảnh vào!")
        return

    image_files = [f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    if len(image_files) == 0:
        print(f"⚠️ Thư mục {DATASET_DIR} đang trống! Hãy thêm ảnh vào trước khi trích xuất.")
        return

    print(f"🔍 Tìm thấy {len(image_files)} ảnh trong {DATASET_DIR}. Đang tiến hành trích xuất đặc trưng...")

    paths = []
    hist_features = []
    moments_features = []
    lbp_features = []
    labels = []

    for idx, filename in enumerate(image_files):
        img_path = os.path.join(DATASET_DIR, filename)
        
        try:
            # 1. Tiền xử lý ảnh
            img = load_and_preprocess_image(img_path)
            
            # 2. Trích xuất 3 loại đặc trưng
            h_feat = extract_color_histogram(img)
            m_feat = extract_color_moments(img)
            l_feat = extract_lbp(img)
            
            paths.append(img_path)
            hist_features.append(h_feat)
            moments_features.append(m_feat)
            lbp_features.append(l_feat)
            
            # Giả lập nhãn lớp dựa theo tên file (ví dụ: beach_001.jpg thuộc lớp 'beach')
            label = filename.split('_')[0] if '_' in filename else "class_0"
            labels.append(label)
            
            if (idx + 1) % 50 == 0:
                print(f"   Đã xử lý {idx + 1}/{len(image_files)} ảnh...")
        except Exception as e:
            print(f"Lỗi xử lý file {filename}: {e}")

    # 3. Lưu vào các file .npz trong thư mục features/
    np.savez(os.path.join(FEATURE_DIR, "color_histogram.npz"), 
             features=np.array(hist_features), paths=np.array(paths), labels=np.array(labels))
    
    np.savez(os.path.join(FEATURE_DIR, "color_moments.npz"), 
             features=np.array(moments_features), paths=np.array(paths), labels=np.array(labels))
    
    np.savez(os.path.join(FEATURE_DIR, "lbp.npz"), 
             features=np.array(lbp_features), paths=np.array(paths), labels=np.array(labels))

    print("✨ Hoàn tất! Đã lưu thành công các file đặc trưng vào thư mục `features/`.")

if __name__ == "__main__":
    build_feature_database()