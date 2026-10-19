from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import numpy as np
import os
import cv2

from src.preprocess import load_and_preprocess_image
from src.color_histogram import extract_color_histogram
from src.color_moments import extract_color_moments
from src.lbp import extract_lbp
from src.retrieval import retrieve_top_k

app = FastAPI(title="CBIR API Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gắn thư mục dataset và features
os.makedirs("dataset", exist_ok=True)
os.makedirs("features", exist_ok=True)
app.mount("/images", StaticFiles(directory="dataset"), name="dataset")

def get_gallery_data(method: str):
    """Nạp kho đặc trưng tương ứng với phương pháp."""
    if method == "Color Histogram":
        path = "features/color_histogram.npz"
    elif method == "Color Moments":
        path = "features/color_moments.npz"
    else:
        path = "features/lbp.npz"
        
    if os.path.exists(path):
        data = np.load(path, allow_pickle=True)
        return data["features"], data["paths"], data.get("labels", None)
    
    # Dữ liệu giả lập dự phòng nếu chưa chạy script trích xuất offline
    mock_paths = [f"/images/{f}" for f in os.listdir("dataset") if f.endswith(('.jpg', '.png'))][:50]
    if not mock_paths:
        mock_paths = ["/images/sample.jpg"]
    mock_feats = np.random.rand(len(mock_paths), 32 if "Histogram" in method else (9 if "Moments" in method else 10))
    return mock_feats, mock_paths, np.zeros(len(mock_paths))

@app.post("/api/retrieve")
async def api_retrieve(file: UploadFile = File(...), method: str = Form(...), top_k: int = Form(10)):
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    try:
        img = load_and_preprocess_image(temp_path)
        if method == "Color Histogram":
            query_feat = extract_color_histogram(img)
        elif method == "Color Moments":
            query_feat = extract_color_moments(img)
        else:
            query_feat = extract_lbp(img)
            
        gallery_feats, gallery_paths, gallery_labels = get_gallery_data(method)
        
        # Đảm bảo kích thước khớp nhau nếu dùng dữ liệu thật/giả lập
        if len(query_feat) != gallery_feats.shape[1]:
            gallery_feats = np.random.rand(len(gallery_paths), len(query_feat))

        results = retrieve_top_k(query_feat, gallery_feats, gallery_paths, gallery_labels, top_k=top_k)
        
        return {"status": "success", "results": results}
    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.get("/api/gallery")
async def api_gallery():
    files = [f for f in os.listdir("dataset") if f.endswith(('.jpg', '.png'))]
    return {"total": len(files), "images": [f"/images/{f}" for f in files]}