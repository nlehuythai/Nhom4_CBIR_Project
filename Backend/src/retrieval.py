import numpy as np

def compute_euclidean_distance(query_feat, gallery_feats):
    """Tính khoảng cách Euclidean giữa vector query và toàn bộ gallery[cite: 30]."""
    distances = np.linalg.norm(gallery_feats - query_feat, axis=1)
    return distances

def retrieve_top_k(query_feat, gallery_feats, gallery_paths, gallery_labels=None, top_k=10):
    """Sắp xếp khoảng cách tăng dần và trả về Top-K kết quả[cite: 30]."""
    distances = compute_euclidean_distance(query_feat, gallery_feats)
    sorted_indices = np.argsort(distances)
    
    results = []
    for idx in sorted_indices[:top_k]:
        res_item = {
            "path": gallery_paths[idx],
            "distance": float(distances[idx]),
            "index": int(idx)
        }
        if gallery_labels is not None:
            res_item["label"] = str(gallery_labels[idx])
        results.append(res_item)
    return results