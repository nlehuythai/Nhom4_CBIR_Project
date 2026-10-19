import numpy as np

def calculate_metrics_for_query(retrieved_labels, query_label, top_k=10, total_relevant=100):
    """Tính toán Precision@K, Recall@K và F1@K cho một truy vấn[cite: 30]."""
    retrieved_k = retrieved_labels[:top_k]
    tp = sum(1 for label in retrieved_k if label == query_label)
    
    precision = tp / top_k
    recall = tp / total_relevant
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return precision, recall, f1