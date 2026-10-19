const API_BASE = "http://localhost:8000";

// Xử lý trang Image Retrieval (index.html)
const uploadInput = document.getElementById('uploadInput');
const imagePreview = document.getElementById('imagePreview');
const previewContainer = document.getElementById('previewContainer');

if (uploadInput) {
    uploadInput.addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file) {
            imagePreview.src = URL.createObjectURL(file);
            previewContainer.classList.remove('hidden');
        }
    });
}

async function handleSearch() {
    const file = uploadInput.files[0];
    const method = document.getElementById('methodSelect').value;
    const topK = document.getElementById('topKSelect').value || 6;

    if (!file) {
        alert('Vui lòng chọn một ảnh truy vấn trước!');
        return;
    }

    const formData = new FormData();
    formData.append('file', file);
    formData.append('method', method);
    formData.append('top_k', topK);

    const resultsGrid = document.getElementById('resultsGrid');
    resultsGrid.innerHTML = '<p class="text-gray-400 col-span-full text-center py-12">Đang xử lý trích xuất và tính khoảng cách...</p>';

    try {
        const response = await fetch(`${API_BASE}/api/retrieve`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        resultsGrid.innerHTML = '';
        if (data.status === 'success' && data.results.length > 0) {
            data.results.forEach(item => {
                resultsGrid.innerHTML += `
                    <div class="border rounded-lg p-3 bg-gray-50 shadow-sm hover:shadow transition">
                        <img src="${API_BASE}${item.path}" class="w-full h-32 object-cover rounded mb-2">
                        <p class="text-xs font-semibold text-gray-700 truncate">${item.path.split('/').pop()}</p>
                        <p class="text-xs text-indigo-600 font-medium">Khoảng cách: ${item.distance.toFixed(4)}</p>
                    </div>
                `;
            });
        } else {
            resultsGrid.innerHTML = '<p class="text-red-400 col-span-full text-center py-12">Không tìm thấy kết quả phù hợp hoặc lỗi server.</p>';
        }
    } catch (error) {
        console.error('Lỗi kết nối Backend:', error);
        alert('Không thể kết nối đến API Server FastAPI!');
    }
}

// Xử lý trang Dataset Gallery (gallery.html)
async function loadGallery() {
    const galleryGrid = document.getElementById('galleryGrid');
    if (!galleryGrid) return;

    try {
        const response = await fetch(`${API_BASE}/api/gallery`);
        const data = await response.json();

        galleryGrid.innerHTML = '';
        data.images.forEach(imgUrl => {
            galleryGrid.innerHTML += `
                <div class="border rounded-lg p-2 bg-white shadow-sm">
                    <img src="${API_BASE}${imgUrl}" class="w-full h-28 object-cover rounded mb-1">
                    <p class="text-[10px] text-gray-500 truncate">${imgUrl.split('/').pop()}</p>
                </div>
            `;
        });
    } catch (e) {
        galleryGrid.innerHTML = '<p class="text-gray-400 col-span-full text-center">Chưa kết nối được với thư mục dataset trên server.</p>';
    }
}
document.addEventListener("DOMContentLoaded", loadGallery);