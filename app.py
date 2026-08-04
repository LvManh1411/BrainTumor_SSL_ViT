import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import cv2
import os
import sys
from torchvision import transforms

# Thêm thư mục gốc vào đường dẫn hệ thống
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from models.classifier import BrainTumorClassifier

# --- CẤU HÌNH TRANG WEB ---
st.set_page_config(
    page_title="Chẩn Đoán U Não MRI - ViT SSL",
    page_icon="🧠",
    layout="wide"
)

CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_NAMES_VIETNAMESE = {
    'glioma': 'U đệm (Glioma)',
    'meningioma': 'U màng não (Meningioma)',
    'notumor': 'Bình thường - Không có u (No Tumor)',
    'pituitary': 'U tuyến yên (Pituitary)'
}

# --- NẠP MÔ HÌNH (CACHE) ---
@st.cache_resource
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BrainTumorClassifier(num_classes=len(CLASSES), freeze_backbone=False)
    
    weights_path = "checkpoints/finetune/best_classifier.pth"
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    else:
        st.error(f" Không tìm thấy file trọng số tại {weights_path}")
        
    model.to(device)
    model.eval()
    return model, device

# --- TRÍCH XUẤT ATTENTION MAP (XAI) ---
def generate_attention_map(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # 1. Trích xuất patch embeddings
        x = model.encoder._process_input(image_tensor)
        
        # 2. Ghép [CLS] token vào vị trí đầu tiên
        n = x.shape[0]
        batch_class_token = model.encoder.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        # 3. Đưa qua Transformer Encoder
        encoder_out = model.encoder.encoder(x)
        
        cls_token = encoder_out[:, 0:1, :]
        patch_tokens = encoder_out[:, 1:, :]
        
        cls_norm = F.normalize(cls_token, p=2, dim=-1)
        patch_norm = F.normalize(patch_tokens, p=2, dim=-1)
        
        similarity = torch.bmm(patch_norm, cls_norm.transpose(1, 2)).squeeze()
        attn_map = similarity.reshape(14, 14).cpu().numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        
    return attn_map

# --- GIAO DIỆN CHÍNH ---
st.title(" HỆ THỐNG CHẨN ĐOÁN U NÃO BẰNG VISION TRANSFORMER (ViT + SSL)")
st.markdown("---")

model, device = load_trained_model()

# Cột bên trái: Upload & Thiết lập
st.sidebar.header(" Tải Ảnh MRI")
uploaded_file = st.sidebar.file_drop_zip = st.sidebar.file_uploader(
    "Chọn file ảnh chụp cắt lớp MRI (.jpg, .png, .jpeg):", 
    type=["jpg", "jpeg", "png"]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Thông tin mô hình:**
* **Kiến trúc:** Vision Transformer (ViT)
* **Phương pháp:** Self-Supervised Learning (SSL)
* **Số lớp chẩn đoán:** 4 lớp
""")

# Luồng xử lý chính
if uploaded_file is not None:
    # 1. Hiển thị ảnh upload
    raw_image = Image.open(uploaded_file).convert("RGB")
    orig_img_resized = raw_image.resize((224, 224))
    
    # 2. Tiền xử lý
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    input_tensor = transform(raw_image).unsqueeze(0)

    # 3. Dự đoán
    with torch.no_grad():
        outputs = model(input_tensor.to(device))
        probs = F.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        pred_class = CLASSES[pred_idx]
        confidence = probs[pred_idx].item() * 100

    # 4. Tạo Attention Heatmap
    attn_map = generate_attention_map(model, input_tensor, device)
    attn_map_resized = cv2.resize(attn_map, (224, 224), interpolation=cv2.INTER_CUBIC)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * attn_map_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    orig_np = np.array(orig_img_resized)
    overlay = cv2.addWeighted(orig_np, 0.6, heatmap, 0.4, 0)

    # --- HIỂN THỊ KẾT QUẢ ---
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. Ảnh MRI Gốc")
        st.image(orig_img_resized, use_container_width=True)
        
    with col2:
        st.subheader("2. Vùng Chú Ý (XAI)")
        st.image(attn_map_resized, use_container_width=True, clamp=True)
        
    with col3:
        st.subheader("3. Heatmap Chồng Lên Ảnh")
        st.image(overlay, use_container_width=True)

    st.markdown("---")
    
    # Bảng kết quả chẩn đoán
    st.subheader(" KẾT QUẢ CHẨN ĐOÁN CHI TIẾT")
    
    if pred_class == "notumor":
        st.success(f" **Chẩn đoán:** {CLASS_NAMES_VIETNAMESE[pred_class]} (Độ tin cậy: {confidence:.2f}%)")
    else:
        st.error(f"**Cảnh báo phát hiện:** {CLASS_NAMES_VIETNAMESE[pred_class]} (Độ tin cậy: {confidence:.2f}%)")
        
    st.write("**Xác suất chi tiết theo từng lớp:**")
    for i, cls in enumerate(CLASSES):
        prob_val = probs[i].item() * 100
        st.progress(int(prob_val), text=f"{CLASS_NAMES_VIETNAMESE[cls]}: {prob_val:.2f}%")

else:
    st.info(" Vui lòng chọn hoặc kéo thả 1 ảnh MRI vào thanh bên trái để bắt đầu chẩn đoán!")