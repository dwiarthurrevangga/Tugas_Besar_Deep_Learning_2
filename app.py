import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Absensi Mahasiswa AI",
    page_icon="🎓",
    layout="centered"
)

# --- LOAD MODEL & RESOURCE (Di-cache biar cepat) ---
@st.cache_resource
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load MTCNN (Untuk Deteksi Wajah)
    mtcnn = MTCNN(keep_all=True, device=device)
    
    # 2. Load InceptionResnetV1 (Untuk Pengenalan)
    resnet = InceptionResnetV1(
        classify=True, 
        pretrained='vggface2', 
        num_classes=70 # Harus sama dengan saat training (70 kelas)
    )
    
    # 3. Load Bobot yang sudah ditraining (.pth)
    try:
        # Ganti nama file sesuai file Anda
        checkpoint = torch.load('8714InceptionResnet_best.pth', map_location=device)
        
        # Load state dictionary
        resnet.load_state_dict(checkpoint['model_state_dict'])
        
        # Ambil daftar nama kelas yang tersimpan
        class_names = checkpoint['class_names']
        
        resnet.to(device)
        resnet.eval()
        
        return mtcnn, resnet, class_names, device
        
    except FileNotFoundError:
        st.error("File model .pth tidak ditemukan! Pastikan file '8714InceptionResnet_best.pth' ada di folder yang sama.")
        return None, None, None, None

# Load Resources
mtcnn, model, class_names, device = load_resources()

# --- UTILS: PREPROCESSING ---
def preprocess_image(img_crop):
    """Mengubah potongan wajah menjadi format yang dimengerti model"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Resize ke 224x224 sesuai training
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(img_crop).unsqueeze(0).to(device)

# --- UI APLIKASI ---
st.title("🎓 Sistem Presensi Face Recognition")
st.write("Upload foto mahasiswa untuk mendeteksi identitas.")

# Upload File
uploaded_file = st.file_uploader("Pilih Foto...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Buka gambar
    image = Image.open(uploaded_file).convert('RGB')
    
    # Tampilkan gambar asli sementara
    st.image(image, caption='Foto Diupload', use_column_width=True)
    
    with st.spinner('Sedang mendeteksi wajah...'):
        # 1. DETEKSI WAJAH (MTCNN)
        boxes, _ = mtcnn.detect(image)
        
        if boxes is not None:
            st.success(f"Ditemukan {len(boxes)} wajah!")
            
            # Draw boxes di gambar
            img_draw = image.copy()
            draw = ImageDraw.Draw(img_draw)
            
            for box in boxes:
                # Koordinat Box Wajah
                x1, y1, x2, y2 = box
                
                # Crop Wajah
                # Tambah sedikit margin biar dagu/rambut tidak terpotong (opsional)
                face_crop = image.crop((x1, y1, x2, y2))
                
                # 2. PENGENALAN WAJAH (INCEPTION RESNET)
                input_tensor = preprocess_image(face_crop)
                
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    top_p, top_class = probs.topk(1, dim=1)
                    
                    prob = top_p.item()
                    name_idx = top_class.item()
                    predicted_name = class_names[name_idx]
                
                # Logika Treshold (Biar tidak asal nebak orang asing)
                threshold = 0.60 # Kalau yakin di atas 60%
                
                if prob > threshold:
                    label = f"{predicted_name} ({prob*100:.1f}%)"
                    color = "green"
                else:
                    label = "Unknown / Tidak Yakin"
                    color = "red"
                
                # Gambar Kotak dan Nama
                draw.rectangle(box.tolist(), outline=color, width=3)
                # Tampilkan teks agak kasar di PIL (opsional bisa dipercantik)
                # Untuk demo sederhana, kita tampilkan di bawah gambar saja biar rapi
                
                # Tampilkan Potongan Wajah & Hasil di UI
                col1, col2 = st.columns([1, 3])
                with col1:
                    st.image(face_crop, width=100)
                with col2:
                    st.markdown(f"### Prediksi: **{label}**")
                    if prob > threshold:
                        st.info("✅ Presensi Berhasil Dicatat")
                    else:
                        st.warning("⚠️ Wajah tidak dikenali dengan cukup yakin.")
                st.divider()
            
            # Tampilkan gambar akhir dengan kotak
            # st.image(img_draw, caption='Hasil Deteksi', use_column_width=True)
            
        else:
            st.warning("Wajah tidak terdeteksi dalam foto ini.")

# Footer
st.markdown("---")
st.caption("Tugas Besar Deep Learning - InceptionResnetV1 + MTCNN")