import streamlit as st
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
from PIL import Image, ImageDraw
import torchvision.transforms as transforms
import numpy as np
import os

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Presensi Mahasiswa dengan Face Recognition",
    page_icon="🎓",
    layout="centered"
)

# --- LOAD MODEL & RESOURCE ---
@st.cache_resource
def load_resources():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # MTCNN dengan threshold lebih ketat untuk mengurangi false positive
    mtcnn = MTCNN(
        keep_all=False,
        device=device,
        min_face_size=40,  # Naikkan dari 20 ke 40 pixels
        thresholds=[0.7, 0.8, 0.9],  # Lebih ketat dari default [0.6, 0.7, 0.7]
        post_process=True  # Aktifkan post-processing
    )
    
    resnet = InceptionResnetV1(
        classify=True,
        pretrained='vggface2',
        num_classes=70
    )
    
    model_path = 'Akurasi_8714_InceptionResnet_best.pth'
    if not os.path.exists(model_path):
        return None, None, None, None
    
    try:
        checkpoint = torch.load(model_path, map_location=device)
        resnet.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint['class_names']
        resnet.to(device)
        resnet.eval()
        return mtcnn, resnet, class_names, device
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None

mtcnn, model, class_names, device = load_resources()

# --- UTILS: PREPROCESSING (REVISED) ---
def detect_face_with_rotation(image, mtcnn):
    """
    Coba deteksi wajah dengan berbagai rotasi jika deteksi normal gagal.
    Returns: (boxes, probs, rotation_angle)
    """
    # Coba deteksi normal dulu
    boxes, probs = mtcnn.detect(image)
    if boxes is not None:
        return boxes, probs, 0
    
    # Jika gagal, coba dengan rotasi 90, 180, 270 derajat
    for angle in [90, 180, 270]:
        rotated = image.rotate(angle, expand=True)
        boxes, probs = mtcnn.detect(rotated)
        if boxes is not None:
            return boxes, probs, angle
    
    # Jika semua gagal
    return None, None, 0

def adjust_box_coordinates(box, angle, original_size, rotated_size):
    """
    Konversi koordinat bounding box dari gambar yang dirotasi 
    kembali ke koordinat gambar asli.
    """
    x1, y1, x2, y2 = box
    w_orig, h_orig = original_size
    w_rot, h_rot = rotated_size
    
    if angle == 90:
        # 90 derajat clockwise: (x,y) -> (h-y, x)
        x1_new = h_rot - y2
        y1_new = x1
        x2_new = h_rot - y1
        y2_new = x2
        return [x1_new, y1_new, x2_new, y2_new]
    
    elif angle == 180:
        # 180 derajat: (x,y) -> (w-x, h-y)
        x1_new = w_rot - x2
        y1_new = h_rot - y2
        x2_new = w_rot - x1
        y2_new = h_rot - y1
        return [x1_new, y1_new, x2_new, y2_new]
    
    elif angle == 270:
        # 270 derajat clockwise: (x,y) -> (y, w-x)
        x1_new = y1
        y1_new = w_rot - x2
        x2_new = y2
        y2_new = w_rot - x1
        return [x1_new, y1_new, x2_new, y2_new]
    
    else:
        return box

def custom_crop_face(image, box):
    """
    Logika crop yang sama persis dengan data_preprocessing.ipynb:
    1. Hitung center point (cx, cy)
    2. Ambil sisi terpanjang * 1.2 untuk margin 20%
    3. Crop berbentuk square dari center
    """
    w_img, h_img = image.size
    x1, y1, x2, y2 = box
    
    # Hitung center dan side (sama dengan preprocessing)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * 1.2  # 20% margin
    
    # Koordinat baru dengan margin
    x1_new = int(max(cx - side / 2, 0))
    y1_new = int(max(cy - side / 2, 0))
    x2_new = int(min(cx + side / 2, w_img))
    y2_new = int(min(cy + side / 2, h_img))
    
    # Crop
    face_crop = image.crop((x1_new, y1_new, x2_new, y2_new))
    return face_crop, (x1_new, y1_new, x2_new, y2_new)

def preprocess_tensor(face_pil):
    # Resize manual ke 224x224 (BILINEAR) - sama dengan preprocessing
    face_resized = face_pil.resize((224, 224), Image.BILINEAR)
    
    # Transform ke Tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        fixed_image_standardization
    ])
    return transform(face_resized).unsqueeze(0).to(device)

# --- UI ---
st.title("Presensi Mahasiswa dengan Face Recognition")
st.write("Aplikasi ini adalah sistem presensi berbasis face recognition yang dibangun dengan Streamlit dan PyTorch. Gambar dari upload atau kamera diproses menggunakan MTCNN yang telah diperketat dan mampu mendeteksi wajah pada berbagai rotasi. Setelah wajah ditemukan, sistem melakukan crop dengan logika preprocessing training (center crop + margin 20%), mengubahnya menjadi ukuran 224×224, lalu menormalisasi dan mengonversinya menjadi tensor. Hasil crop kemudian diprediksi menggunakan model InceptionResnetV1 yang telah di-fine-tune pada data mahasiswa, menghasilkan identitas dan tingkat kepercayaan melalui softmax. Antarmuka menampilkan bounding box, wajah crop, nama yang teridentifikasi, confidence, serta Top-5 prediksi. Sistem ini dikembangkan oleh Freddy Harahap dan Dwi Arthur Revangga.")

if model is None:
    st.warning("⚠️ Model belum ditemukan. Upload file .pth dulu.")
else:
    # --- PILIHAN INPUT ---
    tab1, tab2 = st.tabs(["📂 Upload File", "📸 Ambil Foto"])
    
    image_source = None
    
    with tab1:
        uploaded_file = st.file_uploader("Pilih Foto...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None: 
            image_source = uploaded_file
    
    with tab2:
        camera_file = st.camera_input("Ambil Foto Wajah")
        if camera_file is not None: 
            image_source = camera_file
    
    # --- PROSES GAMBAR ---
    if image_source is not None:
        try:
            image = Image.open(image_source).convert('RGB')
            original_size = image.size
            
            if image_source == uploaded_file:
                st.image(image, caption='Foto Input', use_container_width=True)
            
            with st.spinner('Mendeteksi...'):
                # Deteksi dengan multi-rotasi
                boxes, probs, rotation_angle = detect_face_with_rotation(image, mtcnn)
                
                if boxes is not None:
                    if rotation_angle > 0:
                        st.info(f"ℹ️ Wajah terdeteksi setelah rotasi {rotation_angle}°")
                        # Rotate image untuk processing
                        image_rotated = image.rotate(rotation_angle, expand=True)
                    else:
                        st.success(f"Wajah terdeteksi!")
                        image_rotated = image
                    
                    # Ambil wajah pertama (karena keep_all=False, hanya ada 1)
                    box = boxes[0]
                    
                    # 1. CROP dari gambar yang sudah dirotasi (jika perlu)
                    face_crop, crop_coords = custom_crop_face(image_rotated, box)
                    
                    # 2. PREPROCESS
                    input_tensor = preprocess_tensor(face_crop)
                    
                    # 3. PREDIKSI
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        probs_model = torch.nn.functional.softmax(outputs, dim=1)
                        top_probs, top_classes = probs_model.topk(5, dim=1)
                        
                        best_prob = top_probs[0][0].item()
                        best_idx = top_classes[0][0].item()
                        best_name = class_names[best_idx]
                    
                    # Visualisasi pada gambar yang sudah dirotasi
                    img_draw = image_rotated.copy()
                    draw = ImageDraw.Draw(img_draw)
                    
                    x1, y1, x2, y2 = crop_coords
                    
                    if best_prob > 0.50:
                        color = "green"
                        icon = "✅"
                    else:
                        color = "red"
                        icon = "⚠️"
                    
                    draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
                    
                    # Tampilkan gambar dengan bounding box
                    st.image(img_draw, caption=f'Hasil Deteksi (Rotasi: {rotation_angle}°)', use_container_width=True)
                    
                    # Tampilan Kolom
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.image(face_crop, width=150, caption=f"Crop: 224x224")
                    
                    with col2:
                        st.markdown(f"### {icon} {best_name}")
                        st.progress(best_prob)
                        st.caption(f"Confidence: {best_prob*100:.1f}%")
                        
                        with st.expander("Lihat Top 5"):
                            for i in range(5):
                                p = top_probs[0][i].item()
                                n = class_names[top_classes[0][i].item()]
                                st.write(f"- {n}: {p*100:.1f}%")
                    
                else:
                    st.error("Wajah tidak terdeteksi oleh MTCNN.")
                    st.info("Tips: Pastikan wajah terlihat jelas, pencahayaan cukup, dan tidak terlalu miring.")
                    
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses gambar: {e}")
            st.exception(e)  # Tampilkan detail error untuk debugging
