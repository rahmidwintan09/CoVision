import cv2
import streamlit as st
from PIL import Image, UnidentifiedImageError, ImageOps
import numpy as np
from ultralytics import YOLO
from fpdf import FPDF
import tempfile, gdown, os, json, io, datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av


st.markdown(
    """
    <style id="auto-theme">
    @media (prefers-color-scheme: dark) {
      :root {
        --primary-color:#ff6347;
        --text-color:#eeeeee;
        --background-color:#1e1e1e;
        --secondary-background-color:#262730;
      }
    }
    @media (prefers-color-scheme: light), (prefers-color-scheme: no-preference) {
      :root {
        --primary-color:#d13b0c;
        --text-color:#000000;
        --background-color:#ffffff;
        --secondary-background-color:#fdfdf5;
      }
    }
    body, .stApp {
      background-color: var(--background-color);
      color: var(--text-color);
    }
    input, textarea, .stTextInput > div > div, .stPasswordInput > div > div,
    .stButton > button {
      background-color: var(--secondary-background-color) !important;
      color: var(--text-color) !important;
      border: 1px solid #ccc;
    }
    .stButton > button:hover {
      background-color: #ecebe1 !important;
      color: var(--text-color) !important;
    }
    ::placeholder { color:#666 !important; opacity:1; }
    </style>
    """,
    unsafe_allow_html=True,
)


def force_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

@st.cache_resource
def load_model():
    MODEL_PATH = "best_kopi.pt"
    # download model jika belum ada
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1LVH621YUKJO5XPT4tXkX0hvNj-HxbQYl"
        gdown.download(url, MODEL_PATH, quiet=False)
    
    # load model YOLO
    model = YOLO(MODEL_PATH)
    label_names = model.names
    return model, label_names

# Set konfigurasi halaman di awal
st.set_page_config(page_title="CoVision: Deteksi Tingkat Kematangan Buah Kopi", layout="centered")

# ================= MANAJEMEN PENGGUNA =================
USER_FILE = "users.json"
def load_users():
    return json.load(open(USER_FILE)) if os.path.exists(USER_FILE) else {}
def save_users(u): 
    json.dump(u, open(USER_FILE, "w"))

users = load_users()

# Inisialisasi model langsung ke session state di awal agar tidak None
if "model" not in st.session_state or st.session_state.model is None:
    with st.spinner("Memuat Model AI, mohon tunggu..."):
        st.session_state.model, st.session_state.label_names = load_model()

defaults = { 
    "logged_in": False, 
    "page": "login", 
    "username": "",
    "sub_page": "Deteksi" 
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ================= HALAMAN AUTENTIKASI =================
def signup():
    st.title("Daftar Akun")
    u = st.text_input("Username Baru")
    p = st.text_input("Password", type="password")
    if st.button("Daftar"):
        if u in users:
            st.error("Username sudah ada.")
        elif not u or not p:
            st.warning("Username / Password kosong.")
        else:
            users[u] = p
            save_users(users)
            st.success("Berhasil daftar, silakan login.")
            st.session_state.page = "login"
            force_rerun()
    st.button("Kembali ke Login", on_click=lambda: st.session_state.update(page="login"))

def login():
    st.title("Login CoVision")
    u = st.text_input("Username", key="username_input")
    p = st.text_input("Password", type="password", key="password_input")
    if st.button("Login", key="login_button"):
        if u in users and users[u] == p:
            st.session_state.update(logged_in=True, username=u, page="main")
            force_rerun()
        else:
            st.error("Username / Password salah.")
    st.button("Belum punya akun? Daftar", key="signup_button", on_click=lambda: st.session_state.update(page="signup"))

# ================= HALAMAN INFORMASI =================
def about_page():
    st.title("Tingkat Kematangan Buah Kopi")
    st.write("""
    Kematangan buah kopi merupakan indikator penting dalam penentuan kualitas, rasa, serta waktu panen dan distribusi. Berikut adalah tiga kategori utama tingkat kematangan buah kopi yang digunakan dalam aplikasi CoVision untuk deteksi otomatis:
    """)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.image("https://raw.githubusercontent.com/rahmidwintan09/CoVision/50e0f52a5a238eb3735c3e2d3b407113fa27fa5a/images/matang.jpg", caption="Matang", use_container_width=True)
        st.markdown("""
        **Matang (Grade A)** - Warna merah merata  
        - Siap untuk didistribusikan
        """)

    with col2:
        st.image("https://raw.githubusercontent.com/rahmidwintan09/CoVision/50e0f52a5a238eb3735c3e2d3b407113fa27fa5a/images/setengah_matang.jpg", caption="Setengah Matang", use_container_width=True)
        st.markdown("""
        **Setengah Matang (Grade B)** - Warna kuning  
        - Masih keras sebagian  
        - Belum siap didistribusikan, cocok untuk pematangan lanjutan
        """)

    with col3:
        st.image("https://raw.githubusercontent.com/rahmidwintan09/CoVision/50e0f52a5a238eb3735c3e2d3b407113fa27fa5a/images/mentah.jpg", caption="Mentah", use_container_width=True)
        st.markdown("""
        **Mentah (Grade C)** - Warna hijau 
        - Tekstur keras  
        """)

    st.write("---")
    st.info("Klasifikasi ini digunakan sebagai dasar untuk deteksi otomatis tingkat kematangan buah kopi dalam aplikasi CoVision.")

# ================= HALAMAN DETEKSI =================
def detect_page():
    st.title("CoVision: Deteksi Tingkat Kematangan Buah Kopi")
    st.caption("Deteksi Kopi Sekarang!")
    
    model = st.session_state.model
    metode = st.radio("Pilih Metode Deteksi", ["Upload Gambar", "Deteksi Via Webcam"])
    
    if metode == "Upload Gambar":
        files = st.file_uploader("Upload Gambar Kopi", accept_multiple_files=True, type=["jpg", "png", "jpeg"])
        if files:
            for f in files:
                img = Image.open(f).convert("RGB")
                img = ImageOps.exif_transpose(img)
                st.image(img, caption="Gambar Asli", use_container_width=True)
                
                img_np = np.array(img)
                
                # Proses deteksi YOLO
                with st.spinner("Mendeteksi..."):
                    r = model(img_np)[0]
                    
                annotated = Image.fromarray(r.plot()[..., ::-1])
                st.image(annotated, caption="Hasil Deteksi", use_container_width=True)
    else:
        webcam_detect_page()

# ================= WEBCAM (OPTIMIZED) =================
def webcam_detect_page():
    st.header("Webcam Real-Time Detection")
    model = st.session_state.model

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            results = model(img)[0]
            annotated = results.plot()
            annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            return av.VideoFrame.from_ndarray(annotated_rgb, format="rgb24")

    webrtc_streamer(
        key="webcam",
        video_processor_factory=VideoProcessor,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": True, "audio": False}
    )


def main_app():
    with st.sidebar:
        st.markdown(f"Username")
        st.markdown(f"👤 **{st.session_state.username}**")
        st.session_state.sub_page = st.radio("Menu", ["Deteksi", "Tentang Kopi"])
        if st.button("Logout"):
            st.session_state.update(logged_in=False, page="login", username="")
            force_rerun()
            
    if st.session_state.sub_page == "Tentang Kopi":
        about_page()
    else:
        detect_page()

# ================= AlUR KONTROL HALAMAN =================
if st.session_state.page == "signup":
    signup()
elif not st.session_state.logged_in:
    login()
elif st.session_state.page == "main":
    main_app()
else:
    st.session_state.page = "login"
    login()
