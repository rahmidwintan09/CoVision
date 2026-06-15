import cv2
import streamlit as st
from PIL import Image, UnidentifiedImageError, ImageOps
import numpy as np
from ultralytics import YOLO
from fpdf import FPDF
import tempfile, gdown, os, json, io, datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ================= KUSTOMISASI TEMA =================
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

# ================= FUNGSI UTILITAS & LOAD MODEL =================
def force_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

@st.cache_resource
def load_model():
    MODEL_PATH = "best_kopi.pt"
    
    # Cek dan bersihkan jika file yang ada saat ini korup atau berupa berkas HTML palsu akibat limit kuota
    if os.path.exists(MODEL_PATH):
        is_html = False
        try:
            with open(MODEL_PATH, "r", encoding="utf-8", errors="ignore") as f:
                start_content = f.read(100)
                if "<html" in start_content.lower() or "<!doctype" in start_content.lower():
                    is_html = True
        except:
            pass

        # Jika berisi HTML atau ukurannya terlalu kecil untuk model YOLO, hapus berkas korup tersebut
        if is_html or os.path.getsize(MODEL_PATH) < 2000000:
            try:
                os.remove(MODEL_PATH)
            except:
                pass

    # Jalankan unduhan jika berkas belum ada atau baru saja dihapus karena korup
    if not os.path.exists(MODEL_PATH):
        url = "https://drive.google.com/uc?id=1LVH621YUKJO5XPT4tXkX0hvNj-HxbQYl"
        try:
            # Memperbaiki error: menghapus 'fuzzy=True' agar kompatibel dengan versi gdown di server
            gdown.download(url, MODEL_PATH, quiet=False)
        except Exception as e:
            st.error(f"Gagal mengunduh model dari Google Drive: {e}")
            st.warning("Tips: Google Drive mungkin membatasi unduhan otomatis karena batasan kuota IP server.")
            st.stop()
            
    # Validasi pasca-unduh: pastikan yang terunduh bukan file HTML teks peringatan dari Google
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "r", encoding="utf-8", errors="ignore") as f:
                start_content = f.read(100)
                if "<html" in start_content.lower() or "<!doctype" in start_content.lower():
                    st.error("Google Drive mengembalikan halaman limit kuota (HTML), bukan file model asli. Silakan hapus instans atau klik 'Reboot App' pada dashboard Streamlit beberapa saat lagi.")
                    try: os.remove(MODEL_PATH)
                    except: pass
                    st.stop()
        except:
            pass

    if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) == 0:
        st.error("File model kosong atau tidak ditemukan. Coba refresh halaman.")
        st.stop()

    model = YOLO(MODEL_PATH)
    label_names = model.names
    return model, label_names


# Konfigurasi halaman diletakkan di awal eksekusi utama streamlit
st.set_page_config(page_title="CoVision: Deteksi Tingkat Kematangan Buah Kopi", layout="centered")

# ================= MANAJEMEN USER =================
USER_FILE = "users.json"
def load_users():
    return json.load(open(USER_FILE)) if os.path.exists(USER_FILE) else {}
def save_users(u): 
    json.dump(u, open(USER_FILE, "w"))

users = load_users()

# Mengatasi bug inisialisasi awal session state
defaults = { 
    "logged_in": False, 
    "page": "login", 
    "username": "",
    "sub_page": "Deteksi" 
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# Load model dan simpan ke state JIKA belum ada atau masih None
if "model" not in st.session_state or st.session_state.model is None:
    with st.spinner("Memuat Model AI CoVision... Mohon tunggu"):
        st.session_state.model, st.session_state.label_names = load_model()

# ================= HALAMAN LOGIN & DAFTAR =================
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


# ================= HALAMAN UTAMA & KONTEN =
