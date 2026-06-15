import cv2
import streamlit as st
from PIL import Image, UnidentifiedImageError, ImageOps
from ultralytics import YOLO
from fpdf import FPDF
import tempfile, gdown, os, json, io, datetime
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ================= UI =================
st.set_page_config(page_title="CoVision", layout="centered")

def force_rerun():
    if hasattr(st, "rerun"):
        st.rerun()
    else:
        st.experimental_rerun()

# ================= USER =================
USER_FILE = "users.json"

def load_users():
    return json.load(open(USER_FILE)) if os.path.exists(USER_FILE) else {}

def save_users(u):
    json.dump(u, open(USER_FILE, "w"))

users = load_users()

defaults = {
    "logged_in": False,
    "page": "login",
    "username": "",
    "model": None,
    "label_names": {},
    "sub_page": "Deteksi"
}

for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ================= MODEL LOADER (FIX UTAMA) =================
MODEL_URL  = "https://drive.google.com/uc?id=1LVH621YUKJO5XPT4tXkX0hvNj-HxbQYl"
MODEL_PATH = "best_kopi.pt"

@st.cache_resource
def load_model():
    try:
        # cek file
        if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
            st.warning("⚠️ Model tidak valid, download ulang...")
            gdown.download(MODEL_URL, MODEL_PATH, quiet=False, fuzzy=True)

        st.write("📦 Model size:", os.path.getsize(MODEL_PATH))

        model = YOLO(MODEL_PATH)
        return model, model.names

    except Exception as e:
        st.error("❌ Model custom gagal, pakai fallback YOLO bawaan")
        st.code(str(e))

        model = YOLO("yolov8n.pt")
        return model, model.names

# ================= AUTH =================
def signup():
    st.title("Daftar Akun")
    u = st.text_input("Username Baru")
    p = st.text_input("Password", type="password")

    if st.button("Daftar"):
        if u in users:
            st.error("Username sudah ada.")
        elif not u or not p:
            st.warning("Isi semua field.")
        else:
            users[u] = p
            save_users(users)
            st.success("Berhasil daftar!")
            st.session_state.page = "login"
            force_rerun()

def login():
    st.title("Login CoVision")
    u = st.text_input("Username")
    p = st.text_input("Password", type="password")

    if st.button("Login"):
        if u in users and users[u] == p:
            st.session_state.update(logged_in=True, username=u, page="main")
            force_rerun()
        else:
            st.error("Login gagal.")

# ================= DETECT =================
def detect_page():
    st.title("Deteksi Kopi")

    if st.session_state.model is None:
        st.session_state.model, st.session_state.label_names = load_model()

    model = st.session_state.model

    metode = st.radio("Metode", ["Upload", "Webcam"])

    if metode == "Upload":
        files = st.file_uploader("Upload", accept_multiple_files=True)

        if files:
            pdf = FPDF()

            for f in files:
                img = Image.open(f).convert("RGB")
                img = ImageOps.exif_transpose(img)

                st.image(img)

                with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
                    img.save(tf.name)
                    path = tf.name

                r = model(path)[0]
                annotated = Image.fromarray(r.plot()[..., ::-1])

                st.image(annotated)

                os.remove(path)

    else:
        webcam_detect_page()

# ================= WEBCAM (OPTIMIZED) =================
def webcam_detect_page():
    st.header("Webcam Detection")

    model = st.session_state.model

    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")

            # 🔥 langsung infer (NO FILE)
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

# ================= MAIN =================
def main_app():
    with st.sidebar:
        st.write("👤", st.session_state.username)
        page = st.radio("Menu", ["Deteksi"])

        if st.button("Logout"):
            st.session_state.update(logged_in=False, page="login")
            force_rerun()

    if page == "Deteksi":
        detect_page()

# ================= ROUTER =================
if st.session_state.page == "signup":
    signup()
elif not st.session_state.logged_in:
    login()
else:
    main_app()
