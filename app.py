import streamlit as st    
st.set_page_config(page_title="CoVision: Deteksi Tingkat Kematangan Buah Kopi", layout="centered")

from PIL import Image, UnidentifiedImageError, ExifTags, ImageOps
from ultralytics import YOLO
from fpdf import FPDF
import tempfile, gdown, os, json, io, datetime, cv2
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

USER_FILE = "users.json"
def load_users():
    return json.load(open(USER_FILE)) if os.path.exists(USER_FILE) else {}
def save_users(u): json.dump(u, open(USER_FILE, "w"))

users = load_users()
defaults = { "logged_in": False, "page": "login", "username": "",
             "model": None, "label_names": {}, "sub_page": "Deteksi" }
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

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


def about_page():
    st.title("Tingkat Kematangan Buah Kopi")
    st.write("""
    Kematangan buah kopi merupakan indikator penting dalam penentuan kualitas, rasa, serta waktu panen dan distribusi. Berikut adalah tiga kategori utama tingkat kematangan buah kopi yang digunakan dalam aplikasi CoVision untuk deteksi otomatis:
    """)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    col1, col2, col3 = st.columns(3)

    gambar_data = [
        {"col": col1, "name": "matang.jpg", "label": "Matang (Grade A)", "desc": "- Warna merah merata\n- Siap didistribusikan"},
        {"col": col2, "name": "setengah_matang.jpg", "label": "Setengah Matang (Grade B)", "desc": "- Warna kuning\n- Masih keras sebagian\n- Cocok untuk pematangan lanjutan"},
        {"col": col3, "name": "mentah.jpg", "label": "Mentah (Grade C)", "desc": "- Warna hijau\n- Tekstur keras"}
    ]

    for item in gambar_data:
        with item["col"]:
            img_path = os.path.join(BASE_DIR, "images", item["name"])
            
            # Cek apakah file ada secara fisik
            if os.path.exists(img_path):
                try:
                    # Buka gambar dan PAKSA load pikselnya ke memori + konversi ke RGB
                    with Image.open(img_path) as loaded_img:
                        img_rgb = loaded_img.convert("RGB")
                        img_rgb.load()  # Memaksa pembacaan seluruh data piksel gambar
                    
                    # Tampilkan gambar yang sudah dijamin bertipe RGB murni ke Streamlit
                    st.image(img_rgb, caption=item["label"], use_container_width=True)
                    
                except Exception as e:
                    # Menangkap segala jenis error pembacaan data piksel / tipe data rusak
                    st.error(f"⚠️ Gambar `{item['name']}` tidak dapat diproses.")
                    st.caption(f"Detail error internal: {str(e)}")
            else:
                st.warning(f"❌ File `{item['name']}` tidak ditemukan di folder `images/`.")
                
            st.markdown(f"**{item['label']}**\n{item['desc']}")

    st.write("---")
    st.info("Klasifikasi ini digunakan sebagai dasar untuk deteksi otomatis tingkat kematangan buah kopi dalam aplikasi CoVision.")
    
def upload_image_detect_page():
    uploaded_files = st.file_uploader("Upload Gambar Kopi", accept_multiple_files=True, type=["jpg", "jpeg", "png"])
    st.session_state.uploaded_files = uploaded_files or []

def detect_page():
    st.title("CoVision: Deteksi Tingkat Kematangan Buah Kopi")
    st.caption("Deteksi Kopi Sekarang!")

    MODEL_URL  = "https://drive.google.com/file/d/1LVH621YUKJO5XPT4tXkX0hvNj-HxbQYl/view?usp=sharing"
    MODEL_PATH = "best_kopi.pt"

    if st.session_state.model is None:
        if not os.path.exists(MODEL_PATH):
            with st.spinner("Mengunduh model…"):
                gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
        st.session_state.model = YOLO(MODEL_PATH)
        st.session_state.label_names = st.session_state.model.names
    st.markdown("---")
    st.session_state.detection_method = st.radio("Pilih Metode Deteksi", ["Upload Gambar", "Deteksi via Webcam"],
        key="detection_method_radio"
    )
    st.markdown("---")
    if st.session_state.detection_method == "Upload Gambar":
        upload_image_detect_page()
    else:
        webcam_detect_page()

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)

    uploaded_files = st.session_state.get("uploaded_files", [])
    for idx, uploaded in enumerate(uploaded_files, 1):
        st.markdown(f"###  {uploaded.name}")

        try:
            img = Image.open(uploaded).convert("RGB")
            img = ImageOps.exif_transpose(img)
        except UnidentifiedImageError:
            st.error("Format tidak didukung."); continue
        st.image(img, caption="Gambar Asli", width=600)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            img.save(tf.name)
            temp_path = tf.name

        r = st.session_state.model(temp_path)[0]
        annotated = Image.fromarray(r.plot()[..., ::-1])
        st.image(annotated, caption="Hasil Deteksi", width=600)

        cls = [st.session_state.label_names[int(i)] for i in (r.boxes.cls.tolist() if r.boxes else [])]
        a, b, c = cls.count("A"), cls.count("B"), cls.count("C")
        col1, col2, col3 = st.columns(3)
        col1.metric("Grade A", a); col2.metric("Grade B", b); col3.metric("Grade C", c)

        buf = io.BytesIO()
        annotated.save(buf, format="JPEG")
        st.download_button(f"Download Hasil – {uploaded.name}",
                           buf.getvalue(), f"hasil_{uploaded.name}", "image/jpeg")

        pdf.add_page()
        pdf.set_font("Times", size=10)
        pdf.multi_cell(0, 8,
            f"[{idx}] {uploaded.name}\n"
            f"Grade A : {a}   Grade B : {b}   Grade C : {c}\n"
            f"Tanggal  : {datetime.datetime.now():%d/%m/%Y %H:%M}\n"
            f"Pengguna : {st.session_state.username}"
        )
        img_path = f"{temp_path}_annot.jpg"
        annotated.save(img_path)
        y_position = pdf.get_y() + 10
        pdf.image(img_path, x=20, y=y_position, w=170, h=140)
        os.remove(img_path)
        os.remove(temp_path)

    if uploaded_files:
        pdf_bytes = pdf.output(dest="S").encode("latin1")
        st.download_button("Download Semua Laporan (PDF)",
                           pdf_bytes, "laporan_covision.pdf", "application/pdf")

def webcam_detect_page():
    st.header("Deteksi kopi via Webcam (Real-Time)")
    st.write("Aktifkan webcam untuk mendeteksi kopi secara langsung melalui browser.")

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
        rtc_configuration={
            "iceServers": [
                {"urls": ["stun:stun.l.google.com:19302"]},
                {"urls": ["stun:global.stun.twilio.com:3478"]}
            ]
        },
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
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

if st.session_state.page == "signup":
    signup()
elif not st.session_state.logged_in:
    login()
elif st.session_state.page == "main":
    main_app()
else:
    st.session_state.page = "login"; login()
