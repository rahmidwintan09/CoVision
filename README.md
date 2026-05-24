# CoVision

Aplikasi berbasis web untuk mendeteksi **kualitas buah kopi** (Grade A, B, C) menggunakan model **YOLOv12** dan dibangun dengan framework **Streamlit**.  
Model ini dilatih melalui dataset morfologi fisik buah kopidan di-deploy agar dapat digunakan dengan mudah oleh pengguna.

---

## 🚀 Demo Online
Klik link berikut untuk mencoba aplikasinya secara langsung:  
[🌐 Coba Aplikasinya di Streamlit Cloud](link web)

---

## 📦 Fitur Utama

- 🔍 Deteksi kualitas kopi(Grade A, B, atau C)
- 🖼️ Upload gambar kopidari file
- 🤖 Menggunakan model YOLOv8 hasil fine-tuning
- 📊 Hasil ditampilkan langsung dengan anotasi visual

---

## 🧠 Teknologi yang Digunakan

- Python 3
- YOLOv12 (`ultralytics`)
- Streamlit
- Pillow
- gdown (untuk ambil model dari Google Drive)

---

## Struktur File
   ├── app.py # Streamlit app
   
   ├── requirements.txt # Daftar dependensi
   
   ├── best_fine_tune.pt # Model YOLOv8 (via GDrive)


---

## ⚙️ Cara Menjalankan (Lokal)

1. Clone repo ini:
   ```bash
   git clone https://github.com/USERNAME/NAMA-REPO.git
   cd NAMA-REPO

2. Install Dependensi
   ```bash
   pip install -r requirements.txt

4. Jalankan Aplikasi
   ```bash
   streamlit run app.py

## Dataset
Model dilatih menggunakan dataset deteksi buah kopiyang telah dikurasi berdasarkan kriteria morfologi kualitas.
Untuk keperluan pengujian dan evaluasi, model ini di-fine-tune agar lebih akurat mendeteksi kualitas secara visual.

## Kontributor
Proyek ini dikembangkan oleh Rahmi Dwi Intan, mahasiswi Informatika, dengan fokus pada penerapan Computer Vision di bidang pertanian cerdas.
