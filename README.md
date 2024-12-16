# Project-Pengolahan-Citra
Aplikasi ini adalah proyek berbasis Python yang memungkinkan pengguna untuk menerapkan berbagai transformasi pada gambar dan video secara real-time. Dibangun dengan Streamlit, aplikasi ini memiliki antarmuka pengguna yang intuitif dan mudah digunakan.

## Fitur Utama

### 1. Transformasi Gambar
- **Grayscale**: Mengubah gambar menjadi skala keabuan.
- **Negative**: Membalikkan warna gambar.
- **Thresholding**: Menyaring gambar berdasarkan nilai ambang batas.
- **Brightness**: Menyesuaikan kecerahan gambar.
- **Contrast**: Menyesuaikan kontras gambar.
- **Power Law Transformation**: Menerapkan transformasi gamma.
- **Log Transformation**: Menggunakan fungsi logaritma untuk meningkatkan kontras.
- **Color Filtering**: Menyaring gambar berdasarkan saluran warna (merah, hijau, atau biru).

### 2. Transformasi Real-Time (Kamera)
- Menangkap video dari kamera perangkat.
- Menerapkan transformasi secara langsung pada frame video.
- Menampilkan histogram untuk distribusi intensitas gambar secara real-time.

## Jalankan aplikasi:
   ```bash
   streamlit run app.py
   ```
