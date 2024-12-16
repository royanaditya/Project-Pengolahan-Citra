import streamlit as st
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
import cv2

cap = cv2.VideoCapture(0)  # Ganti 0 dengan indeks lain jika kamera tambahan ada
if not cap.isOpened():
    print("Tidak dapat membuka kamera.")
else:
    print("Kamera berhasil dibuka.")
    cap.release()

# Function to calculate histogram
def calculate_histogram(image):
    grayscale_image = ImageOps.grayscale(image)
    histogram, _ = np.histogram(np.array(grayscale_image).flatten(), bins=256, range=(0, 255))
    return histogram

def calculate_frame_histogram(frame):
    if len(frame.shape) == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame
    histogram = cv2.calcHist([gray_frame], [0], None, [256], [0, 256])
    return histogram.flatten()

# Function for power law transformation
def power_law_transform(image, gamma):
    img_array = np.array(image).astype(float)
    img_array = img_array / 255.0
    img_array = np.power(img_array, gamma)
    img_array = (img_array * 255).astype(np.uint8)
    return Image.fromarray(img_array)

# Function for log transformation
def log_transform(image):
    # Konversi gambar ke array numpy
    img_array = np.array(image, dtype=np.float32)
    
    # Normalisasi gambar ke rentang 0-1
    img_normalized = img_array / 255.0
    
    # Tambahkan 1 untuk menghindari log(0)
    img_log = np.log1p(img_normalized)
    
    # Skala ulang ke rentang 0-255
    c = 255.0 / np.log1p(1.0)  # normalisasi
    log_transformed = c * img_log
    
    # Konversi kembali ke tipe uint8
    log_transformed = np.clip(log_transformed, 0, 255).astype(np.uint8)
    
    return Image.fromarray(log_transformed)

# Function to apply transformations
def transform_image(image, method, params=None):
    # Convert RGBA to RGB if necessary
    if image.mode == 'RGBA':
        # Create a white background
        background = Image.new('RGB', image.size, (255, 255, 255))
        # Paste the image on the background using alpha channel
        background.paste(image, mask=image.split()[3])
        image = background

    if method == "Grayscale":
        return ImageOps.grayscale(image)
    elif method == "Negative":
        if image.mode == 'RGB':
            return ImageOps.invert(image)
        else:
            # Convert to RGB if it's not already
            return ImageOps.invert(image.convert('RGB'))
    elif method == "Thresholding":
        threshold = params.get('threshold', 128)
        grayscale_image = ImageOps.grayscale(image)
        thresholded = np.where(np.array(grayscale_image) < threshold, 0, 255).astype('uint8')
        return Image.fromarray(thresholded)
    elif method == "Brightness":
        factor = params.get('brightness', 1.0)
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)
    elif method == "Contrast":
        factor = params.get('contrast', 1.0)
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)
    elif method == "Power Law":
        gamma = params.get('gamma', 1.0)
        return power_law_transform(image, gamma)
    elif method == "Log Transform":
        return log_transform(image)
    elif method == "Color Filtering":
        color = params.get('color', 'red')
        img_array = np.array(image)
        if color == "red":
            img_array[:, :, [1, 2]] = 0
        elif color == "green":
            img_array[:, :, [0, 2]] = 0
        elif color == "blue":
            img_array[:, :, [0, 1]] = 0
        return Image.fromarray(img_array)
    else:
        return image
    
def transform_frame(frame, method, params=None):
    if method == "Grayscale":
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    elif method == "Negative":
        return cv2.bitwise_not(frame)
    elif method == "Thresholding":
        _, thresholded = cv2.threshold(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 
                                        params.get('threshold', 128), 255, cv2.THRESH_BINARY)
        return thresholded
    elif method == "Brightness":
        return cv2.convertScaleAbs(frame, alpha=1, beta=params.get('brightness', 50))
    elif method == "Contrast":
        alpha = params.get('contrast', 1.0)
        return cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
    elif method == "Power Law":
        gamma = params.get('gamma', 1.0)
        img_array = np.float32(frame) / 255.0
        img_array = np.power(img_array, gamma)
        return np.uint8(img_array * 255)
    elif method == "Log Transform":
        c = 255 / np.log(1 + np.max(frame))
        log_transformed = c * (np.log(frame + 1))
        return np.array(log_transformed, dtype=np.uint8)
    elif method == "Color Filtering":
        filtered_frame = frame.copy()
        color = params.get('color', 'red')
        if color == "red":
            filtered_frame[:, :, :2] = 0            
        elif color == "green":
            filtered_frame[:, :, [0, 2]] = 0
        elif color == "blue":
            filtered_frame[:, :, 1:] = 0
        return filtered_frame
    return frame


uploaded_file = None  # Initialize uploaded_file

st.sidebar.title("Menu")
menu = st.sidebar.selectbox("Pilih Menu", ["Transform Gambar", "Transform Realtime"])

if menu == "Transform Gambar":
    # Title and instructions
    st.title("Aplikasi Pengolahan Citra")
    st.write("Unggah gambar dan pilih transformasi yang diinginkan.")

    # File uploader
    uploaded_file = st.file_uploader("Unggah gambar (JPEG atau PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and display original image
    original_image = Image.open(uploaded_file)
    # Select transformation
    transformation_method = st.selectbox(
        "Pilih Transformasi",
        ["Grayscale", "Negative", "Thresholding", "Brightness", "Contrast", 
         "Power Law", "Log Transform", "Color Filtering"]
    )

    # Parameter controls based on selected transformation
    params = {}
    if transformation_method == "Thresholding":
        params['threshold'] = st.slider("Nilai Threshold", 0, 255, 128)
    elif transformation_method == "Brightness":
        params['brightness'] = st.slider("Brightness Factor", 0.0, 3.0, 1.0, 0.1)
    elif transformation_method == "Contrast":
        params['contrast'] = st.slider("Contrast Factor", 0.0, 3.0, 1.0, 0.1)
    elif transformation_method == "Power Law":
        params['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
    elif transformation_method == "Color Filtering":
        params['color'] = st.selectbox("Pilih Channel Warna", ["red", "green", "blue"])

    # Apply transformation
    result_image = transform_image(original_image, transformation_method, params)

    # Create columns for layout
    col1, col2 = st.columns(2)

    # Display original image and histogram
    with col1:
        st.subheader("Gambar Asli")
        st.image(original_image, caption="Gambar Asli", use_column_width=True)

        st.subheader("Histogram Gambar Asli")
        original_histogram = calculate_histogram(original_image)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(256), original_histogram, color="gray")
        ax.set_title("Histogram Distribusi Nilai Keabuan - Gambar Asli")
        ax.set_xlabel("Intensitas")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)

    # Display transformed image and histogram
    with col2:
        st.subheader("Hasil Transformasi")
        st.image(result_image, caption="Hasil Transformasi", use_column_width=True)

        st.subheader("Histogram Hasil Transformasi")
        result_histogram = calculate_histogram(result_image)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(range(256), result_histogram, color="gray")
        ax.set_title("Histogram Distribusi Nilai Keabuan - Hasil Transformasi")
        ax.set_xlabel("Intensitas")
        ax.set_ylabel("Frekuensi")
        st.pyplot(fig)
elif menu == "Transform Realtime":
    st.title("Transformasi Real-Time (Kamera)")
    run = st.checkbox("Jalankan Kamera")
    
    # Create placeholders for the video frame and histogram
    frame_placeholder = st.empty()
    col1, col2 = st.columns(2)
    hist_placeholder = col2.empty()
    
    transformation_method = st.selectbox(
        "Pilih Transformasi",
        ["Grayscale", "Negative", "Thresholding", "Brightness", "Contrast", 
         "Power Law", "Log Transform", "Color Filtering"]
    )
    
    params = {}
    if transformation_method == "Thresholding":
        params['threshold'] = st.slider("Nilai Threshold", 0, 255, 128)
    elif transformation_method == "Brightness":
        params['brightness'] = st.slider("Brightness Factor", 0, 100, 50)
    elif transformation_method == "Contrast":
        params['contrast'] = st.slider("Contrast Factor", 1.0, 3.0, 1.0, 0.1)
    elif transformation_method == "Power Law":
        params['gamma'] = st.slider("Gamma", 0.1, 3.0, 1.0, 0.1)
    elif transformation_method == "Color Filtering":
        params['color'] = st.selectbox("Pilih Channel Warna", ["red", "green", "blue"])

    if run:
        cap = cv2.VideoCapture(0)
        while run:
            ret, frame = cap.read()
            if not ret:
                st.write("Tidak dapat mengakses kamera.")
                break

            # Transform the frame
            transformed_frame = transform_frame(frame, transformation_method, params)
            
            # Display the frame in the first column
            with col1:
                if len(transformed_frame.shape) == 2:  # Grayscale image
                    frame_placeholder.image(transformed_frame, caption="Live Camera Feed")
                else:  # Color image
                    frame_placeholder.image(
                        cv2.cvtColor(transformed_frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        caption="Live Camera Feed"
                    )
            
            # Calculate and display histogram in the second column
            histogram = calculate_frame_histogram(transformed_frame)
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(range(256), histogram, color="gray")
            ax.set_title("Histogram - Live Feed")
            ax.set_xlabel("Intensitas")
            ax.set_ylabel("Frekuensi")
            hist_placeholder.pyplot(fig)
            plt.close(fig)
            
        cap.release()