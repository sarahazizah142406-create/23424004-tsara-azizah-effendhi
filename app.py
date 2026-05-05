"""
Aplikasi Streamlit untuk Color Detection pada gambar.
Mendukung berbagai color space dan metode thresholding.
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image

# Import modul lokal
from color_converters import (
    to_rgb, to_rg, to_normalized_rgb, 
    to_hsv, to_hsv_display, to_ycrcb, to_ycrcb_display, to_tsl
)
from threshold_methods import (
    static_threshold, dynamic_threshold, distance_threshold
)
from utils import load_image, apply_mask, create_mask_visualization, mask_to_png_bytes


# Konfigurasi halaman
st.set_page_config(
    page_title="Color Detection App",
    page_icon="🎨",
    layout="wide"
)

# Title
st.title("🎨 Color Detection Application")
st.markdown("Deteksi warna pada gambar dengan berbagai color space dan metode thresholding")

# Sidebar untuk konfigurasi
with st.sidebar:
    st.header("⚙️ Konfigurasi")
    
    # Upload gambar
    uploaded_file = st.file_uploader(
        "Upload Gambar",
        type=['jpg', 'jpeg', 'png'],
        help="Upload gambar dalam format JPG atau PNG"
    )
    
    if uploaded_file is not None:
        # Pilihan format warna
        st.subheader("📐 Format Warna")
        color_format = st.radio(
            "Pilih Color Space:",
            ["RGB", "R-G Color Space", "Normalized RGB", "HSV", "YCrCb", "TSL"],
            index=0
        )
        
        # Pilihan metode threshold
        st.subheader("🔍 Metode Threshold")
        threshold_method = st.radio(
            "Pilih Metode:",
            ["Static Threshold", "Dynamic Threshold", "Distance Threshold"],
            index=0
        )
        
        # Dynamic UI berdasarkan metode threshold
        if threshold_method == "Static Threshold":
            st.subheader("📊 Parameter Static Threshold")
            
            # Tentukan jumlah channel berdasarkan format warna
            if color_format in ["RGB", "Normalized RGB", "HSV", "YCrCb", "TSL"]:
                num_channels = 3
                channel_names = ["Channel 0", "Channel 1", "Channel 2"]
            elif color_format == "R-G Color Space":
                num_channels = 3
                channel_names = ["R-G", "B", "Channel 2"]
            else:
                num_channels = 3
                channel_names = ["Channel 0", "Channel 1", "Channel 2"]
            
            min_vals = []
            max_vals = []
            
            for i in range(num_channels):
                st.markdown(f"**{channel_names[i]}**")
                col1, col2 = st.columns(2)
                with col1:
                    min_val = st.slider(
                        f"Min {channel_names[i]}",
                        0, 255, 0,
                        key=f"min_{i}"
                    )
                with col2:
                    max_val = st.slider(
                        f"Max {channel_names[i]}",
                        0, 255, 255,
                        key=f"max_{i}"
                    )
                min_vals.append(min_val)
                max_vals.append(max_val)
        
        elif threshold_method == "Dynamic Threshold":
            st.subheader("📊 Parameter Dynamic Threshold")
            k_factor = st.slider(
                "K Factor",
                0.0, 5.0, 1.0, 0.1,
                help="Faktor k untuk menghitung threshold: mean ± k * std"
            )
        
        elif threshold_method == "Distance Threshold":
            st.subheader("📊 Parameter Distance Threshold")
            
            # Color picker untuk target color
            target_r = st.slider("Target R", 0, 255, 128, key="target_r")
            target_g = st.slider("Target G", 0, 255, 128, key="target_g")
            target_b = st.slider("Target B", 0, 255, 128, key="target_b")
            
            # Preview warna target
            target_color_display = np.zeros((50, 50, 3), dtype=np.uint8)
            target_color_display[:, :] = [target_r, target_g, target_b]
            st.image(target_color_display, caption="Target Color", width=100)
            
            max_distance = st.slider(
                "Max Distance",
                0.0, 500.0, 50.0, 1.0,
                help="Maksimum jarak Euclidean dari target color"
            )
        
        # Bonus: Toggle highlight
        st.subheader("🎨 Tampilan")
        use_highlight = st.checkbox(
            "Gunakan Highlight Merah",
            value=True,
            help="Sorot area terdeteksi dengan warna merah"
        )

# Main area
if uploaded_file is not None:
    # Load image
    original_image = load_image(uploaded_file)
    
    if original_image is not None:
        # Konversi ke format warna terpilih (untuk thresholding)
        color_format_map = {
            "RGB": to_rgb,
            "R-G Color Space": to_rg,
            "Normalized RGB": to_normalized_rgb,
            "HSV": to_hsv,
            "YCrCb": to_ycrcb,
            "TSL": to_tsl
        }
        
        # Map untuk display (HSV dan YCrCb perlu konversi khusus)
        color_format_display_map = {
            "RGB": to_rgb,
            "R-G Color Space": to_rg,
            "Normalized RGB": to_normalized_rgb,
            "HSV": to_hsv_display,
            "YCrCb": to_ycrcb_display,
            "TSL": to_tsl
        }
        
        converter_func = color_format_map[color_format]
        converter_display_func = color_format_display_map[color_format]
        converted_image = converter_func(original_image)
        converted_image_display = converter_display_func(original_image)
        
        # Generate mask berdasarkan metode threshold
        if threshold_method == "Static Threshold":
            mask = static_threshold(converted_image, min_vals, max_vals)
        elif threshold_method == "Dynamic Threshold":
            mask = dynamic_threshold(converted_image, k_factor)
        elif threshold_method == "Distance Threshold":
            # Konversi target color ke color space yang dipilih
            target_rgb = np.array([[[target_r, target_g, target_b]]], dtype=np.uint8)
            target_rgb_bgr = cv2.cvtColor(target_rgb, cv2.COLOR_RGB2BGR)
            
            if color_format == "HSV":
                target_converted = cv2.cvtColor(target_rgb_bgr, cv2.COLOR_BGR2HSV)[0, 0]
            elif color_format == "YCrCb":
                target_converted = cv2.cvtColor(target_rgb_bgr, cv2.COLOR_BGR2YCrCb)[0, 0]
            elif color_format == "R-G Color Space":
                # Konversi manual ke R-G space
                rg_diff = target_r - target_g
                rg_norm = ((rg_diff + 255) / 2)
                target_converted = np.array([rg_norm, target_b, target_b], dtype=np.float32)
            elif color_format == "Normalized RGB":
                # Normalize RGB
                sum_rgb = target_r + target_g + target_b
                if sum_rgb == 0:
                    sum_rgb = 1
                target_converted = np.array([
                    (target_r / sum_rgb) * 255,
                    (target_g / sum_rgb) * 255,
                    (target_b / sum_rgb) * 255
                ], dtype=np.float32)
            elif color_format == "TSL":
                # Konversi ke TSL (simplified, menggunakan nilai RGB langsung)
                # Untuk simplicity, kita gunakan RGB sebagai proxy
                target_converted = np.array([target_r, target_g, target_b], dtype=np.float32)
            else:  # RGB
                target_converted = np.array([target_r, target_g, target_b], dtype=np.float32)
            
            mask = distance_threshold(converted_image, target_converted, max_distance, color_format)
        
        # Apply mask ke gambar
        if use_highlight:
            masked_image = apply_mask(original_image, mask, highlight_color=(255, 0, 0), alpha=0.5)
        else:
            masked_image = apply_mask(original_image, mask, highlight_color=(255, 255, 255), alpha=0.3)
        
        # Visualisasi mask
        mask_visualization = create_mask_visualization(mask, highlight_color=(255, 0, 0))
        
        # Tampilkan hasil dalam grid 2x2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Gambar Asli")
            st.image(original_image, use_container_width=True, channels="RGB")
            
            st.subheader("🎨 Gambar Format Terpilih")
            st.image(converted_image_display, use_container_width=True, channels="RGB")
        
        with col2:
            st.subheader("🔍 Mask Hasil Deteksi")
            st.image(mask_visualization, use_container_width=True, channels="RGB")
            
            st.subheader("✨ Gambar dengan Mask Applied")
            st.image(masked_image, use_container_width=True, channels="RGB")
        
        # Download button untuk mask
        st.subheader("💾 Download")
        mask_png = mask_to_png_bytes(mask)
        st.download_button(
            label="Download Mask sebagai PNG",
            data=mask_png,
            file_name="color_detection_mask.png",
            mime="image/png"
        )
        
        # Informasi statistik
        with st.expander("📈 Statistik Deteksi"):
            total_pixels = mask.shape[0] * mask.shape[1]
            detected_pixels = np.sum(mask == 255)
            detection_percentage = (detected_pixels / total_pixels) * 100
            
            st.metric("Total Pixels", f"{total_pixels:,}")
            st.metric("Detected Pixels", f"{detected_pixels:,}")
            st.metric("Detection Percentage", f"{detection_percentage:.2f}%")
    
    else:
        st.error("Gagal memuat gambar. Pastikan file adalah gambar yang valid.")
else:
    # Placeholder ketika belum ada gambar
    st.info("👆 Silakan upload gambar untuk memulai deteksi warna")
    
    # Contoh penggunaan
    with st.expander("ℹ️ Cara Penggunaan"):
        st.markdown("""
        ### Langkah-langkah:
        1. **Upload Gambar**: Klik tombol "Browse files" di sidebar dan pilih gambar JPG/PNG
        2. **Pilih Format Warna**: Pilih color space yang ingin digunakan (RGB, HSV, dll)
        3. **Pilih Metode Threshold**: Pilih salah satu dari 3 metode:
           - **Static Threshold**: Tentukan range min-max untuk setiap channel
           - **Dynamic Threshold**: Threshold otomatis berdasarkan statistik (mean ± k*std)
           - **Distance Threshold**: Deteksi warna berdasarkan jarak dari warna target
        4. **Atur Parameter**: Sesuaikan slider sesuai metode yang dipilih
        5. **Lihat Hasil**: Hasil akan ditampilkan di 4 panel:
           - Gambar Asli
           - Gambar dalam Format Terpilih
           - Mask Hasil Deteksi
           - Gambar dengan Mask Applied
        6. **Download**: Klik tombol download untuk menyimpan mask sebagai PNG
        """)

