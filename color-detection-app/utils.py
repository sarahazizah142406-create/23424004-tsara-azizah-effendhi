"""
Utility functions untuk image loading dan mask visualization.
"""

import numpy as np
import cv2
from PIL import Image
import io


def load_image(uploaded_file):
    """
    Load image dari Streamlit uploaded file.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
    
    Returns:
        numpy.ndarray: Image dalam format RGB
    """
    # Baca file sebagai bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    
    # Decode menggunakan OpenCV
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Convert BGR ke RGB
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image


def apply_mask(image, mask, highlight_color=(255, 0, 0), alpha=0.5):
    """
    Apply mask ke gambar dengan highlight warna.
    
    Args:
        image: Original image (RGB)
        mask: Binary mask (0 atau 255)
        highlight_color: Warna highlight (R, G, B)
        alpha: Transparency factor untuk highlight (0-1)
    
    Returns:
        numpy.ndarray: Image dengan mask applied
    """
    # Buat copy dari image
    result = image.copy()
    
    # Buat overlay dengan warna highlight
    overlay = result.copy()
    overlay[mask == 255] = highlight_color
    
    # Blend overlay dengan original image
    result = cv2.addWeighted(result, 1 - alpha, overlay, alpha, 0)
    
    return result


def create_mask_visualization(mask, highlight_color=(255, 0, 0)):
    """
    Buat visualisasi mask dengan warna highlight.
    
    Args:
        mask: Binary mask (0 atau 255)
        highlight_color: Warna highlight (R, G, B)
    
    Returns:
        numpy.ndarray: Visualisasi mask sebagai RGB image
    """
    # Buat image 3 channel dari mask
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # Set warna highlight untuk pixel yang terdeteksi
    mask_rgb[mask == 255] = highlight_color
    
    return mask_rgb


def mask_to_png_bytes(mask):
    """
    Convert mask ke PNG bytes untuk download.
    
    Args:
        mask: Binary mask (0 atau 255)
    
    Returns:
        bytes: PNG image bytes
    """
    # Convert mask ke PIL Image
    mask_pil = Image.fromarray(mask, mode='L')
    
    # Convert ke bytes
    img_bytes = io.BytesIO()
    mask_pil.save(img_bytes, format='PNG')
    img_bytes.seek(0)
    
    return img_bytes.getvalue()

