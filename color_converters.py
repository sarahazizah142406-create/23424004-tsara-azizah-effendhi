"""
Modul untuk konversi warna ke berbagai color space.
Mendukung: RGB, R-G, Normalized RGB, HSV, YCrCb, TSL
"""

import numpy as np
import cv2


def to_rgb(image):
    """
    Konversi gambar ke RGB color space.
    Memastikan gambar dalam format RGB (3 channel).
    
    Args:
        image: Input image (BGR dari OpenCV atau RGB)
    
    Returns:
        numpy.ndarray: Gambar dalam format RGB
    """
    # Jika gambar sudah RGB (3 channel), return as is
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Jika dari OpenCV (BGR), convert ke RGB
        if image.dtype == np.uint8:
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    # Jika grayscale, convert ke RGB
    elif len(image.shape) == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    return image


def to_rg(image):
    """
    Konversi ke R-G color space.
    R-G space menggunakan R-G difference dan B channel.
    
    Args:
        image: Input image dalam format RGB
    
    Returns:
        numpy.ndarray: Gambar dalam format R-G (3 channel: R-G, B, 0)
    """
    rgb = to_rgb(image).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    # Hitung R-G difference
    rg_diff = r - g
    
    # Normalisasi ke range 0-255 untuk visualisasi
    rg_normalized = ((rg_diff + 255) / 2).astype(np.uint8)
    
    # Buat 3 channel: R-G difference, B, dan channel ketiga (copy dari B)
    rg_image = np.zeros_like(rgb, dtype=np.uint8)
    rg_image[:, :, 0] = rg_normalized
    rg_image[:, :, 1] = b.astype(np.uint8)
    rg_image[:, :, 2] = b.astype(np.uint8)
    
    return rg_image


def to_normalized_rgb(image):
    """
    Konversi ke Normalized RGB (r, g, b).
    r = R/(R+G+B), g = G/(R+G+B), b = B/(R+G+B)
    
    Args:
        image: Input image dalam format RGB
    
    Returns:
        numpy.ndarray: Gambar dalam normalized RGB (0-255 untuk visualisasi)
    """
    rgb = to_rgb(image).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    # Hitung sum untuk setiap pixel
    sum_rgb = r + g + b
    
    # Handle division by zero (pixel hitam)
    sum_rgb[sum_rgb == 0] = 1
    
    # Normalisasi
    r_norm = r / sum_rgb
    g_norm = g / sum_rgb
    b_norm = b / sum_rgb
    
    # Scale ke 0-255 untuk visualisasi
    normalized = np.zeros_like(rgb, dtype=np.uint8)
    normalized[:, :, 0] = (r_norm * 255).astype(np.uint8)
    normalized[:, :, 1] = (g_norm * 255).astype(np.uint8)
    normalized[:, :, 2] = (b_norm * 255).astype(np.uint8)
    
    return normalized


def to_hsv(image):
    """
    Konversi ke HSV color space menggunakan OpenCV.
    
    Args:
        image: Input image dalam format RGB
    
    Returns:
        numpy.ndarray: Gambar dalam format HSV (untuk thresholding)
    """
    rgb = to_rgb(image)
    # OpenCV expects BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    return hsv


def to_hsv_display(image):
    """
    Konversi ke HSV untuk display (convert kembali ke RGB untuk visualisasi).
    
    Args:
        image: Input image dalam format RGB
    
    Returns:
        numpy.ndarray: Gambar HSV yang dikonversi ke RGB untuk display
    """
    hsv = to_hsv(image)
    # Convert kembali ke RGB untuk konsistensi display
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_ycrcb(image):
    """
    Konversi ke YCrCb color space menggunakan OpenCV.
    
    Args:
        image: Input image dalam format RGB
    
    Returns:
        numpy.ndarray: Gambar dalam format YCrCb (untuk thresholding)
    """
    rgb = to_rgb(image)
    # OpenCV expects BGR
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    return ycrcb


def to_ycrcb_display(image):
    """
    Konversi ke YCrCb untuk display (convert kembali ke RGB untuk visualisasi).
    
    Args:
        image: Input image dalam format RGB
    
    Returns:
        numpy.ndarray: Gambar YCrCb yang dikonversi ke RGB untuk display
    """
    ycrcb = to_ycrcb(image)
    # Convert kembali ke RGB untuk konsistensi display
    bgr = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def to_tsl(image):
    """
    Konversi ke TSL (Tint-Saturation-Luminance) color space.
    Implementasi manual dari RGB.
    
    Rumus TSL:
    - L = 0.299*R + 0.587*G + 0.114*B (luminance)
    - T = atan2(R' - 1/3, G' - 1/3) / (2*pi) + 0.5 (tint)
    - S = 9/4 * sqrt((R' - 1/3)^2 + (G' - 1/3)^2) (saturation)
    dimana R' = R/(R+G+B), G' = G/(R+G+B)
    
    Args:
        image: Input image dalam format RGB
    
    Returns:
        numpy.ndarray: Gambar dalam format TSL (ditampilkan sebagai RGB)
    """
    rgb = to_rgb(image).astype(np.float32)
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    
    # Hitung sum untuk normalized RGB
    sum_rgb = r + g + b
    sum_rgb[sum_rgb == 0] = 1  # Handle division by zero
    
    r_prime = r / sum_rgb
    g_prime = g / sum_rgb
    
    # Hitung Luminance
    L = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Hitung Tint
    r_diff = r_prime - 1.0/3.0
    g_diff = g_prime - 1.0/3.0
    T = np.arctan2(r_diff, g_diff) / (2 * np.pi) + 0.5
    
    # Hitung Saturation
    S = (9.0/4.0) * np.sqrt(r_diff**2 + g_diff**2)
    S = np.clip(S, 0, 1)  # Clip ke range 0-1
    
    # Normalisasi ke 0-255 untuk visualisasi
    T_scaled = (T * 255).astype(np.uint8)
    S_scaled = (S * 255).astype(np.uint8)
    L_scaled = np.clip(L, 0, 255).astype(np.uint8)
    
    # Buat gambar TSL (3 channel)
    tsl_image = np.zeros_like(rgb, dtype=np.uint8)
    tsl_image[:, :, 0] = T_scaled  # Tint
    tsl_image[:, :, 1] = S_scaled  # Saturation
    tsl_image[:, :, 2] = L_scaled  # Luminance
    
    return tsl_image

