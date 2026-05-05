"""
Modul untuk berbagai metode thresholding pada color detection.
Mendukung: Static Threshold, Dynamic Threshold, Distance Threshold
"""

import numpy as np
import cv2


def static_threshold(image, min_vals, max_vals):
    """
    Static threshold dengan range min-max untuk setiap channel.
    
    Args:
        image: Input image (dalam color space terpilih)
        min_vals: List/array nilai minimum untuk setiap channel [min_ch0, min_ch1, min_ch2]
        max_vals: List/array nilai maksimum untuk setiap channel [max_ch0, max_ch1, max_ch2]
    
    Returns:
        numpy.ndarray: Binary mask (0 atau 255)
    """
    # Pastikan image dalam format yang benar
    if len(image.shape) == 2:
        # Grayscale - hanya 1 channel
        mask = cv2.inRange(image, min_vals[0], max_vals[0])
    else:
        # Multi-channel
        min_vals = np.array(min_vals, dtype=np.uint8)
        max_vals = np.array(max_vals, dtype=np.uint8)
        mask = cv2.inRange(image, min_vals, max_vals)
    
    return mask


def dynamic_threshold(image, k_factor):
    """
    Dynamic threshold berdasarkan mean ± k * standard deviation.
    Threshold dihitung otomatis dari histogram setiap channel.
    
    Args:
        image: Input image (dalam color space terpilih)
        k_factor: Faktor k untuk menghitung threshold (mean ± k * std)
    
    Returns:
        numpy.ndarray: Binary mask (0 atau 255)
    """
    # Hitung mean dan std untuk setiap channel
    if len(image.shape) == 2:
        # Grayscale
        mean_val = np.mean(image)
        std_val = np.std(image)
        min_val = max(0, int(mean_val - k_factor * std_val))
        max_val = min(255, int(mean_val + k_factor * std_val))
        mask = cv2.inRange(image, min_val, max_val)
    else:
        # Multi-channel
        masks = []
        for ch in range(image.shape[2]):
            channel = image[:, :, ch]
            mean_val = np.mean(channel)
            std_val = np.std(channel)
            min_val = max(0, int(mean_val - k_factor * std_val))
            max_val = min(255, int(mean_val + k_factor * std_val))
            mask_ch = cv2.inRange(channel, min_val, max_val)
            masks.append(mask_ch)
        
        # Gabungkan semua channel dengan AND operation
        mask = masks[0]
        for m in masks[1:]:
            mask = cv2.bitwise_and(mask, m)
    
    return mask


def distance_threshold(image, target_color, max_distance, color_space='RGB'):
    """
    Distance threshold menggunakan Euclidean distance dari target color.
    
    Args:
        image: Input image (dalam color space terpilih)
        target_color: List/array warna target [R, G, B] atau sesuai color space
        max_distance: Maksimum jarak Euclidean yang diizinkan
        color_space: Color space yang digunakan (untuk referensi)
    
    Returns:
        numpy.ndarray: Binary mask (0 atau 255)
    """
    # Pastikan image dalam format float untuk perhitungan
    img_float = image.astype(np.float32)
    target = np.array(target_color, dtype=np.float32)
    
    # Hitung Euclidean distance untuk setiap pixel
    if len(image.shape) == 2:
        # Grayscale - hanya 1 channel
        distances = np.abs(img_float - target[0])
    else:
        # Multi-channel - Euclidean distance
        diff = img_float - target
        distances = np.sqrt(np.sum(diff**2, axis=2))
    
    # Buat mask: pixel dengan distance <= max_distance
    mask = (distances <= max_distance).astype(np.uint8) * 255
    
    return mask

