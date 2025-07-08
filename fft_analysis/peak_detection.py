import numpy as np
from scipy.signal import find_peaks

def detect_fundamental(freqs: np.ndarray, magnitudes: np.ndarray, min_freq: float = 60.0, max_freq: float = 1000.0):
    """
    Detecta la frecuencia fundamental dentro del rango audible útil para guitarra.

    Parámetros:
    - freqs: array de frecuencias positivas (Hz)
    - magnitudes: array de magnitudes correspondientes
    - min_freq: frecuencia mínima a considerar (Hz)
    - max_freq: frecuencia máxima a considerar (Hz)

    Retorna:
    - fundamental_freq: frecuencia dominante (Hz) o None si no se encuentra
    """
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(mask):
        return None

    filtered_freqs = freqs[mask]
    filtered_mags = magnitudes[mask]

    index = np.argmax(filtered_mags)
    fundamental_freq = filtered_freqs[index]
    return fundamental_freq

def detect_multiple_fundamentals(freqs, magnitudes, min_freq=60.0, max_freq=1000.0, threshold=0.1):
    """
    Detecta múltiples picos (notas posibles) en el espectro.
    Retorna una lista de frecuencias fundamentales candidatas.
    """
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    freqs = freqs[mask]
    mags = magnitudes[mask]

    # Umbral relativo al pico más alto
    height_threshold = threshold * max(mags)
    peaks, _ = find_peaks(mags, height=height_threshold, distance=10)  # evitar picos muy cercanos

    detected_freqs = freqs[peaks]
    return detected_freqs
