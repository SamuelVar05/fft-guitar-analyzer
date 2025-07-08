import numpy as np

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
