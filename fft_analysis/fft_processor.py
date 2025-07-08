import numpy as np

def compute_fft(signal: np.ndarray, sample_rate: int):
    """
    Aplica la FFT a una señal y retorna las frecuencias positivas y sus magnitudes.

    Parámetros:
    - signal: array de la señal en el dominio del tiempo
    - sample_rate: tasa de muestreo en Hz

    Retorna:
    - freqs: array de frecuencias positivas
    - magnitudes: array de magnitudes normalizadas
    """
    n = len(signal)
    fft_result = np.fft.fft(signal)
    freqs = np.fft.fftfreq(n, d=1/sample_rate)

    # Tomar solo la mitad positiva del espectro
    half = n // 2
    freqs = freqs[:half]
    magnitudes = np.abs(fft_result[:half])
    max_mag = np.max(magnitudes)
    if max_mag > 0:
        magnitudes /= max_mag
    else:
        magnitudes[:] = 0  # o simplemente dejarlo como está sin normalizar


    return freqs, magnitudes
