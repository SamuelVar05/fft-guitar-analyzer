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

def averaged_fft(signal, sample_rate, window_size=4096, hop_size=2048, num_windows=5, pre_offset=1024):
    total_len = len(signal)

    # Buscar el índice del pico más fuerte en la señal
    peak_idx = np.argmax(np.abs(signal))

    # Retroceder un poco para capturar el ataque
    start = max(0, peak_idx - pre_offset)

    # Asegurar que haya suficiente espacio para todas las ventanas
    if start + num_windows * hop_size + window_size > total_len:
        start = max(0, total_len - (num_windows * hop_size + window_size))

    spectra = []

    for i in range(num_windows):
        s = start + i * hop_size
        e = s + window_size
        if e > total_len:
            break
        segment = signal[s:e]
        freqs, mags = compute_fft(segment, sample_rate)
        spectra.append(mags)

    avg_mags = np.mean(spectra, axis=0)
    return freqs, avg_mags
