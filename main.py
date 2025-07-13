import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from fft_analysis.fft_processor import averaged_fft
from fft_analysis.peak_detection import detect_multiple_fundamentals

from mapping.note_mapper import freq_to_note, midi_to_note
from mapping.fret_estimator import estimate_string_and_fret

from config import SAMPLE_RATE, WINDOW_SIZE, NUM_WINDOWS, HOP_SIZE

def main():
    # Ruta del archivo de audio WAV
    file_path = "data/test/C.wav"

    # Cargar archivo WAV
    sample_rate, signal = wav.read(file_path)

    # Si es estéreo, convertir a mono
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    # Parámetros usados en averaged_fft
    num_windows = NUM_WINDOWS
    window_size_fft = WINDOW_SIZE
    hop_size = HOP_SIZE
    pre_offset = hop_size // 2  # Offset previo al pico máximo

    # Obtener el índice del pico máximo
    peak_idx = np.argmax(np.abs(signal))
    start = max(0, peak_idx - pre_offset)

    # Visualizar la señal completa con las ventanas marcadas
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label="Señal completa")
    for i in range(num_windows):
        s = start + i * hop_size
        e = s + window_size_fft
        if e > len(signal):
            break
        plt.axvspan(s, e, color='orange', alpha=0.3, label="Ventana usada" if i == 0 else None)
    plt.axvline(peak_idx, color='red', linestyle='--', label='Pico de inicio')
    plt.title("Ventanas de análisis sobre la señal completa")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Aplicar FFT promediada
    freqs, magnitudes = averaged_fft(signal, sample_rate,
                                     num_windows=num_windows,
                                     window_size=window_size_fft,
                                     hop_size=hop_size,
                                     pre_offset=pre_offset)

    # Detectar múltiples frecuencias fundamentales
    fundamentals = detect_multiple_fundamentals(freqs, magnitudes)
    notas_detectadas = []
    for f in fundamentals:
        note_name, midi = freq_to_note(f)
        idx = np.argmin(np.abs(freqs - f))
        notas_detectadas.append({
            "freq": f,
            "magnitude": magnitudes[idx],
            "midi": midi,
            "note": note_name
        })

    # Mostrar espectro de la señal
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitudes, label="Espectro", color='green')
    plt.title("Espectro de la señal (FFT)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud normalizada")
    plt.xlim(0, 1000)
    plt.grid()
    plt.legend()

    # Anotar notas detectadas (máximo 6)
    for f in fundamentals[:6]:
        note_name, _ = freq_to_note(f)
        idx = np.argmin(np.abs(freqs - f))
        mag = magnitudes[idx]
        plt.annotate(
            note_name,
            xy=(f, mag),
            xytext=(f, mag + 0.05),
            ha='center',
            fontsize=9,
            arrowprops=dict(arrowstyle='->', lw=0.5)
        )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
