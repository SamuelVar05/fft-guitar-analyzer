import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from fft_analysis.fft_processor import averaged_fft
from fft_analysis.peak_detection import detect_multiple_fundamentals

from mapping.note_mapper import freq_to_note, midi_to_note
from mapping.fret_estimator import estimate_string_and_fret

from config import SAMPLE_RATE, WINDOW_SIZE, NUM_WINDOWS, HOP_SIZE

def load_audio(file_path):
    sample_rate, signal = wav.read(file_path)
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)  # Convertir a mono si es estéreo
    return sample_rate, signal


def get_analysis_start(signal, hop_size):
    peak_idx = np.argmax(np.abs(signal))
    start = max(0, peak_idx - hop_size // 2)
    return start, peak_idx


def plot_signal_with_windows(signal, start, hop_size, window_size, num_windows, peak_idx):
    plt.figure(figsize=(12, 4))
    plt.plot(signal, label="Señal completa")
    for i in range(num_windows):
        s = start + i * hop_size
        e = s + window_size
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


def detect_notes(freqs, magnitudes):
    fundamentals = detect_multiple_fundamentals(freqs, magnitudes)
    notas = []
    for f in fundamentals:
        note_name, midi = freq_to_note(f)
        idx = np.argmin(np.abs(freqs - f))
        notas.append({
            "freq": f,
            "magnitude": magnitudes[idx],
            "midi": midi,
            "note": note_name
        })
    return fundamentals, notas


def plot_spectrum(freqs, magnitudes, fundamentals):
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitudes, label="Espectro", color='green')
    plt.title("Espectro de la señal (FFT)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud normalizada")
    plt.xlim(0, 1000)
    plt.grid()
    plt.legend()

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

def main():
    file_path = "data/test/C.wav"
    sample_rate, signal = load_audio(file_path)

    num_windows = NUM_WINDOWS
    window_size = WINDOW_SIZE
    hop_size = HOP_SIZE

    start, peak_idx = get_analysis_start(signal, hop_size)
    plot_signal_with_windows(signal, start, hop_size, window_size, num_windows, peak_idx)

    freqs, magnitudes = averaged_fft(
        signal, sample_rate,
        num_windows=num_windows,
        window_size=window_size,
        hop_size=hop_size,
        pre_offset=hop_size // 2
    )

    fundamentals, notas_detectadas = detect_notes(freqs, magnitudes)
    plot_spectrum(freqs, magnitudes, fundamentals)



if __name__ == "__main__":
    main()
