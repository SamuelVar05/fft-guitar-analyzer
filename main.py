import numpy as np
import scipy.io.wavfile as wav
import matplotlib.pyplot as plt

from fft_analysis.fft_processor import compute_fft
from fft_analysis.peak_detection import detect_multiple_fundamentals

from mapping.note_mapper import freq_to_note
from mapping.fret_estimator import estimate_string_and_fret

from config import SAMPLE_RATE, WINDOW_SIZE

def main():
    # Ruta del archivo de audio WAV
    file_path = "data/test/test_la.wav"

    # Cargar archivo WAV
    sample_rate, signal = wav.read(file_path)

    # Si es estéreo, convertir a mono
    if len(signal.shape) == 2:
        signal = signal.mean(axis=1)

    # Seleccionar una ventana de análisis en la mitad
    window_size = WINDOW_SIZE
    half = len(signal) // 2
    start = max(0, half - window_size // 2)
    end = start + window_size
    segment = signal[start:end]

    # Mostrar la forma de onda del segmento
    plt.figure(figsize=(10, 3))
    plt.plot(segment)
    plt.title("Segmento de señal analizado (forma de onda)")
    plt.xlabel("Muestras")
    plt.ylabel("Amplitud")
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Aplicar FFT
    freqs, magnitudes = compute_fft(segment, sample_rate)

    # Detectar múltiples frecuencias fundamentales
    fundamentals = detect_multiple_fundamentals(freqs, magnitudes)

    if len(fundamentals) == 0:
        print("No se detectaron notas.")
    else:
        print("Notas detectadas:")
        for fundamental in fundamentals:
            note_name, midi = freq_to_note(fundamental)
            print(f"- Frecuencia: {fundamental:.2f} Hz → Nota: {note_name} (MIDI {midi})")

            posibles = estimate_string_and_fret(midi)
            if posibles:
                print("  Posibles ubicaciones (cuerda, traste):")
                for string, fret in posibles:
                    print(f"    Cuerda {string}, traste {fret}")
            else:
                print("  No se encontró ninguna cuerda válida para esta nota.")

    # Mostrar espectro de frecuencias
    plt.figure(figsize=(10, 4))
    plt.plot(freqs, magnitudes)
    plt.title("Espectro de la señal (FFT)")
    plt.xlabel("Frecuencia (Hz)")
    plt.ylabel("Magnitud normalizada")
    plt.grid()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


