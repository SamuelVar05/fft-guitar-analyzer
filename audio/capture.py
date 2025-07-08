import sounddevice as sd
import numpy as np

def capture_audio(duration=0.05, sample_rate=44100):
    """
    Captura audio desde el micrófono por un corto periodo (en segundos).
    Devuelve un array NumPy de la señal de audio.

    Por defecto:
    - duration = 0.05 s → 2205 muestras (~2048)
    """
    print("🎙️ Capturando audio...")

    recording = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,
        dtype='float32'
    )
    sd.wait()
    print("✅ Captura finalizada.")
    return recording.flatten(), sample_rate
