import sounddevice as sd
from config import CAPTURE_DURATION, SAMPLE_RATE

def capture_audio(duration=CAPTURE_DURATION, sample_rate=SAMPLE_RATE):
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
