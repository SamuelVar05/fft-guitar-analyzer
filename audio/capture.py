import sounddevice as sd
import numpy as np
from config import SAMPLE_RATE

class Recorder:
    def __init__(self, sample_rate=SAMPLE_RATE):
        self.sample_rate = sample_rate
        self.recording = []
        self.stream = None

    def callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.recording.append(indata.copy())

    def start(self):
        self.recording = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32',
            callback=self.callback
        )
        self.stream.start()
        print("🎙️ Grabando...")

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
        audio = np.concatenate(self.recording, axis=0).flatten()
        print("✅ Grabación finalizada.")
        return audio, self.sample_rate
