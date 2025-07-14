import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from audio.capture import Recorder
from fft_analysis.fft_processor import averaged_fft
from fft_analysis.peak_detection import detect_multiple_fundamentals
from mapping.note_mapper import freq_to_note
from mapping.fret_estimator import estimate_string_and_fret
from config import SAMPLE_RATE, WINDOW_SIZE, HOP_SIZE, NUM_WINDOWS

class GuitarraUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Reconocimiento de Notas de Guitarra")
        self.recorder = Recorder()

        self.notas = []
        self.idx_actual = 0
        self.freqs = None
        self.magnitudes = None
        self.signal_segment = None

        self.frame_top = ttk.Frame(root)
        self.frame_top.pack(pady=10)

        self.button_grabar = ttk.Button(self.frame_top, text="🎙️ Iniciar Grabación", command=self.iniciar_grabacion)
        self.button_grabar.pack()

        self.button_detener = ttk.Button(self.frame_top, text="⏹️ Detener y Analizar", command=self.detener_y_analizar, state=tk.DISABLED)
        self.button_detener.pack(pady=5)

        # Crear 3 subgráficos
        self.fig, (self.ax0, self.ax1, self.ax2) = plt.subplots(3, 1, figsize=(7, 5), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.fig, master=root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack()

        self.frame_nav = ttk.Frame(root)
        self.frame_nav.pack(pady=5)

        self.btn_prev = ttk.Button(self.frame_nav, text="⬅️ Anterior", command=self.mostrar_anterior, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, padx=10)

        self.btn_next = ttk.Button(self.frame_nav, text="Siguiente ➡️", command=self.mostrar_siguiente, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT, padx=10)

    def iniciar_grabacion(self):
        self.ax0.clear()
        self.ax1.clear()
        self.ax2.clear()
        self.canvas.draw()

        self.recorder.start()
        self.button_grabar.config(state=tk.DISABLED)
        self.button_detener.config(state=tk.NORMAL)
        self.btn_prev.config(state=tk.DISABLED)
        self.btn_next.config(state=tk.DISABLED)

    def detener_y_analizar(self):
        self.button_detener.config(state=tk.DISABLED)
        self.button_grabar.config(state=tk.NORMAL)

        signal, sample_rate = self.recorder.stop()

        abs_signal = np.abs(signal)
        peak_idx = np.argmax(abs_signal)
        start = max(0, peak_idx - HOP_SIZE // 2)
        segment = signal[start:start + WINDOW_SIZE * 2]
        self.signal_segment = segment

        freqs, magnitudes = averaged_fft(
            segment,
            sample_rate,
            window_size=WINDOW_SIZE,
            hop_size=HOP_SIZE,
            num_windows=NUM_WINDOWS,
            pre_offset=HOP_SIZE // 2
        )

        if len(freqs) == 0 or len(magnitudes) == 0:
            print("No se pudo analizar el audio. Intenta grabar nuevamente.")
            return

        fundamentals = detect_multiple_fundamentals(freqs, magnitudes)
        notas = []

        for f in fundamentals:
            note_name, midi = freq_to_note(f)
            if note_name is None:
                continue
            estimaciones = estimate_string_and_fret(midi)
            idx = (abs(freqs - f)).argmin()
            notas.append({
                "freq": f,
                "note": note_name,
                "midi": midi,
                "mag": magnitudes[idx],
                "estimaciones": estimaciones
            })

        self.freqs = freqs
        self.magnitudes = magnitudes
        self.notas = notas
        self.idx_actual = 0

        self.btn_prev.config(state=tk.NORMAL if len(notas) > 1 else tk.DISABLED)
        self.btn_next.config(state=tk.NORMAL if len(notas) > 1 else tk.DISABLED)

        self.mostrar_nota_actual()

    def mostrar_nota_actual(self):
        if not self.notas:
            return

        nota = self.notas[self.idx_actual]

        # Diagrama de cuerdas
        self.ax0.clear()
        self.ax0.set_title(f"Nota: {nota['note']} ({nota['freq']:.1f} Hz)")
        self.ax0.set_xlim(0, 22)
        self.ax0.set_ylim(0.5, 6.5)
        self.ax0.set_xticks(np.arange(0, 23, 1))  # <-- Esta línea es nueva
        self.ax0.set_xlabel("Traste")
        self.ax0.set_yticks(range(1, 7))
        self.ax0.set_yticklabels([f"Cuerda {i}" for i in range(1, 7)][::-1])
        self.ax0.grid(True, axis='x', linestyle='--', alpha=0.5)
        self.ax0.tick_params(left=False)


        for i in range(1, 7):
            self.ax0.hlines(i, 0, 22, color='gray', linewidth=1)

        for cuerda, traste in nota["estimaciones"]:
            self.ax0.plot(traste, 7 - cuerda, 'ro')
            self.ax0.text(traste, 7 - cuerda + 0.15, f"{traste}", ha='center', fontsize=9)

        # Segmento de audio
        self.ax1.clear()
        self.ax1.plot(self.signal_segment, color='black')
        self.ax1.set_title("Segmento seleccionado")
        self.ax1.set_xlabel("Muestras")
        self.ax1.set_ylabel("Amplitud")
        self.ax1.grid()

        # Espectro FFT
        self.ax2.clear()
        self.ax2.plot(self.freqs, self.magnitudes, color='green')
        self.ax2.set_title(f"Espectro (Nota {self.idx_actual + 1}/{len(self.notas)})")
        self.ax2.set_xlabel("Frecuencia (Hz)")
        self.ax2.set_ylabel("Magnitud")
        self.ax2.set_xlim(0, 1000)
        self.ax2.grid()

        self.ax2.annotate(
            nota["note"],
            xy=(nota["freq"], nota["mag"]),
            xytext=(nota["freq"], nota["mag"] + 0.05),
            ha="center",
            fontsize=9,
            arrowprops=dict(arrowstyle="->", lw=0.5)
        )

        self.canvas.draw()

    def mostrar_siguiente(self):
        if self.idx_actual < len(self.notas) - 1:
            self.idx_actual += 1
            self.mostrar_nota_actual()

    def mostrar_anterior(self):
        if self.idx_actual > 0:
            self.idx_actual -= 1
            self.mostrar_nota_actual()

def iniciar_ui():
    root = tk.Tk()
    app = GuitarraUI(root)

    def on_close():
        if app.recorder.stream:
            app.recorder.stop()
        plt.close(app.fig)
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()
