# Tamaño de la ventana de análisis FFT
WINDOW_SIZE = 4096

# Frecuencia de muestreo estándar
SAMPLE_RATE = 44100

# Duración de captura en segundos
# 16384 muestras a 44100 Hz ≈ 0.37 segundos
CAPTURE_DURATION = WINDOW_SIZE / SAMPLE_RATE

UPDATE_INTERVAL_MS = int(CAPTURE_DURATION * 1000)  # en milisegundos para .after()

NUM_WINDOWS = 5  # Número de ventanas para el análisis

HOP_SIZE = WINDOW_SIZE // 2  # Tamaño de salto entre ventanas