import math

A4_FREQ = 440.0
A4_MIDI = 69
NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

def freq_to_note(freq: float):
    if freq <= 0:
        return None, None

    # Calcular el número MIDI
    midi = round(12 * math.log2(freq / A4_FREQ) + A4_MIDI)
    note_index = midi % 12
    octave = midi // 12 - 1  # Octava MIDI empieza en -1

    note_name = f"{NOTE_NAMES[note_index]}{octave}"
    return note_name, midi
