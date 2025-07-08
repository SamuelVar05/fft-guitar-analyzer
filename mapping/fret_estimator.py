def estimate_string_and_fret(midi_note: int):
    """
    Estima qué cuerda y traste podrían haber producido una nota dada en MIDI.

    Retorna una lista de tuplas (cuerda, traste) posibles.
    La cuerda 1 es la más delgada (E4), la 6 es la más grave (E2).
    """

    string_tunings = {
        6: 40,  # E2
        5: 45,  # A2
        4: 50,  # D3
        3: 55,  # G3
        2: 59,  # B3
        1: 64   # E4
    }

    posibles = []
    for string, open_midi in string_tunings.items():
        fret = midi_note - open_midi
        if 0 <= fret <= 20:  # Rango de trastes típico
            posibles.append((string, fret))

    return posibles
