import numpy as np
import tensorflow as tf
from keras.models import load_model
from music21 import stream, note, tempo, instrument, chord

instrument_name = "piano"  # Replace "piano" with the desired instrument name


def generate_music(generator, latent_dim, sequence_length, output_file):
    noise = np.random.normal(0, 1, (1, latent_dim))  # Single hand

    # Generate music sequence using the generator
    generated_sequence = generator.predict(noise.reshape(1, latent_dim))

    # **Debugging: Check the shape of the generated sequence**
    print(f"Generated sequence shape: {generated_sequence.shape}")

    # Reshaping generated sequence
    generated_sequence = generated_sequence.reshape(
        (sequence_length, 132))  # Modify based on output shape

    # Convert the generated sequence back to original note values
    generated_notes = decode_sequence(generated_sequence)

    # Create a Music21 stream to store the generated notes
    music_stream = stream.Score()

    # Add generated notes to the stream for one hand
    for pitches, offset, duration, dynamic in generated_notes:
        if len(pitches) > 1:
            # chord
            chord_notes = [note.Note(m, quarterLength=duration)
                           for m in pitches]
            new = chord.Chord(chord_notes)
        else:
            # single note
            new = note.Note(pitches[0], quarterLength=duration)
        # 🔄 changed: you can use 'dynamic' if you want to set velocity later
        new.volume.velocity = int(dynamic)
        music_stream.insert(offset, new)

    music_stream.insert(0, instrument.Piano())
    music_stream.insert(0, tempo.MetronomeMark(number=120))
    music_stream.write('midi', fp=output_file)


def decode_sequence(seq, threshold=0.5):
    """
    seq: np.array of shape (sequence_length, 132)
    returns list of (pitches:list[int], offset:float, duration:float, dynamic:float)
    """
    out = []
    for timestep in seq:
        # 🔄 changed: multi‑hot pitches are in 0–127
        pitches = [i for i, v in enumerate(timestep[:128]) if v > threshold]
        offset = float(timestep[128])   # 🔄 changed
        duration = float(timestep[129])   # 🔄 changed
        dynamic = float(timestep[130])   # 🔄 changed
        # if no pitch detected, you might skip or default
        if not pitches:
            pitches = [60]  # default middle C
        out.append((pitches, offset, duration, dynamic))
    return out


if __name__ == "__main__":
    # Load the saved generator model
    generator = load_model(
        "Trained files/generator_model.keras")

    # Set parameters
    latent_dim = 100
    sequence_length = 30

    # Specify the output file path
    output_file = "generated_music.mid"

    # Generate music using the trained generator
    generate_music(generator, latent_dim,
                   sequence_length, output_file)

    print("Generated music saved successfully.")
