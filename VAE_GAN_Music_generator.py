import numpy as np
import tensorflow as tf
from keras.models import load_model
from music21 import stream, note, tempo, instrument, chord

instrument_name = "piano"  # Replace "piano" with the desired instrument name


def generate_music(generator, latent_dim, sequence_length, n_notes, output_file):
    noise = np.random.normal(0, 1, (1, latent_dim))  # Single hand

    # Generate music sequence using the generator
    generated_sequence = generator.predict(noise.reshape(1, latent_dim))

    # **Debugging: Check the shape of the generated sequence**
    print(f"Generated sequence shape: {generated_sequence.shape}")

    # Reshaping generated sequence
    generated_sequence = generated_sequence.reshape(
        (sequence_length, 4))  # Modify based on output shape

    # Convert the generated sequence back to original note values
    generated_notes = decode_sequence(generated_sequence)

    # Create a Music21 stream to store the generated notes
    music_stream = stream.Score()

    # Add generated notes to the stream for one hand
    for note_value in generated_notes:
        # Handle single note or chord
        if isinstance(note_value, list):  # Chord
            chord_notes = [note.Note(int(n), quarterLength=1)
                           for n in note_value]
            new_chord = chord.Chord(chord_notes)
            music_stream.append(new_chord)
        else:  # Single note
            duration_choices = [0.25, 0.5, 1.0]
            selected_duration = np.random.choice(duration_choices)
            new_note = note.Note(
                int(note_value), quarterLength=selected_duration)
            music_stream.append(new_note)

    # Set the instrument for the entire stream
    instrument_obj = instrument.Piano()  # Replace with the desired instrument
    music_stream.insert(0, instrument_obj)

    # Set the tempo
    music_stream.insert(0, tempo.MetronomeMark(number=120))

    # Save the generated music as a MIDI file
    music_stream.write('midi', fp=output_file)


def decode_sequence(sequence, low_note=60, high_note=87):
    # Scale generated values to the desired MIDI note range
    return [int(x * (high_note - low_note)) + low_note for x in sequence.flatten()]


if __name__ == "__main__":
    # Load the saved generator model
    generator = load_model(
        "generator_model.h5")

    # Set parameters
    latent_dim = 100
    sequence_length = 30
    n_notes = 1  # Number of unique notes in the dataset

    # Specify the output file path
    output_file = "generated_music.mid"

    # Generate music using the trained generator
    generate_music(generator, latent_dim,
                   sequence_length, n_notes, output_file)

    print("Generated music saved successfully.")
