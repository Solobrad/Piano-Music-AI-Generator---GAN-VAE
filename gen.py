import numpy as np
import tensorflow as tf
from keras.models import load_model
from music21 import stream, note, tempo, instrument, chord

instrument_name = "piano"  # Replace "piano" with the desired instrument name


def generate_music(generator, latent_dim, sequence_length, n_notes, output_file):
    # Generate random noise as input for the generator
    noise = np.random.normal(0, 1, (1, latent_dim))

    # Generate a music sequence using the generator
    generated_sequence = generator.predict(noise)
    generated_sequence = generated_sequence.reshape((sequence_length, n_notes))

    # Convert the generated sequence back to original note values
    generated_notes = decode_sequence(generated_sequence)

    # Create a Music21 stream to store the generated notes
    music_stream = stream.Score()

    # Add generated notes to the stream
    for note_value in generated_notes:
        if isinstance(note_value, list):  # Chord
            chord_notes = [note.Note(int(n)) for n in note_value]
            new_chord = chord.Chord(chord_notes)
            music_stream.append(new_chord)
        else:  # Single note
            new_note = note.Note(int(note_value))
            music_stream.append(new_note)

    # Set the instrument for the entire stream
    instrument_obj = instrument.Piano()  # Replace with the desired instrument
    music_stream.insert(0, instrument_obj)

    # Set the tempo
    music_stream.insert(0, tempo.MetronomeMark(number=120))

    # Save the generated music as a MIDI file
    music_stream.write('midi', fp=output_file)


def decode_sequence(sequence):
    # Decode the sequence back to original note values
    return [int(x * 87) + 21 for x in sequence.flatten()]


if __name__ == "__main__":
    # Load the saved generator model
    generator = load_model(
        "C:\\Users\\ASUS\\Desktop\\AI\\AI piano Music GAN\\generator_model.h5")

    # Set parameters
    latent_dim = 100
    sequence_length = 30
    n_notes = 1  # Number of unique notes in the dataset

    # Specify the output file path
    output_file = "C:\\Users\\ASUS\\Desktop\\AI\\AI piano Music GAN\\generated_music.mid"

    # Generate music using the trained generator
    generate_music(generator, latent_dim,
                   sequence_length, n_notes, output_file)

    print("Generated music saved successfully.")
