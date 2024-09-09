import numpy as np
import tensorflow as tf
from keras.models import load_model
from music21 import stream, note, tempo, instrument, chord

instrument_name = "piano"  # Replace "piano" with the desired instrument name


def generate_music(generator, latent_dim, sequence_length, n_notes, output_file):
    # Generate random noise as input for the generator
    # Two hands, two noise vectors
    noise = np.random.normal(0, 1, (2, latent_dim))

    # Generate music sequences using the generator for each hand
    generated_sequence_left = generator.predict(
        noise[0].reshape(1, latent_dim))
    generated_sequence_right = generator.predict(
        noise[1].reshape(1, latent_dim))

    generated_sequence_left = generated_sequence_left.reshape(
        (sequence_length, n_notes))
    generated_sequence_right = generated_sequence_right.reshape(
        (sequence_length, n_notes))

    # Convert the generated sequences back to original note values
    generated_notes_left = decode_sequence(generated_sequence_left)
    generated_notes_right = decode_sequence(generated_sequence_right)

    # Create a Music21 stream to store the generated notes
    music_stream = stream.Score()

    # Add generated notes to the stream
    for i in range(len(generated_notes_left)):
        left_note_value = generated_notes_left[i]
        right_note_value = generated_notes_right[i]

        # Handle left hand notes
        if isinstance(left_note_value, list):  # Chord
            chord_notes_left = [
                note.Note(int(n), quarterLength=1) for n in left_note_value
            ]  # Use half note for chord notes
            new_chord_left = chord.Chord(chord_notes_left)
            music_stream.append(new_chord_left)
        else:  # Single note
            duration_choices = [0.25, 0.5, 1.0]
            selected_duration = np.random.choice(duration_choices)
            new_note_left = note.Note(
                int(left_note_value), quarterLength=selected_duration)
            music_stream.append(new_note_left)

        # Handle right hand notes
        if isinstance(right_note_value, list):  # Chord
            chord_notes_right = [
                note.Note(int(n), quarterLength=1) for n in right_note_value
            ]  # Use half note for chord notes
            new_chord_right = chord.Chord(chord_notes_right)
            music_stream.append(new_chord_right)
        else:  # Single note
            duration_choices = [0.25, 0.5, 1.0]
            selected_duration = np.random.choice(duration_choices)
            new_note_right = note.Note(
                int(right_note_value), quarterLength=selected_duration)
            music_stream.append(new_note_right)

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
