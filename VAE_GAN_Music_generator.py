import numpy as np
import tensorflow as tf
from keras.models import load_model
from music21 import stream, note, tempo, instrument, chord


def generate_music(generator, latent_dim, sequence_length, output_file, num_segments=3):
    music_stream = stream.Score()

    # Add a piano part
    music_stream.insert(0, instrument.Piano())
    music_stream.insert(0, tempo.MetronomeMark(number=120))

    current_offset = 0.0

    # Generate multiple segments
    for segment in range(num_segments):
        print(f"Generating segment {segment+1}/{num_segments}")

        # Generate a slightly different noise vector for each segment
        noise = np.random.normal(0, 1, (1, latent_dim))

        # Generate music sequence using the generator
        generated_sequence = generator.predict(noise.reshape(1, latent_dim))
        print(f"Generated sequence shape: {generated_sequence.shape}")

        # Reshaping generated sequence
        generated_sequence = generated_sequence.reshape((sequence_length, 132))

        # Convert the generated sequence back to original note values
        generated_notes = decode_sequence(generated_sequence, threshold=0.5)

        # Add notes to stream with proper spacing
        for pitches, _, duration, dynamic in generated_notes:
            # Limit number of simultaneous pitches to 4 for more realistic piano music
            if len(pitches) > 4:
                # Take the 4 highest pitches for a more musical result
                pitches = sorted(pitches)[-4:]

            print(
                f"Adding notes: {pitches}, Offset: {current_offset}, Duration: {duration}, Dynamic: {dynamic}")

            if len(pitches) > 1:
                # chord
                chord_notes = [note.Note(p, quarterLength=max(0.25, duration))
                               for p in pitches]
                new_chord = chord.Chord(chord_notes)
                new_chord.volume.velocity = min(
                    127, max(60, int(dynamic * 40 + 70)))
                music_stream.insert(current_offset, new_chord)
            elif len(pitches) == 1:
                # single note
                new_note = note.Note(
                    pitches[0], quarterLength=max(0.25, duration))
                new_note.volume.velocity = min(
                    127, max(60, int(dynamic * 40 + 70)))
                music_stream.insert(current_offset, new_note)

            # Increment offset for next note
            current_offset += max(0.5, duration)

        # Add a short pause between segments (optional)
        current_offset += 1.0

    # Debug output
    print(f"Total song length: {current_offset} quarter notes")

    # Write the MIDI file
    music_stream.write('midi', fp=output_file)


def decode_sequence(seq, threshold=0.5):
    """
    seq: np.array of shape (sequence_length, 132)
    returns list of (pitches:list[int], offset:float, duration:float, dynamic:float)
    """
    out = []
    for timestep in seq:
        # Use a higher threshold to get fewer, more significant pitches
        pitches = [i for i, v in enumerate(timestep[:128]) if v > threshold]

        # Filter to reasonable pitches (middle of piano range)
        pitches = [p for p in pitches if 36 <= p <= 84]  # C2 to C6

        offset = float(timestep[128])
        duration = max(0.25, min(4.0, float(timestep[129])))  # Clamp duration
        dynamic = float(timestep[130])

        # If no pitch detected, you might skip or default
        if not pitches:
            pitches = [60]  # default middle C

        out.append((pitches, offset, duration, dynamic))
    return out


if __name__ == "__main__":
    # Load the saved generator model
    generator = load_model("Trained files/generator_model.h5")

    # Set parameters
    latent_dim = 100
    sequence_length = 30
    num_segments = 5  # Generate 5 segments for a longer song

    # Specify the output file path
    output_file = "generated_long_music.mid"

    # Generate music using the trained generator
    generate_music(generator, latent_dim, sequence_length,
                   output_file, num_segments)

    print("Generated music saved successfully.")
