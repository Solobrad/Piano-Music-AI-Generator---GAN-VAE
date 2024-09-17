import os
import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, BatchNormalization, Reshape, Input, Concatenate, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
import music21 as m21


def load_notes_from_file(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def visualize_generated_samples(generator, epoch, num_samples=5, latent_dim=100):
    # Generate some samples using the generator
    noise = np.random.normal(0, 1, (num_samples, latent_dim))
    generated_samples = generator.predict(noise)

    # Optionally, visualize or save the generated samples
    # For example, you can use matplotlib or save generated MIDI files

    # Print or log a message indicating that visualization is done
    print(f"Generated samples visualized at epoch {epoch}")


def pitch_to_midi(pitch_string):
    try:
        if not pitch_string:
            print(f"Pitch string is empty: {pitch_string}")

        # Replace unsupported accidentals with supported ones
        pitch_string = pitch_string.replace('♯', '#').replace('♭', 'b')

        # Split the pitch string if it contains multiple pitches
        pitch_strings = pitch_string.split('.')
        midi_values = []

        for ps in pitch_strings:
            if ps:
                try:
                    # Assuming `m21` is used for pitch processing
                    pitch_obj = m21.pitch.Pitch(ps)
                    midi_values.append(pitch_obj.midi)
                except Exception as e:
                    print(f"Error processing pitch string {ps}: {e}")
                    continue
        if midi_values:
            return np.mean(midi_values)
        else:
            print(
                f"No valid MIDI values found for pitch string {pitch_string}")
            return 0
    except Exception as e:
        print(f"Error processing pitch string {pitch_string}: {e}")
        return 0


def prepare_sequences(notes_with_details, sequence_length=30):
    network_input = []
    network_output = []

    for i in range(len(notes_with_details) - sequence_length):
        sequence_in = notes_with_details[i:i + sequence_length]
        sequence_out = notes_with_details[i + sequence_length]

        # Extracting pitch, offset, duration, and dynamic from the sequence
        input_data = []
        for item in sequence_in:
            timestep_features = [0, 0, 0, 0]
            if 'pitch' in item:
                # If it's an individual note, update the features accordingly
                pitch = item['pitch']
                print(f"Processing single pitch: {pitch}")
                timestep_features[0] = pitch_to_midi(item['pitch'])
                timestep_features[1] = item['offset']
                timestep_features[2] = item['duration']
                timestep_features[3] = item['dynamic']
            elif 'pitches' in item:
                # If it's a chord, update the features for each pitch in the chord
                pitches = item['pitches']
                print(f"Processing chord pitches: {pitches}")
                midi_values = [pitch_to_midi(p) for p in item['pitches']]

                # Remove None values from midi_values

                midi_values = [m for m in midi_values if m is not None]

                if midi_values:
                    # Compute mean of valid MIDI values
                    timestep_features[0] = np.mean(midi_values)
                else:
                    # Default value if no valid MIDI values
                    timestep_features[0] = 0

                timestep_features[1] = item['offset']
                timestep_features[2] = item['duration']
                timestep_features[3] = item['dynamic']
        # Pad or truncate the input data to ensure each item has four features
            input_data.append(timestep_features)

        network_input.append(input_data)
        network_output.append(pitch_to_midi(sequence_out.get('pitch', '')))

    network_input = np.array(network_input)
    network_output = np.array(network_output)

    # Print the shape of network_input before reshaping
    print("Shape of network_input before reshaping:", network_input.shape)

    # Ensure the network_input has the shape (num_samples, sequence_length, num_features)
    if network_input.ndim == 3 and network_input.shape[2] != 4:
        print("Warning: The number of features in network_input is not 4. Adjusting...")
        network_input = np.array([x + [[0, 0, 0, 0]] * (sequence_length - len(x)) if len(
            x) < sequence_length else x[:sequence_length] for x in network_input])

    network_input = network_input.reshape(
        network_input.shape[0], network_input.shape[1], 4)
    print("Shape of network_input after reshaping:", network_input.shape)

    # Debug print to identify timesteps with only one feature
    for i, timestep in enumerate(network_input):
        if timestep.shape[-1] == 1:
            print(f"Timestep {i} has only one feature:", timestep)

    return network_input, network_output


def pad_or_truncate_input(network_input, sequence_length):
    # Pad or truncate the input data to ensure each item has four features
    for i in range(len(network_input)):
        if len(network_input[i]) < sequence_length:
            # If the sequence is shorter than sequence_length, pad it with zeros
            network_input[i] += [[0, 0, 0, 0]] * \
                (sequence_length - len(network_input[i]))
        elif len(network_input[i]) > sequence_length:
            # If the sequence is longer than sequence_length, truncate it
            network_input[i] = network_input[i][:sequence_length]

    return network_input


def build_generator(latent_dim, sequence_length, n_notes):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Dense(256))  # New Dense layer
    model.add(BatchNormalization())
    model.add(Dense(sequence_length * 4, activation='relu'))  # Set units to 1
    model.add(Reshape((sequence_length, 4)))
    return model


def build_discriminator(sequence_length, n_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape=(
        sequence_length, 4), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(512))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def train_gan(generator, discriminator, gan, network_input, network_output, epochs, batch_size=256):
    # Change the number of batches per epoch by subsetting the data
    subset_size = 100000  # Choose the desired subset size
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for _ in tqdm(range(subset_size // batch_size), desc="Batch Progress"):
            idx = np.random.randint(0, network_input.shape[0], batch_size)
            real_notes = network_input[idx]
            real_notes += np.random.normal(0, 0.1, real_notes.shape)
            labels_real = np.ones((batch_size, 1)) * 0.9

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            generated_notes = generator.predict(noise)
            labels_fake = np.zeros((batch_size, 1))

            d_loss_real = discriminator.train_on_batch(real_notes, labels_real)
            d_loss_fake = discriminator.train_on_batch(
                generated_notes, labels_fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            labels_gan = np.ones((batch_size, 1))

            g_loss = gan.train_on_batch(noise, labels_gan)

            print(
                f"[D loss: {d_loss[0]} | D accuracy: {100 * d_loss[1]}] [G loss: {g_loss}]")
            visualize_interval = 5
            if (epoch + 1) % visualize_interval == 0:
                visualize_generated_samples(generator, epoch + 1)


if __name__ == "__main__":
    dataset_path = "Composer data"
    output_file_path = os.path.join('parsed_notes.json')

    # Load parsed notes from file
    notes = load_notes_from_file(output_file_path)

    # Print the first 10 notes to inspect
    print("Sample notes data:", notes[:10])

    # Prepare sequences for training
    sequence_length = 30
    network_input, network_output = prepare_sequences(notes, sequence_length)

    # Print the shape of the input data before training
    print("Shape of network_input:", network_input.shape)

    # Ensure the shape of the input data matches the expected shape for the LSTM layers
    # Assuming 4 features per timestep
    print("Expected input shape for LSTM layers:", (None, sequence_length, 4))

    # Set parameters for the model
    latent_dim = 100
    # Set different learning rates for generator and discriminator
    # Lower learning rate for the generator
    generator_optimizer = Adam(learning_rate=0.001, beta_1=0.5)
    # Higher learning rate for the discriminator
    discriminator_optimizer = Adam(learning_rate=0.0001, beta_1=0.5)

    pitchnames = sorted(set(note.get('pitch', '') or '.'.join(
        note.get('pitches', [])) for note in notes))
    n_notes = len(pitchnames)

    # Build and compile the discriminator
    discriminator = build_discriminator(sequence_length, n_notes)

    # Print discriminator summary
    print("\nDiscriminator Summary:")
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=discriminator_optimizer, metrics=['accuracy'])

    # Build the generator
    generator = build_generator(latent_dim, sequence_length, n_notes)

    print("Generator Summary:")
    generator.summary()

    # Build and compile the GAN
    discriminator.trainable = False
    gan_input = Input(shape=(latent_dim,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    gan = Model(gan_input, gan_output)
    gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

    # Train the GAN
    train_gan(generator, discriminator, gan, network_input,
              network_output, epochs=5, batch_size=256)
    # Save the models
    generator.save(
        "generator_model.keras")
    discriminator.save(
        "discriminator_model.keras")
    gan.save("gan_model.keras")
