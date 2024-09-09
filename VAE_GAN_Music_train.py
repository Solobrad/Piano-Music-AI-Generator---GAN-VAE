import os
import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, BatchNormalization, Reshape, Input, Concatenate
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm


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
                timestep_features[0] = item['pitch']
                timestep_features[1] = item['offset']
                timestep_features[2] = item['duration']
                timestep_features[3] = item['dynamic']
            elif 'pitches' in item:
                # If it's a chord, update the features for each pitch in the chord
                for chord_pitch in item['pitches']:
                    timestep_features[0] = chord_pitch
                    timestep_features[1] = item['offset']
                    timestep_features[2] = item['duration']
                    timestep_features[3] = item['dynamic']
        # Pad or truncate the input data to ensure each item has four features
            input_data.append(timestep_features)

        network_input.append(input_data)
        network_output.append(sequence_out.get('pitch', ''))

    network_input = np.array(network_input)
    print("Shape of network_input before reshaping:", network_input.shape)

    network_output = np.array(network_output)
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
    model.add(Dense(sequence_length * 1, activation='relu'))  # Set units to 1
    model.add(Reshape((sequence_length, 1)))
    return model


def build_discriminator(sequence_length, n_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape=(
        sequence_length, 4), return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dense(512))
    model.add(Dense(256))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def train_gan(generator, discriminator, gan, network_input, network_output, epochs=5, batch_size=256):
    # Change the number of batches per epoch by subsetting the data
    subset_size = 100000  # Choose the desired subset size
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for _ in tqdm(range(subset_size // batch_size), desc="Batch Progress"):
            idx = np.random.randint(0, network_input.shape[0], batch_size)
            real_notes = network_input[idx]
            labels_real = np.ones((batch_size, 1))

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
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=0.001, beta_1=0.5)

    pitchnames = sorted(set(note.get('pitch', '') or '.'.join(
        note.get('pitches', [])) for note in notes))
    n_notes = len(pitchnames)

    # Build and compile the discriminator
    discriminator = build_discriminator(sequence_length, n_notes)

    # Print discriminator summary
    print("\nDiscriminator Summary:")
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=optimizer, metrics=['accuracy'])

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
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    # Train the GAN
    train_gan(generator, discriminator, gan, network_input,
              network_output, epochs=5, batch_size=256)
    # Save the models
    generator.save(
        "C:\\Users\\ASUS\\Desktop\\AI\\Advance AI Music\\generator_model.h5")
    discriminator.save(
        "C:\\Users\\ASUS\\Desktop\\AI\\Advance AI Music\\discriminator_model.h5")
    gan.save("C:\\Users\\ASUS\\Desktop\\AI\\Advance AI Music\\gan_model.h5")
