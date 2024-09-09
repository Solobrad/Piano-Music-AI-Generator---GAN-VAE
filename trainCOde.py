import os
import json
import numpy as np
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import Dense, LSTM, BatchNormalization, Reshape, Input, concatenate
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


def prepare_sequences(notes, sequence_length=30):
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number)
                       for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input to be suitable for LSTM
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    network_input = network_input / float(len(pitchnames))
    network_output = np.array(network_output)

    return network_input, network_output, pitchnames


def build_generator(latent_dim, sequence_length, n_notes):
    model = Sequential()
    model.add(Dense(256, input_dim=latent_dim))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Dense(sequence_length * 1, activation='relu'))  # Set units to 1
    model.add(Reshape((sequence_length, 1)))
    return model


def build_discriminator(sequence_length, n_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape=(
        sequence_length, 1), return_sequences=True))
    model.add(LSTM(512, return_sequences=True))
    model.add(Dense(256))
    model.add(Dense(1, activation='sigmoid'))
    return model


def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    return model


def train_gan(generator, discriminator, gan, network_input, network_output, epochs=2, batch_size=256):
    # Change the number of batches per epoch by subsetting the data
    subset_size = 50000  # Choose the desired subset size
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
    dataset_path = r"C:\\Users\\ASUS\\Desktop\\AI\\AI piano Music GAN\\Composer data"
    output_file_path = os.path.join(
        r'C:\Users\ASUS\Desktop\AI\AI piano Music GAN', 'parsed_notes.json')

    # Load parsed notes from file
    notes = load_notes_from_file(output_file_path)

    # Prepare sequences for training
    sequence_length = 30
    network_input, network_output, pitchnames = prepare_sequences(
        notes, sequence_length)

    # Set parameters for the model
    latent_dim = 100
    optimizer = tf.keras.optimizers.legacy.Adam(
        learning_rate=0.0002, beta_1=0.5)

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
              network_output, epochs=2, batch_size=256)
    # Save the models
    generator.save(
        "C:\\Users\\ASUS\\Desktop\\AI\\AI piano Music GAN\\generator_model.h5")
    discriminator.save(
        "C:\\Users\\ASUS\\Desktop\\AI\\AI piano Music GAN\\discriminator_model.h5")
    gan.save("C:\\Users\\ASUS\\Desktop\\AI\\AI piano Music GAN\\gan_model.h5")
