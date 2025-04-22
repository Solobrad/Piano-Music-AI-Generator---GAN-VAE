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


def prepare_sequences_with_song_boundaries(songs_data, sequence_length=30):
    network_input = []
    network_output = []

    for song in songs_data:
        notes = song["notes"]

        # Skip songs that are too short
        if len(notes) <= sequence_length:
            continue

        for i in range(len(notes) - sequence_length):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]

            input_data = []
            for item in sequence_in:
                # Start with 128 for pitches + 3 for other features
                timestep_features = [0] * 132
                if 'pitch' in item:
                    # This is a single note (not a chord)
                    midi_value = pitch_to_midi(item['pitch'])
                    if 0 <= int(midi_value) < 128:
                        timestep_features[int(midi_value)] = 1
                    timestep_features[128] = item['offset']
                    timestep_features[129] = item['duration']
                    timestep_features[130] = item['dynamic']
                elif 'pitches' in item:
                    # This is a chord (multiple notes at the same time)
                    for p in item['pitches']:
                        midi_value = pitch_to_midi(p)
                        if 0 <= int(midi_value) < 128:
                            timestep_features[int(midi_value)] = 1
                    timestep_features[128] = item['offset']
                    timestep_features[129] = item['duration']
                    timestep_features[130] = item['dynamic']

                input_data.append(timestep_features)

            network_input.append(input_data)

            # Process output (same as input)
            output_features = [0] * 132
            if 'pitch' in sequence_out:
                output_features[int(pitch_to_midi(sequence_out['pitch']))] = 1
            elif 'pitches' in sequence_out:
                for p in sequence_out['pitches']:
                    output_features[int(pitch_to_midi(p))] = 1
            output_features[128] = sequence_out['offset']
            output_features[129] = sequence_out['duration']
            output_features[130] = sequence_out['dynamic']

            network_output.append(output_features)

    network_input = np.array(network_input)
    network_output = np.array(network_output)

    # Ensure correct shape
    network_input = network_input.reshape(-1, sequence_length, 132)

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
    # Set units to 1
    model.add(Dense(sequence_length * 132, activation='relu'))
    model.add(Reshape((sequence_length, 132)))
    return model


def build_discriminator(sequence_length, n_notes):
    model = Sequential()
    model.add(LSTM(512, input_shape=(
        sequence_length, 132), return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(512, return_sequences=False))
    model.add(Dropout(0.4))
    model.add(Dense(1024, activation='relu'))
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


def build_vae(latent_dim, sequence_length):
    # Encoder
    # Adjust input shape if needed
    encoder_input = Input(shape=(sequence_length, 132), name="encoder_input")
    x = LSTM(128, return_sequences=False)(encoder_input)
    latent_vector = Dense(latent_dim, name="latent_vector")(x)
    encoder = Model(encoder_input, latent_vector, name="encoder")

    # Decoder
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    x = Dense(sequence_length * 132)(decoder_input)
    decoder_output = Reshape((sequence_length, 132), name="decoder_output")(x)
    decoder = Model(decoder_input, decoder_output, name="decoder")

    # VAE (Combining Encoder and Decoder)
    vae_input = encoder_input
    vae_output = decoder(encoder(vae_input))
    vae = Model(vae_input, vae_output, name="vae")

    return vae


def build_gan_vae_hybrid(vae, latent_dim, sequence_length, n_notes):
    if not hasattr(vae, 'get_layer'):
        raise ValueError(
            "The provided VAE model does not have the 'get_layer' method.")

    try:
        encoder = vae.get_layer("encoder")
        decoder = vae.get_layer("decoder")
    except ValueError as e:
        raise ValueError(
            "Encoder or Decoder layer not found in the VAE model") from e

    # Create generator using the decoder part of VAE
    generator_input = Input(shape=(latent_dim,))
    generator_output = decoder(generator_input)
    generator = Model(generator_input, generator_output, name="generator")

    # Create discriminator
    discriminator = build_discriminator(sequence_length, n_notes)

    # Build GAN using the generator and discriminator
    gan = build_gan(generator, discriminator)

    return generator, discriminator, gan


def train_gan_vae_hybrid(encoder, decoder, discriminator, gan,
                         network_input, sequence_length, batch_size, num_epochs=1):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(1e-4)

    # 🔄 changed: compile gan (generator + discriminator) for generator updates
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    num_sequences = len(network_input)
    num_batches = num_sequences // batch_size
    if num_batches == 0:
        print(f"Warning: need ≥{batch_size} sequences, have {num_sequences}.")
        return

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)
        d_losses, g_losses, recon_losses = [], [], []

        for batch_i in tqdm(range(num_batches), desc="Batch"):
            idx = indices[batch_i*batch_size:(batch_i+1)*batch_size]
            real_notes = network_input[idx]
            noisy_notes = real_notes + \
                np.random.normal(0, 0.05, real_notes.shape)

            # 🔄 changed: train decoder to reconstruct all 132 features (including offset & dynamic)
            recon_loss = vae.train_on_batch(noisy_notes, real_notes)

            recon_losses.append(recon_loss)

            # Encode / decode to get fake samples
            latent = encoder.predict(noisy_notes, verbose=0)
            fake_notes = decoder.predict(latent, verbose=0)

            # Train discriminator
            labels_real = np.ones((batch_size, 1))*0.9
            labels_fake = np.zeros((batch_size, 1))
            d_loss_real = discriminator.train_on_batch(real_notes, labels_real)
            d_loss_fake = discriminator.train_on_batch(fake_notes, labels_fake)
            d_losses.append(0.5*(d_loss_real+d_loss_fake))

            # 🔄 changed: train generator (via gan) to fool the discriminator
            misleading_labels = np.ones((batch_size, 1))
            g_loss = gan.train_on_batch(noise=np.random.normal(0, 1, (batch_size, latent.shape[-1])),
                                        y=misleading_labels)
            g_losses.append(g_loss)

            if batch_i % 100 == 0:
                print(
                    f" Batch {batch_i}/{num_batches}: D:{d_losses[-1]:.4f}, G:{g_losses[-1]:.4f}, R:{recon_loss:.4f}")

        print(
            f"Epoch {epoch+1} → Avg D: {np.mean(d_losses):.4f}, G: {np.mean(g_losses):.4f}, Recon: {np.mean(recon_losses):.4f}")


if __name__ == "__main__":
    dataset_path = "Composer data"
    output_file_path = os.path.join('parsed_notes.json')

    # Load parsed notes from file
    songs = load_notes_from_file(output_file_path)

    # Print the first 10 notes to inspect
    print("Sample song data:", songs[0]['notes'][:10])

    # Prepare sequences for training
    sequence_length = 30
    network_input, network_output = prepare_sequences_with_song_boundaries(
        songs, sequence_length)

    # Print the shape of the input data before training
    print("Shape of network_input:", network_input.shape)

    # Ensure the shape of the input data matches the expected shape for the LSTM layers
    # Assuming 4 features per timestep
    print("Expected input shape for LSTM layers:", (None, sequence_length, 132))

    # Set parameters for the model
    latent_dim = 100
    # Set different learning rates for generator and discriminator
    # Lower learning rate for the generator
    generator_optimizer = Adam(learning_rate=0.0005, beta_1=0.5)
    # Higher learning rate for the discriminator
    discriminator_optimizer = Adam(learning_rate=0.0005, beta_1=0.5)

    pitchnames = sorted(set(note.get('pitch', '') or '.'.join(
        note.get('pitches', [])) for note in songs))
    n_notes = len(pitchnames)

    # Build and compile the VAE
    vae = build_vae(latent_dim, sequence_length)

    # Build the hybrid GAN-VAE model
    generator, discriminator, gan = build_gan_vae_hybrid(
        vae, latent_dim, sequence_length, n_notes)

    # Compile the discriminator
    print("\nDiscriminator Summary:")
    discriminator.summary()
    discriminator.compile(loss='binary_crossentropy',
                          optimizer=discriminator_optimizer, metrics=['accuracy'])

    print("VAE Summary:")
    vae.summary()

    # Compile the GAN
    discriminator.trainable = False
    gan.compile(loss='binary_crossentropy', optimizer=generator_optimizer)

    # Train the GAN-VAE hybrid model
    # subset_size = network_input.shape[0]
    encoder = vae.get_layer("encoder")
    decoder = vae.get_layer("decoder")
    vae.compile(optimizer='adam', loss='mse')

    subset_size = 200000
    train_gan_vae_hybrid(encoder, decoder, discriminator, gan, network_input[:subset_size],
                         sequence_length=sequence_length, batch_size=256, num_epochs=30)

    # Save the models
    generator.save("Trained files/generator_model.keras")
    discriminator.save("Trained files/discriminator_model.keras")
    gan.save("Trained files/gan_model.keras")
