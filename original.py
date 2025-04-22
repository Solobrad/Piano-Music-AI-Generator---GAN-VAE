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

        # Process sequences within this song only
        for i in range(len(notes) - sequence_length):
            sequence_in = notes[i:i + sequence_length]
            sequence_out = notes[i + sequence_length]

            # Process the sequence as before...
            input_data = []
            for item in sequence_in:
                timestep_features = [0, 0, 0, 0]
                if 'pitch' in item:
                    timestep_features[0] = pitch_to_midi(item['pitch'])
                    timestep_features[1] = item['offset']
                    timestep_features[2] = item['duration']
                    timestep_features[3] = item['dynamic']
                elif 'pitches' in item:
                    pitches = []
                    for p in item['pitches']:
                        if '.' in p:
                            pitches.extend(p.split('.'))
                        else:
                            pitches.append(p)

                    midi_values = [pitch_to_midi(p) for p in pitches if p]
                    if midi_values:
                        timestep_features[0] = np.mean(midi_values)

                    timestep_features[1] = item['offset']
                    timestep_features[2] = item['duration']
                    timestep_features[3] = item['dynamic']

                input_data.append(timestep_features)

            network_input.append(input_data)

            # Process the output
            if 'pitch' in sequence_out:
                network_output.append(pitch_to_midi(sequence_out['pitch']))
            elif 'pitches' in sequence_out:
                pitches = []
                for p in sequence_out['pitches']:
                    if '.' in p:
                        pitches.extend(p.split('.'))
                    else:
                        pitches.append(p)

                midi_values = [pitch_to_midi(p) for p in pitches if p]
                if midi_values:
                    network_output.append(np.mean(midi_values))
                else:
                    network_output.append(0)

    # Convert to numpy arrays and reshape
    network_input = np.array(network_input)
    network_output = np.array(network_output)

    # Reshape if needed
    if network_input.shape[1] == sequence_length and network_input.shape[2] == 4:
        # Already in the right shape
        pass
    else:
        network_input = pad_or_truncate_input(network_input, sequence_length)
        network_input = np.array(network_input).reshape(-1, sequence_length, 4)

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
    encoder_input = Input(shape=(sequence_length, 4), name="encoder_input")
    x = LSTM(128, return_sequences=False)(encoder_input)
    latent_vector = Dense(latent_dim, name="latent_vector")(x)
    encoder = Model(encoder_input, latent_vector, name="encoder")

    # Decoder
    decoder_input = Input(shape=(latent_dim,), name="decoder_input")
    x = Dense(sequence_length * 4)(decoder_input)
    decoder_output = Reshape((sequence_length, 4), name="decoder_output")(x)
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


def train_gan_vae_hybrid(encoder, decoder, discriminator, network_input, sequence_length, batch_size, num_epochs=1):
    bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    optimizer = tf.keras.optimizers.Adam(1e-4)
    decoder.compile(loss='mse', optimizer=optimizer)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer)

    num_sequences = len(network_input)
    num_batches = num_sequences // batch_size

    if num_batches == 0:
        print(
            f"Warning: Not enough data for even one batch. Have {num_sequences} sequences but need at least {batch_size}.")
        return

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")

        # Shuffle the data
        indices = np.arange(num_sequences)
        np.random.shuffle(indices)

        d_losses = []
        g_losses = []

        for batch_i in tqdm(range(num_batches), desc="Batch Progress"):
            # Get batch indices
            idx = indices[batch_i * batch_size:(batch_i + 1) * batch_size]

            # Get real samples for this batch
            real_notes = network_input[idx]

            # Apply light noise
            noisy_notes = real_notes + \
                np.random.normal(0, 0.05, real_notes.shape)

            # Create labels
            labels_real = np.ones((batch_size, 1)) * 0.9
            labels_fake = np.zeros((batch_size, 1))

            # Encode real notes to latent
            latent_space = encoder.predict(noisy_notes, verbose=0)

            # Decode to fake notes
            generated_notes = decoder.predict(latent_space, verbose=0)

            # Train discriminator
            d_loss_real = discriminator.train_on_batch(real_notes, labels_real)
            d_loss_fake = discriminator.train_on_batch(
                generated_notes, labels_fake)
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            d_losses.append(d_loss)

            # Train generator (via decoder)
            misleading_labels = np.ones((batch_size, 1))
            g_loss = discriminator.train_on_batch(
                generated_notes, misleading_labels)
            g_losses.append(g_loss)

            # Optionally visualize samples every N batches
            if batch_i % 100 == 0:
                print(
                    f"  Batch {batch_i}/{num_batches}, D loss: {d_loss:.4f}, G loss: {g_loss:.4f}")

        # Print epoch stats
        print(
            f"Epoch {epoch+1}/{num_epochs}, Avg D loss: {np.mean(d_losses):.4f}, Avg G loss: {np.mean(g_losses):.4f}")


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
    print("Expected input shape for LSTM layers:", (None, sequence_length, 4))

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
    subset_size = 200000
    train_gan_vae_hybrid(encoder, decoder, discriminator, network_input[:subset_size],
                         sequence_length=sequence_length, batch_size=256, num_epochs=30)

    # Save the models
    generator.save("Trained files/generator_model.keras")
    discriminator.save("Trained files/discriminator_model.keras")
    gan.save("Trained files/gan_model.keras")
