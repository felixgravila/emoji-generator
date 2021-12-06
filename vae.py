#%%

from collections import defaultdict

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.python.keras.regularizers import L1
from tqdm import tqdm
import numpy as np

from load_emojis import get_all_emojis

#%%


all_emojis_dict = get_all_emojis()
emojis = all_emojis_dict["Appl"]
# emojis = []
# for k,v in all_emojis_dict.items():
#     emojis.extend(v)

# sanity check
sizes = defaultdict(int)
for emoji in tqdm(emojis):
    sizes[str(emoji.data.shape)] += 1

if len(sizes) == 1:
    print(f"All ok, only size is {list(sizes.keys())[0]}")
else:
    print(f"Needs fixing! Multiple sizes: {sizes}")
    enforced_size = (72, 72, 4)  # two emojis are (108,108,4)
    to_del_emojis = []
    for emoji in tqdm(emojis):
        if emoji.data.shape != enforced_size:
            to_del_emojis.append(emoji)
    for emoji in to_del_emojis:
        emojis.remove(emoji)

# %%

print("Taking the emoji datas and removing alpha channel")
emoji_datas = [e.data[:, :, :3] for e in emojis]

print("Converting to tf.dataset")
ds = tf.data.Dataset.from_tensor_slices(emoji_datas).shuffle(1000).batch(32)
# %%


class Sampling(L.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


latent_dim = 20

encoder_inputs = tf.keras.Input(shape=(72, 72, 3))
x = L.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)
x = L.Conv2D(32, (3, 3), activation="relu", strides=2, padding="same")(x)
x = L.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")(encoder_inputs)
x = L.Conv2D(64, (3, 3), activation="relu", strides=2, padding="same")(x)
x = L.Flatten()(x)
x = L.Dense(16, activation="relu")(x)
z_mean = L.Dense(latent_dim, name="z_mean")(x)
z_log_var = L.Dense(latent_dim, name="z_log_var")(x)
z = Sampling()([z_mean, z_log_var])
encoder = tf.keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")
encoder.summary()

latent_inputs = tf.keras.Input(shape=(latent_dim,))
x = L.Dense(18 * 18 * 64, activation="relu")(latent_inputs)
x = L.Reshape((18, 18, 64))(x)
x = L.Conv2DTranspose(64, (3, 3), activation="relu", strides=2, padding="same")(x)
x = L.Conv2DTranspose(32, (3, 3), activation="relu", strides=2, padding="same")(x)
decoder_outputs = L.Conv2DTranspose(3, (3, 3), activation="sigmoid", padding="same")(x)
decoder = tf.keras.Model(latent_inputs, decoder_outputs, name="decoder")
decoder.summary()

# %%


class VAE(tf.keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }


# %%

vae = VAE(encoder, decoder)
vae.compile(optimizer=tf.keras.optimizers.Adam())
vae.fit(ds, epochs=200)

# %%
