#%%

from collections import defaultdict

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as L
from tqdm import tqdm
import numpy as np

from load_emojis import get_all_emojis
from model import ResBlock
from tqdm import tqdm

from sklearn.manifold import TSNE

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

print("Taking the emoji datas, keeping the alpha channel")
emoji_datas = [e.data[:, :, :] for e in emojis if int(e.desc.split("_")[0]) <= 942]

print("Converting to tf.dataset")
ds = tf.data.Dataset.from_tensor_slices(emoji_datas).shuffle(1000).batch(32)
# %%

LATENT_DIM = 16


class Sampling(L.Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class Encoder(tf.keras.Model):
    def __init__(self, latent_dim, input_shape, **kwargs):
        super().__init__(**kwargs)
        self._input_shape = input_shape
        self.def_layers = [
            ResBlock(8, "elu"),
            L.MaxPool2D((2, 2)),
            ResBlock(16, "elu"),
            ResBlock(16, "elu"),
            L.MaxPool2D((2, 2)),
            ResBlock(32, "elu"),
            ResBlock(32, "elu"),
            L.Flatten(),
            L.Dense(64, activation="elu"),
            L.Dense(64, activation="elu"),
            L.Dense(128, activation="elu"),
        ]
        self.z_mean = L.Dense(latent_dim, activation=None)
        self.z_log_var = L.Dense(latent_dim, activation=None)
        self.sampling = Sampling()

    def call(self, inputs):
        x = inputs
        for l in self.def_layers:
            x = l(x)

        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        z = self.sampling([z_mean, z_log_var])
        return [z_mean, z_log_var, z]

    def summary(self):
        input = L.Input(shape=self._input_shape)
        output = self.call(input)
        model = tf.keras.Model(inputs=input, outputs=output)
        model.summary()


enc = Encoder(LATENT_DIM, input_shape=(72, 72, 4))
enc.build(input_shape=(None, 72, 72, 4))
enc.summary()

#%%


class Decoder(tf.keras.Model):
    def __init__(self, latent_dim, **kwargs):
        super().__init__(**kwargs)
        self._input_shape = (latent_dim,)
        self.def_layers = [
            L.Dense(128, activation="elu"),
            L.Dense(128, activation="elu"),
            L.Dense(10368, activation="elu"),
            L.Reshape((18, 18, 32)),
            L.Conv2DTranspose(64, (2, 2), activation="elu", strides=2),
            ResBlock(64, "elu"),
            L.Conv2DTranspose(32, (2, 2), activation="elu", strides=2),
            ResBlock(32, "elu"),
            ResBlock(16, "elu"),
            L.Conv2D(4, (1, 1), activation="sigmoid"),
        ]

    def call(self, inputs):
        x = inputs
        for l in self.def_layers:
            x = l(x)
        return x

    def summary(self):
        input = L.Input(shape=self._input_shape)
        output = self.call(input)
        model = tf.keras.Model(inputs=input, outputs=output)
        model.summary()


dec = Decoder(latent_dim=16)
dec.build(input_shape=(None, 16))
dec.summary()

# %%


class VAE(tf.keras.Model):
    def __init__(self, latent_dim, input_shape, kl_loss_epsilon=0.1, **kwargs):
        super().__init__(**kwargs)
        self.kl_loss_epsilon = kl_loss_epsilon
        self._input_shape = input_shape
        self.encoder = Encoder(latent_dim, input_shape, **kwargs)
        self.decoder = Decoder(latent_dim, **kwargs)
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

    def call(self, inputs):
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        return z_mean, z_log_var, z, reconstruction

    def train_step(self, data):
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z, reconstruction = self.call(data)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.keras.losses.binary_crossentropy(data, reconstruction),
                    axis=(1, 2),
                )
            )
            # reconstruction_loss = tf.reduce_sum(tf.losses.mse(data, reconstruction))
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = (
                tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1)) * self.kl_loss_epsilon
            )
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

    # def build(self, batch_size=None):
    #     return super().build([batch_size] + list(self._input_shape))


# %%

vae = VAE(latent_dim=LATENT_DIM, input_shape=(72, 72, 4), kl_loss_epsilon=0.00001)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001))

#%%


vae.fit(
    ds,
    epochs=300,
    callbacks=[
        tf.keras.callbacks.ReduceLROnPlateau(monitor="loss", factor=0.75, patience=5)
    ],
)

# %%

x = next(iter(ds))
fig = plt.figure(figsize=(10, 20), facecolor="white")
for i, e in enumerate(x[:4]):
    orig = fig.add_subplot(4, 2, (i * 2) + 1)
    reconst = fig.add_subplot(4, 2, (i * 2) + 2)
    orig.imshow(e)
    _, _, z = vae.encoder(np.array([e]))
    r = vae.decoder(z)
    reconst.imshow(r[0])


# %%

encodings = []
ds = tf.data.Dataset.from_tensor_slices(emoji_datas).batch(128)
for e in tqdm(ds):
    _, _, z = vae.encoder(e)
    encodings.extend(z)


# %%


X = TSNE(n_components=2, learning_rate="auto", init="random").fit_transform(
    np.array(encodings)
)
plt.scatter(X[:, 0], X[:, 1])
# %%
