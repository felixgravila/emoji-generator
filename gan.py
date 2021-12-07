#%%

from collections import defaultdict
import time

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.layers as L
from tqdm import tqdm
import numpy as np
import os

from load_emojis import get_all_emojis
from model import ResBlock

from IPython import display

BATCH_SIZE = 16
NOISE_DIM = 100

#%%


all_emojis_dict = get_all_emojis()
emojis = all_emojis_dict["Appl"]
# emojis = []
# for k,v in all_emojis_dict.items():
#     emojis.extend(v)

enforced_size = (72, 72, 4)  # two emojis are (108,108,4)
to_del_emojis = []
for emoji in tqdm(emojis):
    if emoji.data.shape != enforced_size:
        to_del_emojis.append(emoji)
for emoji in to_del_emojis:
    emojis.remove(emoji)

print(f"Using {len(emojis)} emojis")

# %%

print("Taking the emoji datas, keeping the alpha channel")
emoji_datas = [e.data[:, :, :] for e in emojis if int(e.desc.split("_")[0]) <= 102]

print("Converting to tf.dataset")
ds = tf.data.Dataset.from_tensor_slices(emoji_datas).shuffle(102).batch(BATCH_SIZE)


#%%


class Generator(tf.keras.Model):
    def __init__(self, input_dim, **kwargs):
        super().__init__(**kwargs)
        self._input_shape = (input_dim,)
        self.def_layers = [
            L.Dense(9 * 9 * 32, activation="elu"),
            L.Reshape((9, 9, 32)),
            L.Conv2DTranspose(128, (2, 2), strides=(2, 2), use_bias=False),
            L.BatchNormalization(),
            L.LeakyReLU(),
            ResBlock(64, "elu"),
            ResBlock(64, "elu"),
            ResBlock(64, "elu"),
            L.Conv2DTranspose(128, (2, 2), strides=(2, 2), use_bias=False),
            L.BatchNormalization(),
            L.LeakyReLU(),
            ResBlock(64, "elu"),
            ResBlock(64, "elu"),
            ResBlock(64, "elu"),
            L.Conv2DTranspose(128, (2, 2), strides=(2, 2), use_bias=False),
            L.BatchNormalization(),
            L.LeakyReLU(),
            ResBlock(32, "elu"),
            ResBlock(32, "elu"),
            ResBlock(32, "elu"),
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


generator = Generator(input_dim=NOISE_DIM)
generator.build(input_shape=(None, NOISE_DIM))
generator.summary()


class Discriminator(tf.keras.Model):
    def __init__(self, input_shape, **kwargs):
        super().__init__(**kwargs)
        self._input_shape = input_shape
        self.def_layers = [
            ResBlock(32, "elu"),
            ResBlock(32, "elu"),
            L.MaxPool2D((2, 2)),
            ResBlock(64, "elu"),
            ResBlock(64, "elu"),
            L.MaxPool2D((2, 2)),
            ResBlock(128, "elu"),
            ResBlock(128, "elu"),
            L.MaxPool2D((2, 2)),
            ResBlock(128, "elu"),
            ResBlock(128, "elu"),
            L.Flatten(),
            L.Dense(512, activation="elu"),
            L.Dense(1, activation="sigmoid"),
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


discriminator = Discriminator(input_shape=(72, 72, 4))
discriminator.build(input_shape=(None, 72, 72, 4))
discriminator.summary()

# %%

cross_entropy = tf.keras.losses.BinaryCrossentropy()


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

#%%

EPOCHS = 10000
num_examples_to_generate = 36

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

#%%


def generate_and_save_images(model, epoch, test_input):
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10, 10))

    for i in range(predictions.shape[0]):
        plt.subplot(6, 6, i + 1)
        plt.imshow(predictions[i, :, :, :])
        plt.axis("off")

    plt.savefig(f"gan_images/image_at_epoch_{epoch:04d}.png")
    plt.show()


checkpoint_dir = "./gan_ckpts"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)


#%%


@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    return gen_loss, disc_loss


def train(dataset, epochs):

    losses = {
        "gen_loss": [],
        "disc_loss": [],
    }

    total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
    generator_loss_tracker = tf.keras.metrics.Mean(name="generator_loss")
    discriminator_loss_tracker = tf.keras.metrics.Mean(name="discriminator_loss")

    for epoch in range(epochs):

        total_loss_tracker.reset_states()
        generator_loss_tracker.reset_states()
        discriminator_loss_tracker.reset_states()

        start = time.time()

        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
            total_loss_tracker.update_state(gen_loss + disc_loss)
            generator_loss_tracker.update_state(gen_loss)
            discriminator_loss_tracker.update_state(disc_loss)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print(
            f"\
Time for epoch {epoch + 1} is {(time.time() - start):.02f}s.\
Total loss {total_loss_tracker.result().numpy():.05f}, \
Generator loss {generator_loss_tracker.result().numpy():.05f}, \
Discriminator loss {discriminator_loss_tracker.result().numpy():.05f}"
        )

        losses["gen_loss"].append(generator_loss_tracker.result().numpy())
        losses["disc_loss"].append(discriminator_loss_tracker.result().numpy())

    # Generate after the final epoch
    display.clear_output(wait=True)

    return losses


#%%

losses = train(ds, EPOCHS)

np.save("losses.npy", losses, allow_pickle=True)

#%%
