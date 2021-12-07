#%%

import tensorflow as tf
import tensorflow.keras.layers as L

#%%


class ConvBlock(L.Layer):
    def __init__(self, filters: int, kernel_size: tuple([int, int]), activation: str):
        super().__init__()
        self.conv = L.Conv2D(
            filters,
            kernel_size=kernel_size,
            activation=None,
            padding="same",
            use_bias=False,
        )
        self.batchnorm = L.BatchNormalization()
        self.act = L.Activation(activation)

    def call(self, inputs):
        inner = self.conv(inputs)
        inner = self.batchnorm(inner)
        return self.act(inner)


class ResBlock(L.Layer):
    def __init__(self, filters: int, activation: str):
        super().__init__()
        self.conv_res = L.Conv2D(
            filters, kernel_size=(1, 1), activation=None, padding="same"
        )
        self.conv_1 = ConvBlock(filters, (3, 3), activation)
        self.conv_2 = ConvBlock(filters, (1, 1), activation)
        self.conv_3 = ConvBlock(filters, (3, 3), activation)

    def call(self, inputs):
        res = self.conv_res(inputs)
        inner = self.conv_1(inputs)
        inner = self.conv_2(inner)
        inner = self.conv_3(inner)
        return inner + res
