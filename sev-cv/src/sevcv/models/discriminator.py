"""
Minimal TensorFlow Discriminator (SEV-D) with conv stem and transformer head.
Suitable for CIFAR-10 32x32 to validate the training loop.
"""
from __future__ import annotations
import tensorflow as tf


class ConvStem(tf.keras.layers.Layer):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.seq = tf.keras.Sequential([
            tf.keras.layers.Conv2D(ch, 3, strides=2, padding="same"),
            tf.keras.layers.LeakyReLU(0.2),
            tf.keras.layers.Conv2D(ch * 2, 3, strides=2, padding="same"),
            tf.keras.layers.LeakyReLU(0.2),
        ])

    def call(self, x, training=False):
        return self.seq(x, training=training)


class TransformerHead(tf.keras.layers.Layer):
    def __init__(self, dim: int = 128, heads: int = 4, depth: int = 2):
        super().__init__()
        self.proj = tf.keras.layers.Conv2D(dim, 1)
        self.flatten = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]]))
        self.blocks = []
        for _ in range(depth):
            self.blocks.append(tf.keras.layers.LayerNormalization(epsilon=1e-6))
            self.blocks.append(tf.keras.layers.MultiHeadAttention(num_heads=heads, key_dim=dim // heads))
        self.pool = tf.keras.layers.GlobalAveragePooling1D()

    def call(self, x, training=False):
        x = self.proj(x)
        t = self.flatten(x)
        for i in range(0, len(self.blocks), 2):
            ln = self.blocks[i]
            attn = self.blocks[i + 1]
            t = t + attn(ln(t), ln(t), training=training)
        return self.pool(t)


class SEVDiscriminator(tf.keras.Model):
    def __init__(self, ch: int = 64):
        super().__init__()
        self.stem = ConvStem(ch)
        self.head = TransformerHead(dim=ch * 2, heads=4, depth=2)
        self.out = tf.keras.layers.Dense(1)

    def call(self, x, training=False):
        h = self.stem(x, training=training)
        h = self.head(h, training=training)
        logit = self.out(h)
        return logit


def build_discriminator(ch: int = 64) -> tf.keras.Model:
    return SEVDiscriminator(ch=ch)
