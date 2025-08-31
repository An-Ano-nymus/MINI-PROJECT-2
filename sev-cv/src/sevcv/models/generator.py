"""
Minimal TensorFlow/Keras Generator (SEV-G) using ViT-like blocks.
Phase 0 target: CIFAR-10 32x32 sanity.
"""
from __future__ import annotations
import tensorflow as tf
import keras
from keras import layers


class StyleMLP(layers.Layer):
    def __init__(self, latent_dim: int, channels: int, name: str | None = None):
        super().__init__(name=name)
        self.fc = layers.Dense(channels * 2)

    def call(self, z):
        h = self.fc(z)
        scale, shift = tf.split(h, 2, axis=-1)
        return scale, shift


class ModulatedDense(layers.Layer):
    def __init__(self, units: int, name: str | None = None):
        super().__init__(name=name)
        self.units = units
        self.dense = layers.Dense(units)

    def call(self, x, scale, shift):
        y = self.dense(x)
        return y * tf.expand_dims(scale, axis=1) + tf.expand_dims(shift, axis=1)


class MLPBlock(layers.Layer):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, name: str | None = None):
        super().__init__(name=name)
        hidden = int(dim * mlp_ratio)
        self.fc1 = layers.Dense(hidden)
        self.act = layers.Activation(tf.nn.gelu)
        self.fc2 = layers.Dense(dim)
        self.drop = layers.Dropout(drop)

    def call(self, x, training=False):
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h, training=training)
        h = self.fc2(h)
        h = self.drop(h, training=training)
        return h


class WindowAttention(layers.Layer):
    def __init__(self, dim: int, num_heads: int, name: str | None = None):
        super().__init__(name=name)
        self.attn = layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.norm_q = layers.LayerNormalization(epsilon=1e-6)
        self.norm_kv = layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        q = self.norm_q(x)
        kv = self.norm_kv(x)
        y = self.attn(q, kv, training=training)
        return y


class TransformerBlock(layers.Layer):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, name: str | None = None):
        super().__init__(name=name)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.attn = WindowAttention(dim, heads)
        self.drop = layers.Dropout(drop)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLPBlock(dim, mlp_ratio, drop)

    def call(self, x, training=False):
        h = self.attn(self.norm1(x), training=training)
        x = x + self.drop(h, training=training)
        h = self.mlp(self.norm2(x), training=training)
        x = x + self.drop(h, training=training)
        return x


class PatchUpsample(layers.Layer):
    def __init__(self, in_ch: int, out_ch: int, name: str | None = None):
        super().__init__(name=name)
        self.conv = layers.Conv2DTranspose(out_ch, kernel_size=4, strides=2, padding="same")
        self.act = layers.Activation(tf.nn.gelu)

    def call(self, x, training=False):
        x = self.conv(x)
        return self.act(x)


class SEVGenerator(keras.Model):
    def __init__(self, latent_dim: int = 128, base_dim: int = 256, img_size: int = 32, img_channels: int = 3,
                 depth: int = 4, heads: int = 4, name: str | None = None):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.base_dim = base_dim
        if (img_size % 4) != 0:
            raise ValueError(f"img_size must be divisible by 4, got {img_size}")
        start_res = img_size // 4
        self._start_res = start_res

        # Start at (img_size//4)x(img_size//4) so two upsample stages reach img_size
        self.fc = layers.Dense(start_res * start_res * base_dim)
        self.reshape = layers.Reshape((start_res, start_res, base_dim))

        self.ups1 = PatchUpsample(base_dim, base_dim // 2)
        self.ups2 = PatchUpsample(base_dim // 2, base_dim // 4)

        # Transformer over flattened tokens
        self.to_tokens = layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]]))
        self.blocks = [TransformerBlock(dim=base_dim // 4, heads=heads) for _ in range(depth)]
        self.to_feat = layers.Lambda(
            lambda x: tf.reshape(x, [tf.shape(x)[0], self.img_size, self.img_size, tf.shape(x)[-1]])
        )

        self.out = layers.Conv2D(filters=img_channels, kernel_size=1, padding="same", activation="tanh")

    def call(self, z, training=False):
        h = self.fc(z)
        h = self.reshape(h)
        h = self.ups1(h, training=training)  # 16x16
        h = self.ups2(h, training=training)  # 32x32
        t = self.to_tokens(h)
        for blk in self.blocks:
            t = blk(t, training=training)
        f = self.to_feat(t)
        x = self.out(f)
        return x


def build_generator(latent_dim: int = 128, img_size: int = 32, channels: int = 3) -> keras.Model:
    return SEVGenerator(latent_dim=latent_dim, img_size=img_size, img_channels=channels)
