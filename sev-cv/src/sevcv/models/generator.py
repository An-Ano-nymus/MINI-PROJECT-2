"""
Minimal TensorFlow Generator (SEV-G) using ViT-like blocks with style modulation.
This is a simplified backbone to get CIFAR-10 32x32 working as Phase 0.
"""
from __future__ import annotations
import tensorflow as tf


class StyleMLP(tf.keras.layers.Layer):
    def __init__(self, latent_dim: int, channels: int, name: str | None = None):
        super().__init__(name=name)
        self.fc = tf.keras.layers.Dense(channels * 2)

    def call(self, z):
        # returns (scale, shift)
        h = self.fc(z)
        scale, shift = tf.split(h, 2, axis=-1)
        return scale, shift


class ModulatedDense(tf.keras.layers.Layer):
    def __init__(self, units: int, name: str | None = None):
        super().__init__(name=name)
        self.units = units
        self.dense = tf.keras.layers.Dense(units)

    def call(self, x, scale, shift):
        y = self.dense(x)
        # scale and shift broadcasting
        return y * tf.expand_dims(scale, axis=1) + tf.expand_dims(shift, axis=1)


class MLPBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0, name: str | None = None):
        super().__init__(name=name)
        hidden = int(dim * mlp_ratio)
        self.fc1 = tf.keras.layers.Dense(hidden)
        self.act = tf.keras.layers.GELU()
        self.fc2 = tf.keras.layers.Dense(dim)
        self.drop = tf.keras.layers.Dropout(drop)

    def call(self, x, training=False):
        h = self.fc1(x)
        h = self.act(h)
        h = self.drop(h, training=training)
        h = self.fc2(h)
        h = self.drop(h, training=training)
        return h


class WindowAttention(tf.keras.layers.Layer):
    def __init__(self, dim: int, num_heads: int, name: str | None = None):
        super().__init__(name=name)
        self.attn = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=dim // num_heads)
        self.norm_q = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm_kv = tf.keras.layers.LayerNormalization(epsilon=1e-6)

    def call(self, x, training=False):
        # x: [B, N, C]
        q = self.norm_q(x)
        kv = self.norm_kv(x)
        y = self.attn(q, kv, training=training)
        return y


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0, drop: float = 0.0, name: str | None = None):
        super().__init__(name=name)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.attn = WindowAttention(dim, heads)
        self.drop = tf.keras.layers.Dropout(drop)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.mlp = MLPBlock(dim, mlp_ratio, drop)

    def call(self, x, training=False):
        h = self.attn(self.norm1(x), training=training)
        x = x + self.drop(h, training=training)
        h = self.mlp(self.norm2(x), training=training)
        x = x + self.drop(h, training=training)
        return x


class PatchUpsample(tf.keras.layers.Layer):
    def __init__(self, in_ch: int, out_ch: int, name: str | None = None):
        super().__init__(name=name)
        self.conv = tf.keras.layers.Conv2DTranspose(out_ch, kernel_size=4, strides=2, padding="same")
        self.act = tf.keras.layers.GELU()

    def call(self, x, training=False):
        x = self.conv(x)
        return self.act(x)


class SEVGenerator(tf.keras.Model):
    def __init__(self, latent_dim: int = 128, base_dim: int = 256, img_size: int = 32, img_channels: int = 3,
                 depth: int = 4, heads: int = 4, name: str | None = None):
        super().__init__(name=name)
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.base_dim = base_dim

        # map latent to a low-res grid, start at 8x8 for 32x32 target
        self.fc = tf.keras.layers.Dense(8 * 8 * base_dim)
        self.reshape = tf.keras.layers.Reshape((8, 8, base_dim))

        self.ups1 = PatchUpsample(base_dim, base_dim // 2)
        self.ups2 = PatchUpsample(base_dim // 2, base_dim // 4)

        # Transformer over flattened tokens
        self.to_tokens = tf.keras.layers.Lambda(lambda x: tf.reshape(x, [tf.shape(x)[0], -1, tf.shape(x)[-1]]))
        self.blocks = [TransformerBlock(dim=base_dim // 4, heads=heads) for _ in range(depth)]
        self.to_feat = tf.keras.layers.Lambda(
            lambda x: tf.reshape(x, [tf.shape(x)[0], self.img_size, self.img_size, tf.shape(x)[-1]])
        )

        self.out = tf.keras.layers.Conv2D(filters=img_channels, kernel_size=1, padding="same", activation="tanh")

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


def build_generator(latent_dim: int = 128, img_size: int = 32, channels: int = 3) -> tf.keras.Model:
    return SEVGenerator(latent_dim=latent_dim, img_size=img_size, img_channels=channels)
