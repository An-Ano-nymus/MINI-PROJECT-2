from __future__ import annotations
import tensorflow as tf
from typing import Dict, Any
from sevcv.models import SEVGenerator, SEVDiscriminator
from sevcv.data.cifar10 import make_cifar10
from sevcv.evolution.controller import EvolutionController, Individual


class HingeGAN(tf.keras.Model):
    def __init__(self, g: tf.keras.Model, d: tf.keras.Model):
        super().__init__()
        self.g = g
        self.d = d

    def d_loss(self, x_real, z):
        x_fake = self.g(z, training=True)
        real_logit = self.d(x_real, training=True)
        fake_logit = self.d(x_fake, training=True)
        loss_real = tf.reduce_mean(tf.nn.relu(1.0 - real_logit))
        loss_fake = tf.reduce_mean(tf.nn.relu(1.0 + fake_logit))
        return loss_real + loss_fake

    def g_loss(self, z):
        x_fake = self.g(z, training=True)
        fake_logit = self.d(x_fake, training=True)
        return -tf.reduce_mean(fake_logit)


def train_phase0(
    steps: int = 2000,
    batch_size: int = 128,
    img_size: int = 32,
    z_dim: int = 128,
    evo: EvolutionController | None = None,
    log_every: int = 100,
):
    ds = make_cifar10(batch_size=batch_size, img_size=img_size)
    it = iter(ds)

    # Choose policy from evolution controller or defaults
    if evo is None:
        lr = 2e-4
        heads = 4
        depth = 4
    else:
        ind: Individual = evo.population[0]
        lr = ind.policy["lr"]
        heads = ind.micro["heads"]
        depth = ind.micro["depth"]

    g = SEVGenerator(latent_dim=z_dim, img_size=img_size, img_channels=3, depth=depth, heads=heads)
    d = SEVDiscriminator(ch=64)
    gan = HingeGAN(g, d)

    g_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.99)
    d_opt = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.0, beta_2=0.99)

    @tf.function(jit_compile=False)
    def d_step(x_real):
        z = tf.random.normal([tf.shape(x_real)[0], z_dim])
        with tf.GradientTape() as tape:
            loss = gan.d_loss(x_real, z)
        grads = tape.gradient(loss, gan.d.trainable_variables)
        d_opt.apply_gradients(zip(grads, gan.d.trainable_variables))
        return loss

    @tf.function(jit_compile=False)
    def g_step(bs: int):
        z = tf.random.normal([bs, z_dim])
        with tf.GradientTape() as tape:
            loss = gan.g_loss(z)
        grads = tape.gradient(loss, gan.g.trainable_variables)
        g_opt.apply_gradients(zip(grads, gan.g.trainable_variables))
        return loss

    for step in range(1, steps + 1):
        try:
            x_real = next(it)
        except StopIteration:
            it = iter(ds)
            x_real = next(it)
        d_loss = d_step(x_real)
        g_loss = g_step(tf.shape(x_real)[0])
        if step % log_every == 0:
            tf.print("step", step, "d_loss", d_loss, "g_loss", g_loss)

    # Return a simple fitness proxy: variance of generated samples to encourage diversity
    z = tf.random.normal([256, z_dim])
    samples = g(z, training=False)
    fitness = float(tf.math.reduce_std(samples).numpy())
    return {"fitness_proxy": fitness, "generator": g, "discriminator": d}
