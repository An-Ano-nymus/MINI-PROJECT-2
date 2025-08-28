from __future__ import annotations
import tensorflow as tf
import tensorflow_datasets as tfds


def preprocess(example, img_size=32):
    image = tf.cast(example["image"], tf.float32)
    image = tf.image.resize(image, (img_size, img_size), method="bilinear")
    image = (image / 127.5) - 1.0
    return image


def make_cifar10(batch_size: int = 128, img_size: int = 32, shuffle: bool = True, split: str = "train"):
    ds = tfds.load("cifar10", split=split, as_supervised=False, shuffle_files=True)
    if shuffle:
        ds = ds.shuffle(10_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda ex: preprocess(ex, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds
