from __future__ import annotations
import tensorflow as tf
import tensorflow_datasets as tfds


def _center_crop(image: tf.Tensor, target_size: int) -> tf.Tensor:
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]
    side = tf.minimum(h, w)
    offset_h = (h - side) // 2
    offset_w = (w - side) // 2
    image = tf.image.crop_to_bounding_box(image, offset_h, offset_w, side, side)
    image = tf.image.resize(image, (target_size, target_size), method="bilinear")
    return image


def preprocess(example: dict, img_size: int = 128, center_crop: bool = True) -> tf.Tensor:
    image = tf.cast(example["image"], tf.float32)
    if center_crop:
        image = _center_crop(image, img_size)
    else:
        image = tf.image.resize(image, (img_size, img_size), method="bilinear")
    image = (image / 127.5) - 1.0
    return image


def make_coco2017(
    split: str = "train",
    batch_size: int = 64,
    img_size: int = 128,
    shuffle: bool = True,
    center_crop: bool = True,
):
    ds = tfds.load("coco/2017", split=split, as_supervised=False, shuffle_files=True)
    if shuffle:
        ds = ds.shuffle(20_000, reshuffle_each_iteration=True)
    ds = ds.map(lambda ex: preprocess(ex, img_size, center_crop), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
    return ds
