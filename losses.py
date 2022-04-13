import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

from metrics import dice_coeff


def dice_loss(targets, inputs):
    loss = 1 - dice_coeff(targets, inputs)
    return loss


def bce_dice_loss(targets, inputs):
    loss = BinaryCrossentropy(targets, inputs) + dice_loss(targets, inputs)
    return loss


def weighted_cross_entropy(targets, inputs, beta=0.9):
    weight_a = beta * tf.cast(targets, tf.float32)
    weight_b = 1 - tf.cast(targets, tf.float32)

    o = (tf.math.log1p(tf.exp(-tf.abs(inputs))) + tf.nn.relu(-inputs)) * (weight_a + weight_b) + inputs * weight_b

    return tf.reduce_mean(o)
