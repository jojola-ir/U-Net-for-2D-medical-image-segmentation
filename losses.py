import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

from metrics import dice_coeff


def dice_loss(targets, inputs):
    loss = 1 - dice_coeff(targets, inputs)
    return loss


def bce_dice_loss(targets, inputs):
    loss = BinaryCrossentropy(targets, inputs) + dice_loss(targets, inputs)
    return loss


def weighted_cross_entropy(beta):
    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)

        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        return tf.reduce_mean(o)

    return loss
