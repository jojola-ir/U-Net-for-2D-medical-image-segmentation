import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy

from metrics import dice_coeff, tversky


def dice_loss(targets, inputs):
    loss = 1 - dice_coeff(targets, inputs)
    return loss


def log_cosh_dice_loss(targets, inputs):
    x = dice_loss(targets, inputs)
    return tf.math.log((tf.exp(x) + tf.exp(-x)) / 2.0)


def weighted_cross_entropy(targets, inputs, beta=0.2):
    weight_a = beta * tf.cast(targets, tf.float32)
    weight_b = 1 - tf.cast(targets, tf.float32)
    o = (tf.math.log1p(tf.exp(-tf.abs(inputs))) + tf.nn.relu(-inputs)) * (weight_a + weight_b) + inputs * weight_b
    return tf.reduce_mean(o)


def bce_dice_loss(targets, inputs):
    loss = BinaryCrossentropy() + dice_loss(targets, inputs)
    return loss


def wce_dice_loss(targets, inputs):
    loss = weighted_cross_entropy(targets, inputs) + dice_loss(targets, inputs)
    return loss


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, inputs):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - inputs) ** gamma * targets
        weight_b = (1 - alpha) * inputs ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
                weight_a + weight_b) + logits * weight_b

    def loss(targets, logits):
        inputs = tf.math.sigmoid(logits)
        loss = focal_loss_with_logits(logits=logits, targets=targets, alpha=alpha, gamma=gamma, inputs=inputs)

        return tf.reduce_mean(loss)

    return loss


def tversky_loss(targets, inputs):
    return 1 - tversky(targets, inputs)


def focal_tversky_loss(targets, inputs):
    pt_1 = tversky_loss(targets, inputs)
    gamma = 0.75
    return tf.keras.backend.pow((1 - pt_1), gamma)
