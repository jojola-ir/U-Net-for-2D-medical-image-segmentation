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


def focal_loss(alpha=0.25, gamma=2):
    def focal_loss_with_logits(logits, targets, alpha, gamma, y_pred):
        targets = tf.cast(targets, tf.float32)
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (
                weight_a + weight_b) + logits * weight_b

    def loss(y_true, logits):
        y_pred = tf.math.sigmoid(logits)
        loss = focal_loss_with_logits(logits=logits, targets=y_true, alpha=alpha, gamma=gamma, y_pred=y_pred)

        return tf.reduce_mean(loss)

    return loss
