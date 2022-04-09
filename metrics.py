import tensorflow as tf


def dice_coeff(targets, inputs, smooth=1):
    # Flatten
    y_true_f = tf.reshape(targets, [-1])
    y_pred_f = tf.reshape(inputs, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return score
