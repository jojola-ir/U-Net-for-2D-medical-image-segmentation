import tensorflow as tf
import tensorflow.keras.backend as K


def dice_coeff(targets, inputs, smooth=1):
    # Flatten
    y_true_f = tf.reshape(targets, [-1])
    y_pred_f = tf.reshape(inputs, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)

    return score


def recall(targets, inputs):
    true_positives = K.sum(K.round(K.clip(targets * inputs, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(targets, 0, 1)))

    return true_positives / (possible_positives + K.epsilon())


def specificity(targets, inputs):
    tn = K.sum(K.round(K.clip((1 - targets) * (1 - inputs), 0, 1)))
    fp = K.sum(K.round(K.clip((1 - targets) * inputs, 0, 1)))
    return tn / (tn + fp + K.epsilon())


def tversky(targets, inputs):
    smooth = tf.keras.backend.epsilon()
    targets_pos = tf.keras.backend.flatten(targets)
    inputs_pos = tf.keras.backend.flatten(inputs)
    true_pos = tf.keras.backend.sum(targets_pos * inputs_pos)
    false_neg = tf.keras.backend.sum(targets_pos * (1 - inputs_pos))
    false_pos = tf.keras.backend.sum((1 - targets_pos) * inputs_pos)
    alpha = 0.7
    return (true_pos + smooth) / (true_pos + alpha * false_neg + (1 - alpha) * false_pos + smooth)
