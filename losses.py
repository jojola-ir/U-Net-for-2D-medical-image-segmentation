import tensorflow as tf

def DiceLoss(targets, inputs, smooth=1e-6):

    # flatten label and prediction tensors
    inputs = tf.keras.layers.flatten(inputs)
    targets = tf.keras.layers.flatten(targets)

    intersection = tf.reduce_sum(targets * inputs)
    dice = (2*intersection + smooth) / (tf.reduce_sum(targets) + tf.reduce_sum(inputs) + smooth)

    return 1 - dice


def weighted_cross_entropy(beta):

    def loss(y_true, y_pred):
        weight_a = beta * tf.cast(y_true, tf.float32)
        weight_b = 1 - tf.cast(y_true, tf.float32)

        o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
        return tf.reduce_mean(o)

    return loss