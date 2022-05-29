import math
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

from sklearn.utils.extmath import cartesian

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


def BinaryCrossEntropy(y_true, y_pred):
    y_pred = tf.keras.backend.clip(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
    term_0 = (1 - y_true) * tf.keras.backend.log(1 - y_pred + tf.keras.backend.epsilon())
    term_1 = y_true * tf.keras.backend.log(y_pred + tf.keras.backend.epsilon())
    return -tf.keras.backend.mean(term_0 + term_1, axis=0)


def bce_dice_loss(targets, inputs):
    loss = BinaryCrossEntropy(targets, inputs) + dice_loss(targets, inputs)
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


IMG_SIZE = 160

def cdist(A, B):
    """
    Computes the pairwise Euclidean distance matrix between two tensorflow matrices A & B, similiar to scikit-learn cdist.
    For example:
    A = [[1, 2],
         [3, 4]]
    B = [[1, 2],
         [3, 4]]
    should return:
        [[0, 2.82],
         [2.82, 0]]
    :param A: m_a x n matrix
    :param B: m_b x n matrix
    :return: euclidean distance matrix (m_a x m_b)
    """
    # squared norms of each row in A and B
    na = tf.reduce_sum(tf.square(A), 1)
    nb = tf.reduce_sum(tf.square(B), 1)

    # na as a row and nb as a co"lumn vectors
    na = tf.reshape(na, [-1, 1])
    nb = tf.reshape(nb, [1, -1])

    # return pairwise euclidead difference matrix
    D = tf.sqrt(tf.maximum(na - 2 * tf.matmul(A, B, False, True) + nb, 0.0))
    return D


def weighted_hausdorff_distance(y_true, y_pred, alpha=0):
    w = IMG_SIZE
    h = IMG_SIZE
    max_dist = math.sqrt(w ** 2 + h ** 2)
    all_img_locations = tf.convert_to_tensor(cartesian([np.arange(w), np.arange(h)]), dtype=tf.float32)

    def hausdorff_loss(y_true, y_pred):
        def loss(y_true, y_pred):
            eps = 1e-6
            y_true = K.reshape(y_true, [w, h])
            gt_points = K.cast(tf.where(y_true > 0.5), dtype=tf.float32)
            num_gt_points = tf.shape(gt_points)[0]
            y_pred = K.flatten(y_pred)
            p = y_pred
            p_replicated = tf.squeeze(K.repeat(tf.expand_dims(p, axis=-1), num_gt_points))
            d_matrix = cdist(all_img_locations, gt_points)
            num_est_pts = tf.reduce_sum(p)
            term_1 = (1 / (num_est_pts + eps)) * K.sum(p * K.min(d_matrix, 1))

            d_div_p = K.min((d_matrix + eps) / (p_replicated ** alpha + (eps / max_dist)), 0)
            d_div_p = K.clip(d_div_p, 0, max_dist)
            term_2 = K.mean(d_div_p, axis=0)

            return term_1 + term_2

        batched_losses = tf.map_fn(lambda x:
                                   loss(x[0], x[1]),
                                   (y_true, y_pred),
                                   dtype=tf.float32)
        return K.mean(tf.stack(batched_losses))

    return hausdorff_loss(y_true, y_pred)