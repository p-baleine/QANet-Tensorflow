import tensorflow as tf

VERY_NEGATIVE_NUMBER = - 1e30

def exp_mask(val, mask):
    """maskに従ってvalに大きな負の値を足し込む
    softmax向けのmask

    参考: https://github.com/allenai/bi-att-flow/blob/master/my/tensorflow/general.py#L104
    """
    return val + (1. - tf.cast(mask, tf.float32)) * VERY_NEGATIVE_NUMBER

def layer_dropout(x, y, layer_idx, num_total_layers, p_L, training):
    if not training:
        return x + y

    survival_prob = 1. - (layer_idx / num_total_layers) * (1. - p_L)
    is_decayed = tf.random_uniform([]) < (1.0 - survival_prob)
    return tf.cond(is_decayed, lambda: x, lambda: x + y)

