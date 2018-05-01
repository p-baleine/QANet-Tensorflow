import tensorflow as tf

VERY_NEGATIVE_NUMBER = - 1e30

def exp_mask(val, mask):
    """maskに従ってvalに大きな負の値を足し込む
    softmax向けのmask

    参考: https://github.com/allenai/bi-att-flow/blob/master/my/tensorflow/general.py#L104
    """
    return val + (1. - tf.cast(mask, tf.float32)) * VERY_NEGATIVE_NUMBER
