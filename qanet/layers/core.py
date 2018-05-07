import tensorflow as tf

from tensor2tensor.layers.common_attention import add_timing_signal_1d

from .layer_utils import exp_mask

class Highway(tf.keras.layers.Layer):
    def __init__(self,
                 dropout_rate=0.0,
                 is_training=True,
                 kernel_regularizer=None,
                 **kwargs):
        self._dropout_rate = dropout_rate
        self._is_training = is_training
        self._kernel_regularizer = kernel_regularizer
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]

        self._W_T = self.add_variable(
            'weight_transform',
            [d, d],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=self._kernel_regularizer)
        self._b_T = self.add_variable(
            'bias_transform',
            [d],
            initializer=tf.zeros_initializer())
        self._W = self.add_variable(
            'weight',
            [d, d],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=self._kernel_regularizer)
        self._b = self.add_variable(
            'bias',
            [d],
            initializer=tf.zeros_initializer())

    def call(self, x):
        T = tf.sigmoid(tf.tensordot(x, self._W_T, [[2], [0]]) + self._b_T)
        H = tf.nn.relu(tf.tensordot(x, self._W, [[2], [0]]) + self._b)
        if self._is_training:
            H = tf.nn.dropout(H, 1. - self._dropout_rate)
        return H * T + (1. - T) * x

    def compute_output_shape(self, input_shape):
        return input_shape

class HighwayNetwork(tf.keras.models.Model):
    def __init__(self,
                 num_layers,
                 dropout_rate=0.0,
                 is_training=True,
                 kernel_regularizer=None,
                 **kwargs):
        super(HighwayNetwork, self).__init__(**kwargs)

        for l in range(num_layers):
            setattr(self, 'highway_{}'.format(l), Highway(
                is_training=is_training,
                dropout_rate=dropout_rate,
                kernel_regularizer=kernel_regularizer))

    def call(self, x):
        y = x
        for layer in self.layers:
            y = layer(y)
        return y

    def get_config(self):
        return dict()

class PositionEncoding(tf.keras.layers.Lambda):
    def __init__(self, **kwargs):
        super(PositionEncoding, self).__init__(
            function=lambda x: add_timing_signal_1d(x),
            **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape

class PositionPrediction(tf.keras.layers.Layer):
    """positionの予測

    Input:
      M_a: (batch_size, N, dim * 4)
      M_b: (batch_size, N, dim * 4)
      context_mask: (batch_size, N)

    Output: (batch_size, N)
    """

    def __init__(self,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 regularizer=None,
                 **kwargs):
        self._initializer = initializer
        self._regularizer = regularizer
        super(PositionPrediction, self).__init__(**kwargs)

    def build(self, input_shape):
        M_a_shape, M_b_shape, _ = input_shape

        self._W = self.add_variable(
            'weight',
            [M_a_shape[-1] + M_b_shape[-1], 1],
            initializer=self._initializer,
            regularizer=self._regularizer)

        super(PositionPrediction, self).build(input_shape)

    def call(self, x):
        M_a, M_b, context_mask = x
        M = tf.concat([M_a, M_b], axis=2)
        # (batch_size, N)
        logits = tf.squeeze(tf.tensordot(M, self._W, [[2], [0]]))
        logits = exp_mask(logits, context_mask)
        # shape情報が落ちちゃうので明示的にreshapeしておく
        logits = tf.reshape(logits, [-1, M_a.shape[1]])
        return tf.nn.softmax(logits)

    def compute_output_shape(self, input_shape):
        M_a_shape, _, _ = input_shape
        return tf.TensorShape([M_a_shape[0], M_a.shape[1]])

class ExpandDims(tf.keras.layers.Lambda):
    def __init__(self, axis, **kwargs):
        super(ExpandDims, self).__init__(
            function=lambda x: tf.expand_dims(x, axis),
            **kwargs)

    def compute_output_shape(self, input_shape):
        # TODO 間違ってね？ってか間違ってても怒られないの？
        return input_shape

class Squeeze(tf.keras.layers.Lambda):
    def __init__(self, axis, **kwargs):
        super(Squeeze, self).__init__(
            function=lambda x: tf.squeeze(x, axis),
            **kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape
