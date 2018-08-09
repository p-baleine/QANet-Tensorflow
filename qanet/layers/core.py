import tensorflow as tf

from tensor2tensor.layers.common_attention import add_timing_signal_1d

from .layer_utils import exp_mask

class Highway(tf.keras.layers.Layer):
    def __init__(self, regularizer=None, dropout_rate=0.0, **kwargs):
        self._regularizer = regularizer
        self._dropout_rate = dropout_rate
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        d = input_shape[-1]

        self._W_T = self.add_variable(
            'weight_transform',
            [d, d],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=self._regularizer)
        self._b_T = self.add_variable(
            'bias_transform',
            [d],
            initializer=tf.zeros_initializer(),
            regularizer=self._regularizer)
        self._W = self.add_variable(
            'weight',
            [d, d],
            initializer=tf.contrib.layers.xavier_initializer(),
            regularizer=self._regularizer)
        self._b = self.add_variable(
            'bias',
            [d],
            initializer=tf.zeros_initializer(),
            regularizer=self._regularizer)

    def call(self, x, training):
        T = tf.sigmoid(tf.tensordot(x, self._W_T, [[2], [0]]) + self._b_T)
        H = tf.nn.relu(tf.tensordot(x, self._W, [[2], [0]]) + self._b)

        if training:
            H = tf.nn.dropout(H, keep_prob=1.0 - self._dropout_rate)

        return H * T + (1. - T) * x

    def compute_output_shape(self, input_shape):
        return input_shape

class HighwayNetwork(tf.keras.models.Model):
    def __init__(self,
                 num_layers,
                 regularizer=None,
                 dropout_rate=0.0,
                 **kwargs):
        super(HighwayNetwork, self).__init__(**kwargs)

        for l in range(num_layers):
            setattr(self, 'highway_{}'.format(l),
                    Highway(regularizer=regularizer,
                            dropout_rate=dropout_rate))

    def call(self, x, training):
        y = x
        for layer in self.layers:
            y = layer(y, training=training)
        return y

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
        logits = tf.reshape(logits, [-1, tf.shape(M_a)[1]])
        return logits

    def compute_output_shape(self, input_shape):
        M_a_shape, _, _ = input_shape
        return tf.TensorShape([M_a_shape[0], M_a.shape[1]])

class FeedForward(tf.keras.models.Model):
    def __init__(self,
                 dim,
                 regularizer,
                 **kwargs):
        super(FeedForward, self).__init__(**kwargs)

        self.ff1 = tf.keras.layers.Dense(
            dim,
            activation='relu',
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            activity_regularizer=regularizer)
        self.ff2 = tf.keras.layers.Dense(
            dim,
            activation=None,
            kernel_regularizer=regularizer,
            bias_regularizer=regularizer,
            activity_regularizer=regularizer)

    def call(self, inputs):
        x = self.ff1(inputs)
        x = self.ff2(x)
        return x

