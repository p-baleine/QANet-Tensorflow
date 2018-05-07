import tensorflow as tf

class ResidualNormed(tf.keras.layers.Wrapper):
    def __init__(self,
                 layer,
                 epsilon=1e-6,
                 dropout_rate=0.0,
                 is_training=True,
                 **kwargs):
        self._epsilon = epsilon
        self._dropout_rate = dropout_rate
        self._is_training = is_training
        super(ResidualNormed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        if type(input_shape) is list:
            # multiple input(multihead-attention)の時は先頭要素の次元
            filters = input_shape[0][-1]
        else:
            filters = input_shape[-1]

        self._scale = self.add_variable(
            'scale',
            [filters],
            initializer=tf.ones_initializer())
        self._bias = self.add_variable(
            'bias',
            [filters],
            initializer=tf.zeros_initializer())

        self.layer.build(input_shape)

        super(ResidualNormed, self).build()

    def call(self, input):
        output = self.layer.call(input)
        x = input[0] if type(input) is list else input
        mean = tf.reduce_mean(output, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(output - mean), axis=[-1], keep_dims=True)
        norm_output = (output - mean) * tf.rsqrt(variance + self._epsilon)
        residual_output = x + (norm_output * self._scale + self._bias)
        return (tf.nn.dropout(residual_output, 1. - self._dropout_rate)
                if self._is_training else residual_output)

    def compute_output_shape(self, input_shape):
        return input_shape
