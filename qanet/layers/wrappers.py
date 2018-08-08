import tensorflow as tf

class LayerNormed(tf.keras.layers.Wrapper):
    def __init__(self,
                 layer,
                 epsilon=1e-6,
                 regularizer=None,
                 dropout_rate=0.0,
                 **kwargs):
        self._epsilon = epsilon
        self._regularizer = regularizer
        self._dropout_rate = dropout_rate
        super(LayerNormed, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        if type(input_shape) is list:
            # multiple input(multihead-attention)の時は先頭要素の次元
            filters = input_shape[0][-1]
        else:
            filters = input_shape[-1]

        self._scale = self.add_variable(
            'scale',
            [filters],
            initializer=tf.ones_initializer(),
            regularizer=self._regularizer)
        self._bias = self.add_variable(
            'bias',
            [filters],
            initializer=tf.zeros_initializer(),
            regularizer=self._regularizer)

        self.layer.build(input_shape)

        super(LayerNormed, self).build()

    def call(self, inputs, training):
        x = inputs[0] if type(inputs) is list else inputs
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + self._epsilon)
        norm_x = norm_x * self._scale + self._bias

        if type(inputs) is list:
            # On self attetion, inputs are (x, x, x, x_mask)
            output = self.layer.call([norm_x] * (len(inputs) - 1) + [inputs[-1]])
        else:
            output = self.layer.call(norm_x)

        if training:
            output = tf.nn.dropout(output, keep_prob=1.0 - self._dropout_rate)

        return output

    def compute_output_shape(self, input_shape):
        return input_shape
