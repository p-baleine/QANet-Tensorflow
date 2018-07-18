import tensorflow as tf

class ResidualNormed(tf.keras.layers.Wrapper):
    def __init__(self,
                 layer,
                 epsilon=1e-6,
                 regularizer=None,
                 **kwargs):
        self._epsilon = epsilon
        self._regularizer = regularizer
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
            initializer=tf.ones_initializer(),
            regularizer=self._regularizer)
        self._bias = self.add_variable(
            'bias',
            [filters],
            initializer=tf.zeros_initializer(),
            regularizer=self._regularizer)

        self.layer.build(input_shape)

        super(ResidualNormed, self).build()

    def call(self, inputs):
        x = inputs[0] if type(inputs) is list else inputs
        mean = tf.reduce_mean(x, axis=[-1], keep_dims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keep_dims=True)
        norm_x = (x - mean) * tf.rsqrt(variance + self._epsilon)
        norm_x = norm_x * self._scale + self._bias

        if type(inputs) is list:
            # On self attetion, inputs are (x, x, x, x_mask)
            return self.layer.call([norm_x] * (len(inputs) - 1) + [inputs[-1]]) + x
        else:
            return self.layer.call(norm_x) + x

    def compute_output_shape(self, input_shape):
        return input_shape

class LayerDropped(tf.keras.layers.Wrapper):
    def __init__(self, layer, layer_idx, num_total_layers, p_L, **kwargs):
        self._survival_prob = 1. - (layer_idx / num_total_layers) * (1. - p_L)
        super(LayerDropped, self).__init__(layer, **kwargs)

    def build(self, input_shape):
        self.layer.build(input_shape)
        super(LayerDropped, self).build()

    def call(self, inputs, training):
        output = self.layer.call(inputs)
        inputs = inputs[0] if type(inputs) is list else inputs

        if not training:
            return output

        is_decayed = tf.random_uniform([]) < (1.0 - self._survival_prob)

        return tf.cond(is_decayed, lambda: inputs, lambda: output)

    def compute_output_shape(self, input_shape):
        return input_shape
