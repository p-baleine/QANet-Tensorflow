import tensorflow as tf

from .core import ExpandDims, Squeeze

class MultipleSeparableConv1D(tf.keras.models.Model):
    def __init__(self,
                 num_layers,
                 filters,
                 width,
                 padding,
                 activation,
                 **kwargs):
        super(MultipleSeparableConv1D, self).__init__(**kwargs)

        self.input_reshape = ExpandDims(1)

        for l in range(num_layers):
            layer = tf.keras.layers.SeparableConv2D(
                filters=filters,
                kernel_size=(1, width),
                padding=padding,
                activation=activation)
            setattr(self, 'conv_{}'.format(l), layer)

        self.output_reshape = Squeeze(1)

    def call(self, x):
        # (batch_size, 1, N, input_dim)
        x = self.input_reshape(x)

        for layer in self.layers[1:-1]:
            # (batch_size, 1, N, out_dim)
            x = layer(x)

        # (batch_size, N, out_dim)
        return self.output_reshape(x)

    def compute_output_shape(self, input_shape):
        return input_shape
