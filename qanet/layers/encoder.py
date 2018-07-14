import tensorflow as tf

from .attention import MultiHeadAttention
from .core import PositionEncoding
from .wrappers import ResidualNormed, LayerDropped

class Encoder(tf.keras.models.Model):
    """Encoder

    引数:
      dim: 出力のサイズ
      filter_size: 畳み込みのfilterサイズ
      num_conv_layers: 畳み込みの層数

    Input:
      x: (batch_size, input_size, input_dim)

    Output: (batch_size, input_size, dim)
    """

    def __init__(self,
                 dim,
                 filter_size,
                 num_conv_layers,
                 num_heads,
                 layer_dropout_survival_prob,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        num_total_layers = num_conv_layers + 2
        layer_idx = 0

        self.position_encoding = PositionEncoding()

        # Convolution-layer
        self.conv_layers = []

        layer_idx += 1

        for idx in range(num_conv_layers):
            layer = LayerDropped(
                ResidualNormed(
                    tf.keras.layers.SeparableConv2D(
                        filters=dim,
                        kernel_size=(1, filter_size),
                        padding='same',
                        activation='relu')),
                layer_idx=layer_idx,
                num_total_layers=num_total_layers,
                p_L=layer_dropout_survival_prob)
            self.conv_layers.append(layer)
            setattr(self, 'separable_conv-{}'.format(idx), layer)
            layer_idx += 1

        # Self-attention-layer
        self.attention = LayerDropped(
            ResidualNormed(
                MultiHeadAttention(
                    num_heads=num_heads,
                    input_dim=dim,
                    d_k=dim,
                    d_v=dim)),
            layer_idx=layer_idx,
            num_total_layers=num_total_layers,
            p_L=layer_dropout_survival_prob)

        layer_idx += 1

        # Feed-forward-layer
        self.feed_forward = LayerDropped(
            ResidualNormed(
                tf.keras.layers.Dense(
                    dim,
                    activation='relu')),
            layer_idx=layer_idx,
            num_total_layers=num_total_layers,
            p_L=layer_dropout_survival_prob)

    def call(self, inputs, training):
        x, mask = inputs

        # (batch_size, N, input_dim)
        x = self.position_encoding(x)

        # (batch_size, 1, N, input_dim)
        x = tf.expand_dims(x, axis=1)

        for conv in self.conv_layers:
            # (batch_size, 1, N, dim)
            x = conv(x, training=training)

        # (batch_size, N, dim)
        x = tf.squeeze(x, axis=1)

        # (batch_size, N, dim)
        x = self.attention([x] * 3 + [mask], training=training)

        # (batch_size, N, dim)
        x = self.feed_forward(x, training=training)

        return x
