import tensorflow as tf

from .attention import MultiHeadAttention
from .convolutional import MultipleSeparableConv1D
from .core import PositionEncoding, ExpandDims, Squeeze
from .wrappers import ResidualNormed

# TODO layer dropou

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

    # TODO dropout

    def __init__(self,
                 dim,
                 filter_size,
                 num_conv_layers,
                 num_heads,
                 dropout_rate=0.0,
                 is_training=True,
                 kernel_regularizer=None,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)

        self.position_encoding = PositionEncoding()

        # Convolution-layer
        self.multiple_separable_conv = ResidualNormed(
            MultipleSeparableConv1D(
                num_layers=num_conv_layers,
                filters=dim,
                width=filter_size,
                padding='same',
                activation='relu',
                kernel_regularizer=kernel_regularizer),
            dropout_rate=dropout_rate,
            is_training=is_training)

        # Self-attention-layer
        self.attention = ResidualNormed(
            MultiHeadAttention(
                num_heads=num_heads,
                input_dim=dim,
                d_k=dim,
                d_v=dim,
                regularizer=kernel_regularizer),
            dropout_rate=dropout_rate,
            is_training=is_training)

        # Feed-forward-layer
        self.feed_forward = ResidualNormed(
            tf.keras.layers.Dense(
                dim,
                activation='relu',
                kernel_regularizer=kernel_regularizer),
            dropout_rate=dropout_rate,
            is_training=is_training)

    def call(self, x):
        # (batch_size, N, input_dim)
        x = self.position_encoding(x)
        # (batch_size, N, input_dim)
        x = self.multiple_separable_conv(x)
        # (batch_size, N, dim)
        x = self.attention([x] * 3)
        # (batch_size, N, dim)
        x = self.feed_forward(x)
        return x

    def get_config(self):
        # TODO 実装する
        return dict()
