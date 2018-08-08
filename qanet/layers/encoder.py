import tensorflow as tf

from .attention import MultiHeadAttention
from .core import FeedForward, PositionEncoding
from .layer_utils import layer_dropout
from .wrappers import LayerNormed

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
                 dropout_rate=0.0,
                 conv_regularizer=None,
                 attention_regularizer=None,
                 ff_regularizer=None,
                 **kwargs):
        super(Encoder, self).__init__(**kwargs)
        self._layer_dropout_survival_prob = layer_dropout_survival_prob
        self._dropout_rate = dropout_rate

        self.position_encoding = PositionEncoding()

        # Convolution-layer
        self.conv_layers = []

        for idx in range(num_conv_layers):
            layer = LayerNormed(
                tf.keras.layers.SeparableConv1D(
                    filters=dim,
                    kernel_size=filter_size,
                    padding='same',
                    activation='relu',
                    depthwise_regularizer=conv_regularizer,
                    pointwise_regularizer=conv_regularizer,
                    bias_regularizer=conv_regularizer,
                    activity_regularizer=conv_regularizer),
                dropout_rate=dropout_rate,
                regularizer=conv_regularizer)
            self.conv_layers.append(layer)
            setattr(self, 'separable_conv-{}'.format(idx), layer)

        self.attention = LayerNormed(
            MultiHeadAttention(
                num_heads=num_heads,
                input_dim=dim,
                d_k=dim,
                d_v=dim,
                regularizer=attention_regularizer),
            dropout_rate=dropout_rate,
            regularizer=attention_regularizer)

        self.feed_forward = LayerNormed(
            FeedForward(dim, ff_regularizer),
            dropout_rate=dropout_rate,
            regularizer=ff_regularizer)

    def call(self, inputs, training):
        x, mask = inputs

        total_layers = len(self.conv_layers) + 2
        layer_idx = 0

        # (batch_size, N, input_dim)
        x = self.position_encoding(x)
        layer_idx += 1

        for idx, conv in enumerate(self.conv_layers):
            # (batch_size, N, dim)
            y = conv(x, training=training)
            y = layer_dropout(x, y, layer_idx, total_layers,
                              p_L=self._layer_dropout_survival_prob,
                              training=training)
            x = y
            layer_idx += 1

        # (batch_size, N, dim)
        y = self.attention([x] * 3 + [mask], training=training)
        y = layer_dropout(x, y, layer_idx, total_layers,
                          p_L=self._layer_dropout_survival_prob,
                          training=training)
        x = y
        layer_idx += 1

        # (batch_size, N, dim)
        y = self.feed_forward(x, training=training)
        y = layer_dropout(x, y, layer_idx, total_layers,
                          p_L=self._layer_dropout_survival_prob,
                          training=training)

        return y
