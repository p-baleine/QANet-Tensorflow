import tensorflow as tf

class WordEmbedding(tf.keras.layers.Layer):
    """単語の埋め込み

    引数:
      embedding_matrix: 学習済の行列

    Input:
      words: (batch_size, N)
      word_unk_label: (batch_size, N)

    Output:
      (batch_size, N, dim)
          dimはembedding_matrixのshape[1]
    """

    def __init__(self,
                 embeddind_matrix,
                 unk_initializer=tf.contrib.layers.xavier_initializer(),
                 unk_regularizer=None,
                 **kwargs):
        self._embedding_matrix = embeddind_matrix
        self._V = self._embedding_matrix.shape[0]
        self._dim = self._embedding_matrix.shape[1]
        self._unk_initializer = unk_initializer
        self._unk_regularizer = unk_regularizer
        super(WordEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        words_shape, word_unk_label_shape = input_shape

        with tf.device('/cpu:0'):
            self._W = self.add_variable(
                'embedding',
                [self._V, self._dim],
                initializer=tf.constant_initializer(self._embedding_matrix),
                trainable=False)

        # All the out-of-vocabulary words are mapped to an <UNK> token,
        # whose embedding is trainable with random initialization.
        self._W_unk = self.add_variable(
            'unk_embedding',
            [1, self._dim],
            initializer=self._unk_initializer,
            regularizer=self._unk_regularizer)

        super(WordEmbedding, self).build(input_shape)

    def call(self, x):
        words, word_unk_label = x
        # なぜかfloatで渡ってくる…
        words = tf.cast(words, tf.int32)
        word_unk_label = tf.cast(word_unk_label, tf.bool)

        # (batch_size, N, dim)
        with tf.device('/cpu:0'):
            return tf.where(
                tf.tile(tf.expand_dims(word_unk_label, -1), [1, 1, self._dim]),
                tf.nn.embedding_lookup(
                    self._W_unk, tf.zeros_like(word_unk_label, dtype=tf.int32)),
                tf.nn.embedding_lookup(self._W, words))

    def compute_output_shape(self, input_shape):
        word_shape, _ = input_shape
        return tf.TensorShape([word_shape[0], word_shape[1], self._dim])

class CharacterEmbedding(tf.keras.layers.Layer):
    def __init__(self,
                 vocab_size,
                 emb_dim,
                 out_dim,
                 filter_size,
                 embedding_initializer=tf.contrib.layers.xavier_initializer(),
                 conv_kernel_initializer=tf.contrib.layers.xavier_initializer(),
                 conv_bias_initializer=tf.zeros_initializer(),
                 regularizer=None,
                 **kwargs):
        """文字の埋め込み

        引数:
          vocab_size: 文字の辞書のサイズ
          emb_dim: embeddingの次元
          out_dim: 出力の次元
          filter_size: 畳み込みのフィルターサイズ

        Input: (batch_size, N, W)

        Output: (batch_size, N, out_dim)
        """
        self._vocab_size = vocab_size
        self._emb_dim = emb_dim
        self._out_dim = out_dim
        self._filter_size = filter_size
        self._embedding_initializer = embedding_initializer
        self._conv_kernel_initializer = conv_kernel_initializer
        self._conv_bias_initializer = conv_bias_initializer
        self._regularizer = regularizer
        super(CharacterEmbedding, self).__init__(**kwargs)

    def build(self, input_shape):
        with tf.device('/cpu:0'):
            self._embedding = self.add_variable(
                'embedding',
                [self._vocab_size, self._emb_dim],
                initializer=self._embedding_initializer,
                regularizer=self._regularizer)
        self._kernel = self.add_variable(
            'kernel',
            [self._filter_size, self._emb_dim, self._out_dim],
            initializer=self._conv_kernel_initializer,
            regularizer=self._regularizer)
        self._bias = self.add_variable(
            'bias',
            [1, 1, self._out_dim],
            initializer=self._conv_bias_initializer,
            regularizer=self._regularizer)
        super(CharacterEmbedding, self).build(input_shape)

    def call(self, x):
        x = tf.cast(x, tf.int32)
        N, W = tf.shape(x)[1], tf.shape(x)[2]

        # from BiDAF
        # Characters are embedded into vectors, which can be
        # considered as 1D inputs to the CNN, and whose size is
        # the input channel size of the CNN.
        # The outputs of the CNN are max-pooled over the entire
        # width to obtain a fixed-size vector for each word.

        # (batch_size, N, W, p2)
        with tf.device('/cpu:0'):
            x_ = tf.nn.embedding_lookup(self._embedding, x)
        # (batch_size * N, W, p2)
        x_ = tf.reshape(x_, [-1, W, self._emb_dim])
        # (batch_size * N, W - filter_size + 1, p2)
        x_ = tf.nn.conv1d(x_, self._kernel, 1, 'VALID') + self._bias
        # (batch_size, N, W - filter_size + 1, p2)
        x_ = tf.reshape(x_, [-1, N, W - self._filter_size + 1, self._out_dim])
        # (batch_size, N, p2)
        return tf.reduce_max(tf.nn.relu(x_), 2)

    def compute_output_shape(self, input_shape):
        return tf.TensorShape(
            [input_shape[0], input_shape[1], self._out_dim])
