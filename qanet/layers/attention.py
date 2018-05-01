import tensorflow as tf

from .layer_utils import exp_mask

class MultiHeadAttention(tf.keras.layers.Layer):
    """multihead-attention

    引数:
      num_heads: headの数
      input_dim: 入力の次元
      d_k: queryとkeyに対する重みの次元
      d_v: valueに対する重みの次元

    Input:
      q: (batch_size, query_length, input_dim)
      k: (batch_size, query_length, input_dim)
      v: (batch_size, query_length, input_dim)

    Output: (batch_size, query_length, input_dim)
    """

    def __init__(self,
                 num_heads,
                 input_dim,
                 d_k,
                 d_v,
                 q_initializer=tf.contrib.layers.xavier_initializer(),
                 k_initializer=tf.contrib.layers.xavier_initializer(),
                 v_initializer=tf.contrib.layers.xavier_initializer(),
                 o_initializer=tf.contrib.layers.xavier_initializer(),
                 **kwargs):
        self._num_heads = num_heads
        self._input_dim = input_dim
        self._d_k = d_k
        self._d_v = d_v
        self._q_initializer = q_initializer
        self._k_initializer = k_initializer
        self._v_initializer = v_initializer
        self._o_initializer = o_initializer
        super(MultiHeadAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self._W_Q = self.add_variable(
            'W_Q',
            [self._input_dim, self._num_heads * self._d_k],
            initializer=self._q_initializer)
        self._W_K = self.add_variable(
            'W_K',
            [self._input_dim, self._num_heads * self._d_k],
            initializer=self._k_initializer)
        self._W_V = self.add_variable(
            'W_V',
            [self._input_dim, self._num_heads * self._d_v],
            initializer=self._v_initializer)
        self._W_O = self.add_variable(
            'W_O',
            [self._num_heads * self._d_v, self._input_dim],
            initializer=self._o_initializer)

        return super(MultiHeadAttention, self).build(input_shape)

    def call(self, x):
        # (batch_size, length, input_dim)
        q, k, v = x

        # (batch_size, length, num_heads * d_k)
        q_W_Q = tf.tensordot(q, self._W_Q, [[2], [0]])
        k_W_K = tf.tensordot(k, self._W_K, [[2], [0]])
        # (batch_size, length, num_heads * d_v)
        v_W_V = tf.tensordot(v, self._W_V, [[2], [0]])
        # (batch_size, num_heads, length, d_k)
        q_W_Q = split_heads(q_W_Q, self._num_heads)
        k_W_K = split_heads(k_W_K, self._num_heads)
        # (batch_size, num_heads, length, d_v)
        v_W_V = split_heads(v_W_V, self._num_heads)

        # (batch_size, num_heads, length, d_v)
        x = dot_product_attention(q_W_Q, k_W_K, v_W_V)
        # (batch_size, length, num_heads * d_v)
        x = combine_heads(x)
        # (batch_size, length, input_dim)
        return tf.tensordot(x, self._W_O, [[2], [0]])

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class SimilarityMaxtirx(tf.keras.layers.Layer):
    """contexとqueryのconfusion matrixを計算する

    Input:
      context: (batch_size, N, dim)
      query: (batch_size, M, dim)
      context_mask: (batch_size, N)
      query_mask: (batch_size, M)

    Output: (batch_size, N, M)
    """

    def __init__(self,
                 initializer=tf.contrib.layers.xavier_initializer(),
                 **kwargs):
        self._initializer = initializer
        super(SimilarityMaxtirx, self).__init__(**kwargs)

    def build(self, input_shape):
        c_shape, _, _, _ = input_shape

        self._W = self.add_variable(
            'weight',
            [c_shape[-1] * 3, 1],
            initializer=self._initializer)

        super(SimilarityMaxtirx, self).build(input_shape)

    def call(self, x):
        c, q, c_mask, q_mask = x
        N, M, d = c.shape[1], q.shape[1], c.shape[-1]

        # (batch_size, N, M, d)
        c = tf.tile(tf.expand_dims(c, 2), [1, 1, M, 1])
        q = tf.tile(tf.expand_dims(q, 1), [1, N, 1, 1])
        # (batch_size, N * M, d)
        c = tf.reshape(c, [-1, N * M, d])
        q = tf.reshape(q, [-1, N * M, d])
        c_q = c * q
        # (batch_size, N * M, d * 3)
        S = tf.concat([c, q, c_q], 2)
        # (batch_size, N, M)
        logits = tf.reshape(tf.tensordot(S, self._W, [[2], [0]]), [-1, N, M])

        # logitsと同じ形のmaskを作る
        # (batch_size, N, M)
        c_mask = tf.cast(tf.tile(tf.expand_dims(c_mask, 2), [1, 1, M]), tf.bool)
        q_mask = tf.cast(tf.tile(tf.expand_dims(q_mask, 1), [1, N, 1]), tf.bool)

        return exp_mask(logits, c_mask & q_mask)

    def compute_output_shape(self, input_shape):
        c_shape, q_shape, _, _ = input_shape
        return tf.TensorShape([
            c_shape[0], c_shape[1], q_shape[1]])

class ContextQueryAttention(tf.keras.layers.Lambda):
    """context-to-query attention

    Input:
      S_r: (batch_size, N, M)
          similarity-matrixを行方向にsoftmaxしたもの
      query: (batch_size, M, dim)

    Output: (batch_size, N, dim)
    """

    def __init__(self, **kwargs):
        def fn(x):
            S_r, q = x
            return tf.matmul(S_r, q)

        super(ContextQueryAttention, self).__init__(
            function=fn, **kwargs)

    def compute_output_shape(self, input_shape):
        d = input_shape[1][-1]
        N = input_shape[0][1]
        return tf.TensorShape([input_shape[0][0], N, d])

class QueryContextAttention(tf.keras.layers.Lambda):
    """query-to-context attention

    Input:
      S_r: (batch_size, N, M)
          similarity-matrixを行方向にsoftmaxしたもの
      S_c: (batch_size, N, M)
          similarity-matrixを列方向にsoftmaxしたもの
      context: (batch_size, N, dim)

    Output: (batch_size, N, dim)
    """

    def __init__(self, **kwargs):
        def fn(x):
            S_r, S_c, c = x
            return tf.matmul(tf.matmul(S_r, S_c, transpose_b=True), c)

        super(QueryContextAttention, self).__init__(
            function=fn, **kwargs)

    def compute_output_shape(self, input_shape):
        d = input_shape[2][-1]
        N = input_shape[0][1]
        return tf.TensorShape([input_shape[0][0], N, d])

def dot_product_attention(Q, K, V):
    d_k = tf.cast(K.shape[-1], tf.float32)
    return tf.matmul(tf.nn.softmax(
        tf.matmul(Q, K, transpose_b=True) / tf.sqrt(d_k)), V)

def split_heads(x, num_heads):
    """xの最後の次元をnum_headsに分ける

    引数:
      x: (batch_size, length, num_heads * dim)
    戻り値: (batch_size, num_heads, length, dim)
    """
    shape = x.shape.as_list()
    # (batch_size, length, num_heads, dim)
    splitted = tf.reshape(
        x, [-1] + shape[1:-1] + [num_heads, shape[-1] // num_heads])
    # (batch_size, num_heads, length, dim)
    return tf.transpose(splitted, [0, 2, 1, 3])

def combine_heads(x):
    """xの最後の次元をnum_heads * d_vに戻す

    引数:
      x: (batch_size, num_heads, length, d_v)
    戻り値: (batch_size, length, num_heads * d_v)
    """
    x = tf.transpose(x, [0, 2, 1, 3])
    shape = x.shape.as_list()
    return tf.reshape(x, [-1] + shape[1:-2] + [shape[-2] * shape[-1]])
