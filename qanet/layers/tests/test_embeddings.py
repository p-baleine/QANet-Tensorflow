import numpy as np
import tensorflow as tf
import unittest

from nose.tools import ok_, eq_

from ..embeddings import WordEmbedding, CharacterEmbedding
from ..testing_util import create_model

np.random.seed(1234)

class TestWordEmbedding(unittest.TestCase):
    def test_embedding(self):
        embedding_matrix = np.random.randn(100, 50)

        in_words = tf.keras.layers.Input(shape=(3,))
        in_unk_label = tf.keras.layers.Input(shape=(3,))

        out_word_emb = WordEmbedding(embedding_matrix)([
            in_words, in_unk_label])

        model = create_model([in_words, in_unk_label], out_word_emb)

        word_emb = model.predict([
            np.array([[1, 2, 3]]),
            np.array([[True, False, False]])])

        # 一つ目はUNK
        ok_(not all(np.isclose(
            word_emb[0][0], embedding_matrix[1])))
        # 二つ目はembedding_matrixと同じ
        ok_(all(np.isclose(
            word_emb[0][1], embedding_matrix[2])))

class TestCharacterEmbegging(unittest.TestCase):
    def test_embedding(self):
        V = 100
        N = 20
        C = 16
        emb_dim = 30
        out_dim = 40
        filter_size = 3

        embedding_matrix = np.random.randn(V, emb_dim)
        kernel_init = np.random.randn(1, filter_size, emb_dim, out_dim)

        inputs = tf.keras.layers.Input(shape=(N, C))

        x = CharacterEmbedding(
            V,
            emb_dim,
            out_dim,
            filter_size,
            embedding_initializer=tf.constant_initializer(embedding_matrix),
            conv_kernel_initializer=tf.constant_initializer(kernel_init),
            conv_bias_initializer=tf.zeros_initializer())(inputs)

        model = create_model(inputs, x)
        chars = np.random.randint(V, size=(2, N, C))
        char_emb = model.predict(chars)

        # TODO 出力の値のテスト
        eq_(list(char_emb.shape), [2, N, out_dim])
