import numpy as np
import tensorflow as tf
import unittest

from nose.tools import ok_, eq_

from ..attention import SimilarityMaxtirx
from ..testing_util import create_model

np.random.seed(1234)

class SimilarityMaxtirxTest(unittest.TestCase):
    def test_similarity_matrix(self):
        in_context = tf.keras.layers.Input(shape=(10, 128))
        in_query = tf.keras.layers.Input(shape=(5, 128))
        in_context_mask = tf.keras.layers.Input(shape=(10,))
        in_query_mask = tf.keras.layers.Input(shape=(5,))

        S = SimilarityMaxtirx(
            initializer=tf.keras.initializers.Ones()
        )([in_context, in_query, in_context_mask, in_query_mask])

        model = create_model(
            [in_context, in_query,
             in_context_mask, in_query_mask],
            S)

        context = np.random.randn(2, 10, 128)
        query = np.random.randn(2, 5, 128)
        context_mask = np.array([[1.] * 8 + [0.] * 2] * 2)
        query_mask = np.array([[1.] * 4 + [0.] * 1] * 2)
        prediction = model.predict([context, query, context_mask, query_mask])

        c_ = context[0]
        q_ = query[0]

        ok_(np.isclose(
            sum(np.sum(x) for x in [c_[0], q_[0], c_[0] * q_[0]]),
            prediction[0][0][0]))
        ok_(np.isclose(
            sum(np.sum(x) for x in [c_[0], q_[1], c_[0] * q_[1]]),
            prediction[0][0][1]))
        ok_(np.isclose(
            sum(np.sum(x) for x in [c_[0], q_[2], c_[0] * q_[2]]),
            prediction[0][0][2]))

        ok_(np.isclose(
            sum(np.sum(x) for x in [c_[1], q_[0], c_[1] * q_[0]]),
            prediction[0][1][0]))
        ok_(np.isclose(
            sum(np.sum(x) for x in [c_[1], q_[1], c_[1] * q_[1]]),
            prediction[0][1][1]))
        ok_(np.isclose(
            sum(np.sum(x) for x in [c_[1], q_[2], c_[1] * q_[2]]),
            prediction[0][1][2]))

        ok_((prediction[0][8] < -1e29).all())
        ok_((prediction[0][9] < -1e29).all())

        ok_((prediction[0][:, 4] < -1e29).all())
