import numpy as np
import tensorflow as tf

from .layers.attention import SimilarityMaxtirx
from .layers.attention import ContextQueryAttention, QueryContextAttention
from .layers.core import HighwayNetwork, PositionPrediction, ExpandDims
from .layers.embeddings import WordEmbedding, CharacterEmbedding
from .layers.encoder import Encoder

class QANet(tf.keras.Model):
    def __init__(self, embedding_matrix, hparams, **kwargs):
        super(QANet, self).__init__(**kwargs)

        self.dim = hparams.dim
        self.N = hparams.max_context_length
        self.M = hparams.max_question_length
        self.num_gpus = hparams.num_gpus

        self.word_embedding = WordEmbedding(embedding_matrix)
        self.char_embedding = CharacterEmbedding(
            vocab_size=hparams.char_vocab_size,
            emb_dim=hparams.char_emb_dim,
            out_dim=hparams.char_dim,
            filter_size=hparams.char_conv_filter_size)

        self.context_highway = HighwayNetwork(hparams.highway_num_layers)
        self.question_highway = HighwayNetwork(hparams.highway_num_layers)

        # the input of this layer is a vector of dimension
        # p1 + p2 = 500 for each individual word, which is immediately
        # mapped to d = 128 by a one-dimensional convolution.
        self.projection = tf.keras.layers.Conv2D(
            filters=hparams.dim,
            kernel_size=(1, hparams.embedding_encoder_filter_size),
            padding='same',
            activation='relu')

        self.embedding_encoder = Encoder(
            dim=hparams.dim,
            filter_size=hparams.embedding_encoder_filter_size,
            num_conv_layers=hparams.embedding_encoder_num_conv_layers,
            num_heads=hparams.embedding_encoder_num_heads)

        self.similarity_matrix = SimilarityMaxtirx()
        self.context_query_attention = ContextQueryAttention()
        self.query_context_attention = QueryContextAttention()

        # Model encoders
        self.model_encoders = []

        for block in range(hparams.model_encoder_num_blocks):
            e = Encoder(
                dim=hparams.dim * 4,
                filter_size=hparams.model_encoder_filter_size,
                num_conv_layers=hparams.model_encoder_num_conv_layers,
                num_heads=hparams.model_encoder_num_heads)
            self.model_encoders.append(e)
            # make keras.Model classes to track all the variables in
            # a list of Layer objects.
            setattr(self, 'model_encoder-{}'.format(block), e)

        self.position_prediction1 = PositionPrediction()
        self.porision_prediction2 = PositionPrediction()

    def call(self, inputs, training):
        # TODO trainingで切り分け
        in_context, in_context_unk_label,\
            in_context_chars, in_context_mask,\
            in_question, in_question_unk_label,\
            in_question_chars, in_question_mask = inputs

        # Input Embedding Layer.

        context_emb = self.word_embedding((in_context, in_context_unk_label))
        question_emb = self.word_embedding((in_question, in_question_unk_label))

        context_char_emb = self.char_embedding(in_context_chars)
        question_char_emb = self.char_embedding(in_question_chars)

        context = tf.concat([context_emb, context_char_emb], axis=2)
        question = tf.concat([question_emb, question_char_emb], axis=2)

        context = self.context_highway(context)
        question = self.question_highway(question)

        # Embedding Encoder Layer.

        # (batch_size, 1, N, input_dim)
        context = tf.expand_dims(context, axis=1)
        # (batch_size, 1, M, input_dim)
        question = tf.expand_dims(question, axis=1)
        # (batch_size, 1, N, out_dim)
        context = self.projection(context)
        # (batch_size, 1, M, out_dim)
        question = self.projection(question)
        # (batch_size, N, out_dim)
        context = tf.reshape(context, [-1, self.N, self.dim])
        # (batch_size, M, out_dim)
        question = tf.reshape(question, [-1, self.M, self.dim])
        # (batch_size, N, out_dim)
        context = self.embedding_encoder(context)
        # (batch_size, M, out_dim)
        question = self.embedding_encoder(question)

        # Context-Query Attention Layer.

        S = self.similarity_matrix(
            (context, question, in_context_mask, in_question_mask))
        S_r = tf.nn.softmax(S, 2)
        S_c = tf.nn.softmax(S, 1)
        A = self.context_query_attention((S_r, question))
        B = self.query_context_attention((S_r, S_c, context))

        # Model Encoder Layer.

        x = tf.concat([
            context,
            A,
            context * A,
            context * B
        ], axis=2)

        M_0 = self._multiple_encoder_block(x)
        M_1 = self._multiple_encoder_block(M_0)
        M_2 = self._multiple_encoder_block(M_1)

        # Output layer.

        p_1 = self.position_prediction1((M_0, M_1, in_context_mask))
        p_2 = self.porision_prediction2((M_0, M_2, in_context_mask))

        return [p_1, p_2]

    def _multiple_encoder_block(self, x):
        # TODO Currentry only 2 gpus are assumed.
        for idx, encoder in enumerate(self.model_encoders):
            device = 0 if idx < 3 else 1
            with tf.device('/gpu:{}'.format(device)):
                x = encoder(x)
        return x

def loss_fn(model, inputs, targets, training):
    def compute_loss(labels, outputs):
        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=labels, logits=outputs))

    l1, l2 = targets
    o1, o2 = model(inputs, training=training)

    return compute_loss(l1, o1) + compute_loss(l2, o2)

def accuracy_fn(model, inputs, targets, batch_size, training):
    def compute_accuracy(predictions, labels):
        return tf.reduce_sum(tf.cast(
            tf.equal(predictions, labels), dtype=tf.float32)) / batch_size

    l1, l2 = targets
    l1, l2 = tf.cast(l1, tf.int64), tf.cast(l2, tf.int64)
    o1, o2 = model(inputs, training=training)
    p1 = tf.argmax(o1, axis=1, output_type=tf.int64)
    p2 = tf.argmax(o2, axis=1, output_type=tf.int64)

    return compute_accuracy(p1, l1), compute_accuracy(p2, l2)
