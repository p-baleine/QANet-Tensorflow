import numpy as np
import tensorflow as tf

from .layers.attention import SimilarityMaxtirx
from .layers.core import HighwayNetwork, PositionPrediction
from .layers.embeddings import WordEmbedding, CharacterEmbedding
from .layers.encoder import Encoder
from .layers.layer_utils import exp_mask

class QANet(tf.keras.Model):
    def __init__(self, embedding_matrix, hparams, **kwargs):
        super(QANet, self).__init__(**kwargs)

        self.dim = hparams.dim
        self.W = hparams.max_word_length
        self.word_dropout_rate = hparams.word_dropout_rate
        self.char_dropout_rate = hparams.char_dropout_rate
        self.dropout_rate = hparams.dropout_rate
        self.num_gpus = hparams.num_gpus

        self._regularizer = tf.contrib.layers.l2_regularizer(
            scale=hparams.l2_regularizer_scale)

        self.word_embedding = WordEmbedding(
            embedding_matrix,
            unk_regularizer=self._regularizer)
        self.char_embedding = CharacterEmbedding(
            vocab_size=hparams.char_vocab_size,
            emb_dim=hparams.char_emb_dim,
            out_dim=hparams.char_dim,
            filter_size=hparams.char_conv_filter_size,
            regularizer=self._regularizer)

        self.embedding_encoder_projection = tf.keras.layers.Conv1D(
            filters=hparams.dim,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_regularizer=self._regularizer,
            bias_regularizer=self._regularizer)

        self.highway_network = HighwayNetwork(
            hparams.highway_num_layers,
            regularizer=self._regularizer,
            dropout_rate=hparams.dropout_rate)

        self.embedding_encoder = Encoder(
            dim=hparams.dim,
            filter_size=hparams.embedding_encoder_filter_size,
            num_conv_layers=hparams.embedding_encoder_num_conv_layers,
            num_heads=hparams.embedding_encoder_num_heads,
            layer_dropout_survival_prob=hparams.layer_dropout_survival_prob,
            dropout_rate=hparams.dropout_rate,
            conv_regularizer=self._regularizer,
            attention_regularizer=self._regularizer,
            ff_regularizer=self._regularizer)

        self.similarity_matrix = SimilarityMaxtirx(regularizer=self._regularizer)

        self.model_encoder_projection = tf.keras.layers.Conv1D(
            filters=hparams.dim,
            kernel_size=1,
            padding='same',
            use_bias=False,
            kernel_regularizer=self._regularizer,
            bias_regularizer=self._regularizer)

        # Model encoders
        self.model_encoders = []

        for block in range(hparams.model_encoder_num_blocks):
            e = Encoder(
                dim=hparams.dim,
                filter_size=hparams.model_encoder_filter_size,
                num_conv_layers=hparams.model_encoder_num_conv_layers,
                num_heads=hparams.model_encoder_num_heads,
                layer_dropout_survival_prob=hparams.layer_dropout_survival_prob,
                dropout_rate=hparams.dropout_rate,
                conv_regularizer=self._regularizer,
                attention_regularizer=self._regularizer,
                ff_regularizer=self._regularizer)
            self.model_encoders.append(e)
            # make keras.Model classes to track all the variables in
            # a list of Layer objects.
            setattr(self, 'model_encoder-{}'.format(block), e)

        self.position_prediction1 = PositionPrediction(
            regularizer=self._regularizer)
        self.porision_prediction2 = PositionPrediction(
            regularizer=self._regularizer)

    def call(self, inputs, training):
        N, M, in_context, in_context_unk_label,\
            in_context_chars, in_context_mask,\
            in_question, in_question_unk_label,\
            in_question_chars, in_question_mask = self.truncate_inputs(
                inputs, self.W)

        # Input Embedding Layer.

        context_emb = self.word_embedding((in_context, in_context_unk_label))
        question_emb = self.word_embedding((in_question, in_question_unk_label))

        if training:
            keep_prob = 1.0 - self.word_dropout_rate
            context_emb = tf.nn.dropout(context_emb, keep_prob)
            question_emb = tf.nn.dropout(question_emb, keep_prob)

        context_char_emb = self.char_embedding(in_context_chars)
        question_char_emb = self.char_embedding(in_question_chars)

        if training:
            keep_prob = 1.0 - self.char_dropout_rate
            context_char_emb = tf.nn.dropout(context_char_emb, keep_prob)
            question_char_emb = tf.nn.dropout(question_char_emb, keep_prob)

        context = tf.concat([context_emb, context_char_emb], axis=2)
        question = tf.concat([question_emb, question_char_emb], axis=2)

        # (batch_size, N, out_dim)
        context = self.embedding_encoder_projection(context)
        # (batch_size, M, out_dim)
        question = self.embedding_encoder_projection(question)
        context = self.highway_network(context, training=training)
        question = self.highway_network(question, training=training)

        # Embedding Encoder Layer.

        context = self.embedding_encoder(
            (context, in_context_mask), training=training)
        # (batch_size, M, out_dim)
        question = self.embedding_encoder(
            (question, in_question_mask), training=training)

        # Context-Query Attention Layer.

        # (batch_size, N, M)
        S = self.similarity_matrix(
            (context, question, in_context_mask, in_question_mask))
        S_r = tf.nn.softmax(exp_mask(S, tf.expand_dims(in_question_mask, 1)), 2)
        S_c = tf.nn.softmax(exp_mask(S, tf.expand_dims(in_context_mask, 2)), 1)

        # (batch_size, N, dim)
        A = tf.matmul(S_r, question)
        # (batch_size, N, dim)
        B = tf.matmul(tf.matmul(S_r, tf.transpose(S_c, [0, 2, 1])), context)

        # Model Encoder Layer.

        # (batch_size, N, dim * 4)
        x = tf.concat([
            context,
            A,
            context * A,
            context * B
        ], axis=2)

        # (batch_size, N, dim)
        x = self.model_encoder_projection(x)

        M_0 = self._multiple_encoder_block(
            x, in_context_mask, training=training)
        M_1 = self._multiple_encoder_block(
            M_0, in_context_mask, training=training)
        M_2 = self._multiple_encoder_block(
            M_1, in_context_mask, training=training)

        # Output layer.

        p_1 = self.position_prediction1((M_0, M_1, in_context_mask))
        p_2 = self.porision_prediction2((M_0, M_2, in_context_mask))

        return [p_1, p_2]

    @property
    def regularizer(self):
        return self._regularizer

    def _multiple_encoder_block(self, x, mask, training):
        # TODO Currentry only 2 gpus are assumed.
        for idx, encoder in enumerate(self.model_encoders):
            device = 0 if idx < 3 else 1
            with tf.device('/gpu:{}'.format(device)):
                x = encoder((x, mask), training=training)
        return x

    @staticmethod
    def truncate_inputs(inputs, W):
        in_context, in_context_unk_label,\
            in_context_chars, in_context_mask,\
            in_question, in_question_unk_label,\
            in_question_chars, in_question_mask = inputs

        batch_size = tf.shape(in_context)[0]
        N = tf.reduce_max(tf.reduce_sum(
            tf.cast(in_context_mask, tf.int32), axis=1))
        M = tf.reduce_max(tf.reduce_sum(
            tf.cast(in_question_mask, tf.int32), axis=1))
        in_context = tf.slice(in_context, [0, 0], [batch_size, N])
        in_context_unk_label = tf.slice(
            in_context_unk_label, [0, 0], [batch_size, N])
        in_context_chars = tf.slice(
            in_context_chars, [0, 0, 0], [batch_size, N, W])
        in_context_mask = tf.slice(
            in_context_mask, [0, 0], [batch_size, N])
        in_question = tf.slice(in_question, [0, 0], [batch_size, M])
        in_question_unk_label = tf.slice(
            in_question_unk_label, [0, 0], [batch_size, M])
        in_question_chars = tf.slice(
            in_question_chars, [0, 0, 0], [batch_size, M, W])
        in_question_mask = tf.slice(
            in_question_mask, [0, 0], [batch_size, M])

        return N, M, in_context, in_context_unk_label,\
            in_context_chars, in_context_mask,\
            in_question, in_question_unk_label,\
            in_question_chars, in_question_mask

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
