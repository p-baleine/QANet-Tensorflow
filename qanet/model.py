import numpy as np
import tensorflow as tf

from .layers.attention import SimilarityMaxtirx
from .layers.attention import ContextQueryAttention, QueryContextAttention
from .layers.core import HighwayNetwork, PositionPrediction, ExpandDims
from .layers.embeddings import WordEmbedding, CharacterEmbedding
from .layers.encoder import Encoder

def create_model(embedding_matrix, hparams):
    """QANet

    引数:
      hparams/default.jsonを参照

      TODO 詳細を記述する

    Input:
      context: (batch_size, N)
      context_unk_label: (batch_size, N)
      context_chars: (batch_size, N, W)
      context_mask: (batch_size, N)
      question: (batch_size, M)
      question_unk_label: (batch_size, M)
      question_chars: (batch_size, M, W)
      question_mask: (batch_size, M)

    Output:
      (batch_size, N)
    """

    # Input

    N = hparams.max_context_length
    M = hparams.max_question_length
    W = hparams.max_word_length

    in_context = tf.keras.layers.Input((N,))
    in_context_unk_label = tf.keras.layers.Input((N,))
    in_context_chars = tf.keras.layers.Input((N, W))
    in_context_mask = tf.keras.layers.Input((N,))
    in_question = tf.keras.layers.Input((M,))
    in_question_unk_label = tf.keras.layers.Input((M,))
    in_question_chars = tf.keras.layers.Input((M, W))
    in_question_mask = tf.keras.layers.Input((M,))

    # Input Embedding Layer.

    word_embedding_layer = WordEmbedding(embedding_matrix)
    char_embedding_layer = CharacterEmbedding(
        vocab_size=hparams.char_vocab_size,
        emb_dim=hparams.char_emb_dim,
        out_dim=hparams.char_dim,
        filter_size=hparams.char_conv_filter_size)

    context_emb = word_embedding_layer((in_context, in_context_unk_label))
    question_emb = word_embedding_layer((in_question, in_question_unk_label))

    context_char_emb = char_embedding_layer(in_context_chars)
    question_char_emb = char_embedding_layer(in_question_chars)

    context = tf.keras.layers.Concatenate(axis=2)([
        context_emb, context_char_emb])
    question = tf.keras.layers.Concatenate(axis=2)([
        question_emb, question_char_emb])

    context = HighwayNetwork(hparams.highway_num_layers)(context)
    question = HighwayNetwork(hparams.highway_num_layers)(question)

    # Embedding Encoder Layer.

    # the input of this layer is a vector of dimension
    # p1 + p2 = 500 for each individual word, which is immediately
    # mapped to d = 128 by a one-dimensional convolution.
    projection_conv = tf.keras.layers.Conv2D(
        filters=hparams.dim,
        kernel_size=(1, hparams.embedding_encoder_filter_size),
        padding='same',
        activation='relu')

    # (batch_size, 1, N, input_dim)
    context = ExpandDims(1)(context)
    # (batch_size, 1, M, input_dim)
    question = ExpandDims(1)(question)
    # (batch_size, 1, N, out_dim)
    context = projection_conv(context)
    # (batch_size, 1, M, out_dim)
    question = projection_conv(question)
    # (batch_size, N, out_dim)
    context = tf.keras.layers.Reshape((N, hparams.dim))(context)
    # (batch_size, M, out_dim)
    question = tf.keras.layers.Reshape((M, hparams.dim))(question)

    # We also share weights of the context and question encoder
    embedding_encoder = Encoder(
        dim=hparams.dim,
        filter_size=hparams.embedding_encoder_filter_size,
        num_conv_layers=hparams.embedding_encoder_num_conv_layers,
        num_heads=hparams.embedding_encoder_num_heads)

    # 今は(paperに従い)embedding_encoderは1ブロックのみ対応
    assert hparams.embedding_encoder_num_blocks == 1

    # (batch_size, N, out_dim)
    context = embedding_encoder(context)
    # (batch_size, M, out_dim)
    question = embedding_encoder(question)

    # Context-Query Attention Layer.

    S = SimilarityMaxtirx()((
        context,
        question,
        in_context_mask,
        in_question_mask))

    S_r = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, 2))(S)
    S_c = tf.keras.layers.Lambda(lambda x: tf.nn.softmax(x, 1))(S)
    A = ContextQueryAttention()((S_r, question))
    B = QueryContextAttention()((S_r, S_c, context))

    # Model Encoder Layer.

    # We share weights between each of the 3 repetitions
    # of the model encoder.
    num_blocks = hparams.model_encoder_num_blocks
    num_gpus = hparams.num_gpus

    model_encoders = []

    for b in range(num_blocks):
        model_encoders.append(Encoder(
            dim=hparams.dim * 4,
            filter_size=hparams.model_encoder_filter_size,
            num_conv_layers=hparams.model_encoder_num_conv_layers,
            num_heads=hparams.model_encoder_num_heads))

    x = tf.keras.layers.Concatenate(axis=2)([
        context,
        A,
        tf.keras.layers.Multiply()([context, A]),
        tf.keras.layers.Multiply()([context, B])])

    M_0 = multiple_encoder_block(model_encoders, x, num_gpus)
    M_1 = multiple_encoder_block(model_encoders, M_0, num_gpus)
    M_2 = multiple_encoder_block(model_encoders, M_1, num_gpus)

    # Output layer.

    p_1 = PositionPrediction()((M_0, M_1, in_context_mask))
    p_2 = PositionPrediction()((M_0, M_2, in_context_mask))

    return tf.keras.models.Model(
        inputs=[
            in_context,
            in_context_unk_label,
            in_context_chars,
            in_context_mask,
            in_question,
            in_question_unk_label,
            in_question_chars,
            in_question_mask,
        ],
        outputs=[p_1, p_2])

def multiple_encoder_block(encoders, x, num_gpus=1):
    for idx, encoder in enumerate(encoders):
        # with tf.device('/gpu:{}'.format(idx % num_gpus)):
        device = 0 if idx < 4 else 1
        with tf.device('/gpu:{}'.format(device)):
            x = encoder(x)
    return x
