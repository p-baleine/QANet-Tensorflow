import contextlib
import json
import logging
import numpy as np
import os
import pickle
import tensorflow as tf

from .preprocess import TransformedOutput

logger = logging.getLogger(__name__)

def load_hparams(save_path, preprocessor=None, hparams_path=None):
    """Return the loaded hyper parameters.

    If there is a file under `save_path` then return this.
    If there isn's a file under `save_path` then load hyper
    parameters from `hparamth_path` and also save them
    in `save_path`.
    """

    saved_path = os.path.join(save_path, 'hparams.json')

    if os.path.exists(saved_path):
        logger.info('Saved hparams found at {}, '
                    'use this instead of {}'.format(saved_path, hparams_path))

        with open(saved_path) as f:
            return tf.contrib.training.HParams(**json.load(f))

    with open(saved_path, 'wt') as fw, open(hparams_path, 'rt') as fr:
        hparams = tf.contrib.training.HParams(**json.load(fr))
        hparams.add_hparam('char_vocab_size', len(preprocessor.char_dict))
        fw.write(hparams.to_json())

    return hparams

def load_embedding(data_path):
    return np.load(os.path.join(data_path, 'vectors.npy'))

def get_answer_spane(start_preds, end_preds):
    # FIXME If probabilities are needed, then we have to do softmax.
    best_score = 0.0
    best_start_idx = 0
    best_span = (0, 1)

    for i in range(len(start_preds)):
        start_score = np.exp(start_preds[best_start_idx])

        if start_score < np.exp(start_preds[i]):
            start_score = np.exp(start_preds[i])
            best_start_idx = i

        end_score = np.exp(end_preds[i])

        if start_score * end_score > best_score:
            best_span = (best_start_idx, i)
            best_score = start_score * end_score

    return best_span[0], best_span[1], best_score

def parse_tfrecord(record):
    context, feature_lists = tf.parse_single_sequence_example(
        serialized=record,
        context_features={
            'id': tf.FixedLenFeature([], tf.string),
        },
        sequence_features={
            'context': tf.FixedLenSequenceFeature([], tf.int64),
            'context_unk_label': tf.FixedLenSequenceFeature([], tf.int64),
            'context_chars': tf.FixedLenSequenceFeature([16], tf.int64),
            'question': tf.FixedLenSequenceFeature([], tf.int64),
            'question_unk_label': tf.FixedLenSequenceFeature([], tf.int64),
            'question_chars': tf.FixedLenSequenceFeature([16], tf.int64),
            'answer_start': tf.FixedLenSequenceFeature([], tf.int64),
            'answer_end': tf.FixedLenSequenceFeature([], tf.int64),
        })
    return (context,
            (feature_lists['context'],
             feature_lists['context_unk_label'],
             feature_lists['context_chars'],
             feature_lists['question'],
             feature_lists['question_unk_label'],
             feature_lists['question_chars'],
            ),
            (feature_lists['answer_start'],
             feature_lists['answer_end']))

def create_dataset(tfrecord_path, batch_size,
                   do_shuffle=False, repeat_count=None):
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    dataset = dataset.map(parse_tfrecord)

    if do_shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    if repeat_count is not None:
        dataset = dataset.repeat(repeat_count)

    dataset = dataset.batch(batch_size)

    return dataset

@contextlib.contextmanager
def monitored_session(save_path, scaffold, hooks=[]):
    session_creator = tf.train.ChiefSessionCreator(
        scaffold=scaffold,
        checkpoint_dir=save_path,
        config=tf.ConfigProto(allow_soft_placement=True))

    with tf.train.MonitoredSession(
            session_creator=session_creator,
            hooks=hooks) as sess:
        yield sess
