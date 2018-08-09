import contextlib
import json
import logging
import numpy as np
import os
import pickle
import tensorflow as tf

from .preprocess import TransformedOutput
from .data_utils import create_transposed_data

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

def load_data(data_path):
    with open(os.path.join(data_path, 'train.json')) as f:
        train_data = json.load(f)
    with open(os.path.join(data_path, 'dev.json')) as f:
        dev_data = json.load(f)
    return (
        [TransformedOutput.from_raw_array(d) for d in train_data],
        [TransformedOutput.from_raw_array(d) for d in dev_data])

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

def create_iterator(data, hparams, do_shuffle, repeat_count=None,
                    remove_longer_data=True):
    if remove_longer_data:
        max_context_length = hparams.max_context_length
        max_question_length = hparams.max_question_length
    else:
        max_context_length = 1e4
        max_question_length = 1e4

    id, title, inputs, labels = create_transposed_data(
        data,
        max_context_length=max_context_length,
        max_question_length=max_question_length,
        max_word_length=hparams.max_word_length)

    # Create iterator.
    input_placeholders = inputs._replace(**dict(
        (k, tf.placeholder(d.dtype, d.shape))
        for k, d in zip(inputs._fields, inputs)))
    label_placeholders = tuple(
        tf.placeholder(d.dtype, d.shape) for d in labels)

    dataset = tf.data.Dataset.from_tensor_slices(
        (input_placeholders, label_placeholders))

    if do_shuffle:
        dataset = dataset.shuffle(buffer_size=10000)

    dataset = dataset.batch(hparams.batch_size)

    if repeat_count is not None:
        dataset = dataset.repeat(repeat_count)

    iterator = dataset.make_initializable_iterator()

    # Create feed_dict.
    feed_dict = dict((p, d) for p, d in zip(
        input_placeholders + label_placeholders,
        inputs + labels))

    return id, title, iterator, feed_dict

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

def get_training_session_run_hooks(
        save_path,
        train_loss,
        dev_loss,
        scaffold,
        steps_per_epoch,
        train_iterator,
        train_feed_dict,
        dev_iterator,
        dev_feed_dict,
        log_steps=10,
        save_steps=600,
        summary_steps=100):
    train_iterator_init_hook = DatasetInitializerHook(
        train_iterator, train_feed_dict)
    dev_iterator_init_hook = DatasetInitializerHook(
        dev_iterator, dev_feed_dict)
    nan_hook = tf.train.NanTensorHook(train_loss)
    checkpoint_hook = tf.train.CheckpointSaverHook(
        save_path,
        save_steps=save_steps,
        scaffold=scaffold)
    counter_hook = tf.train.StepCounterHook(
        output_dir=save_path,
        every_n_steps=log_steps)
    train_logging_hook = tf.train.LoggingTensorHook(
        {'train_loss': train_loss},
        every_n_iter=log_steps)
    all_logging_hook = tf.train.LoggingTensorHook(
        {'train_loss': train_loss, 'dev_loss': dev_loss},
        every_n_iter=summary_steps)
    summary_hook = tf.train.SummarySaverHook(
        scaffold=scaffold,
        output_dir=save_path,
        save_steps=summary_steps)
    log_epoch_hook = LogEpochHook(steps_per_epoch)

    return [
        train_iterator_init_hook,
        dev_iterator_init_hook,
        nan_hook,
        checkpoint_hook,
        counter_hook,
        train_logging_hook,
        all_logging_hook,
        summary_hook,
        log_epoch_hook]

class LogEpochHook(tf.train.SessionRunHook):
    """Hook that just output epoch count."""

    def __init__(self, steps_per_epoch):
        self._steps_per_epoch = steps_per_epoch
        self._timer = tf.train.SecondOrStepTimer(
            every_steps=1) # dummy parameter

    def begin(self):
        self._global_step_tensor = tf.train.get_global_step()

        if self._global_step_tensor is None:
            raise RuntimeError(
                'Global step should be created to use LogEpochHook')

    def before_run(self, run_context):
        return tf.train.SessionRunArgs(self._global_step_tensor)

    def after_run(self, run_context, run_values):
        global_step = run_values.results

        if global_step != 0 and global_step % self._steps_per_epoch == 0:
            elapsed_time, _ = self._timer.update_last_triggered_step(global_step)
            logger.info('Epoch {} ({} sec)'.format(
                global_step // self._steps_per_epoch,
                elapsed_time))

class DatasetInitializerHook(tf.train.SessionRunHook):
    """Hook that initialize an iterator.

    See: https://github.com/tensorflow/tensorflow/issues/12859#issuecomment-348251009
    """

    def __init__(self, iterator, feed_dict):
        self._iterator = iterator
        self._feed_dict = feed_dict

    def begin(self):
        self._initializer = self._iterator.initializer

    def after_create_session(self, session, coord):
        del coord
        session.run(self._initializer, self._feed_dict)
