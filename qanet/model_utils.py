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

def load_hparams(save_path, hparams_path=None):
    """ハイパーパラメータを読みこんで返す
    save_path配下にファイルが存在する場合、これを読み出す
    save_path配下にファイルが存在しない場合、
    hparamth_pathから読み出して、且つsave_pathに保存しておく
    """

    saved_path = os.path.join(save_path, 'hparams.json')

    if os.path.exists(saved_path):
        logger.info('Saved hparams found at {}, '
                    'use this instead of {}'.format(saved_path, hparams_path))

        with open(saved_path) as f:
            return tf.contrib.training.HParams(**json.load(f))

    with open(saved_path, 'wt') as fw, open(hparams_path, 'rt') as fr:
        hparams = tf.contrib.training.HParams(**json.load(fr))
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
    # FIXME 今はlogitsをそのままscoreとしているため、scoreが確率ではない
    # 確率が欲しい場合はsoftmaxする必要がある
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

def create_iterator(data, hparams, do_sort, repeat=True):
    id, title, inputs, labels = create_transposed_data(
        data,
        do_sort=do_sort,
        max_context_length=hparams.max_context_length,
        max_question_length=hparams.max_question_length,
        max_word_length=hparams.max_word_length)

    dataset = tf.data.Dataset.from_tensor_slices((inputs, labels))
    dataset = dataset.batch(hparams.batch_size)

    if repeat:
        dataset = dataset.repeat(hparams.epochs)

    return id, title, dataset.make_one_shot_iterator()

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
        log_steps=10,
        save_steps=100,
        summary_steps=1056//32):
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
