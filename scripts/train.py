"""
学習する

動かし方:

  python -m scripts.train \
    --data /path/to/preprocessed_data_dir \
    --hparams /path/to/hparams.json \
    --save_path /path/to/save_dir
"""

import click
import functools
import json
import logging
import numpy as np
import os
import tensorflow as tf

import qanet.model as qanet_model
import qanet.data_utils as data_util

from qanet.data_utils import PaddedInput
from qanet.model_utils import create_iterator, get_training_session_run_hooks
from qanet.model_utils import load_data, load_embedding, load_hparams
from qanet.model_utils import monitored_session
from qanet.preprocess import Preprocessor

logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)

@click.command()
@click.option('--data', type=click.Path(exists=True))
@click.option('--hparams', type=click.Path(exists=True), default=None)
@click.option('--save-path', type=click.Path(exists=True))
def main(data, hparams, save_path):
    processor = Preprocessor.restore(os.path.join(data, 'preprocessor.pickle'))
    hparams = load_hparams(
        save_path, hparams_path=hparams, preprocessor=processor)

    logger.info('Hyper parameters:')
    logger.info(json.dumps(json.loads(hparams.to_json()), indent=2))

    logger.info('Loading data...')

    train_data, dev_data = load_data(data)
    _, _, train_iterator, train_feed_dict = create_iterator(
        train_data, hparams, do_sort=True, repeat_count=hparams.epochs)
    _, _, dev_iterator, dev_feed_dict = create_iterator(
        dev_data, hparams, do_sort=False, repeat_count=-1)
    embedding = load_embedding(data)

    logger.info('Preparing model...')

    train_inputs, train_labels = train_iterator.get_next()
    dev_inputs, dev_labels = dev_iterator.get_next()

    model = qanet_model.QANet(embedding, hparams)
    global_step = tf.train.get_or_create_global_step()

    # learning rate warm-up scheme
    learning_rate = tf.minimum(
        hparams.learning_rate,
        0.001 / tf.log(tf.cast(hparams.warmup_steps - 1, tf.float32))
        * tf.log(tf.cast(global_step, tf.float32) + 1))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_loss = qanet_model.loss_fn(
        model, train_inputs, train_labels, training=True)
    grads = optimizer.compute_gradients(
        train_loss, colocate_gradients_with_ops=True)
    train_op = optimizer.apply_gradients(grads, global_step=global_step)
    train_acc = qanet_model.accuracy_fn(
        model, train_inputs, train_labels, hparams.batch_size, training=True)

    dev_loss = qanet_model.loss_fn(
        model, dev_inputs, dev_labels, training=False)
    dev_acc = qanet_model.accuracy_fn(
        model, dev_inputs, dev_labels, hparams.batch_size, training=False)

    tf.summary.scalar('train_loss', train_loss)
    tf.summary.scalar('dev_loss', dev_loss)
    tf.summary.scalar('train_acc_p1', train_acc[0])
    tf.summary.scalar('train_acc_p2', train_acc[1])
    tf.summary.scalar('dev_acc_p1', dev_acc[0])
    tf.summary.scalar('dev_acc_p2', dev_acc[1])
    tf.summary.scalar('learning_rate', learning_rate)

    merged = tf.summary.merge_all()

    logger.info('Start training...')

    scaffold = tf.train.Scaffold()
    steps_per_epoch = len(train_data) // hparams.batch_size
    hooks = get_training_session_run_hooks(
        save_path, train_loss, dev_loss, scaffold, steps_per_epoch)

    with monitored_session(save_path, scaffold, hooks=hooks) as sess:
        sess.run([train_iterator.initializer, dev_iterator.initializer],
                 feed_dict={**train_feed_dict, **dev_feed_dict})

        while not sess.should_stop():
            sess.run([train_op, merged])

if __name__ == '__main__':
    main()
