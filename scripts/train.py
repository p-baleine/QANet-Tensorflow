"""
Training.

Usage:

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
        train_data, hparams, do_shuffle=True, repeat_count=hparams.epochs)
    _, _, dev_iterator, dev_feed_dict = create_iterator(
        dev_data, hparams, do_shuffle=False, repeat_count=-1)
    embedding = load_embedding(data)

    logger.info('Preparing model...')

    train_inputs, train_labels = train_iterator.get_next()
    dev_inputs, dev_labels = dev_iterator.get_next()

    model = qanet_model.QANet(embedding, hparams)
    global_step = tf.train.get_or_create_global_step()

    # learning rate warm-up scheme
    learning_rate = get_scheduled_learning_rate(hparams, global_step)
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate,
        beta1=0.8,
        beta2=0.999,
        epsilon=1e-7)
    train_loss = qanet_model.loss_fn(
        model, train_inputs, train_labels, training=True)

    if hparams.l2_regularizer_scale is not None:
        # Apply l2 regularization.
        variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        l2_loss = tf.contrib.layers.apply_regularization(
            model.regularizer, variables)
        train_loss += l2_loss

    grads, tvars = zip(*optimizer.compute_gradients(
        train_loss, colocate_gradients_with_ops=True))
    clipped_grads, _ = tf.clip_by_global_norm(grads, hparams.max_grad_norm)
    apply_gradient_op = optimizer.apply_gradients(
        zip(clipped_grads, tvars), global_step=global_step)
    train_acc = qanet_model.accuracy_fn(
        model, train_inputs, train_labels, hparams.batch_size, training=True)

    dev_loss = qanet_model.loss_fn(
        model, dev_inputs, dev_labels, training=False)
    dev_acc = qanet_model.accuracy_fn(
        model, dev_inputs, dev_labels, hparams.batch_size, training=False)

    if hparams.ema_decay < 1.0:
        # Apply exponential moving average.
        ema = tf.train.ExponentialMovingAverage(
            hparams.ema_decay, global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = ema.apply(tf.trainable_variables())
    else:
        train_op = apply_gradient_op

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
        save_path, train_loss, dev_loss, scaffold, steps_per_epoch,
        train_iterator, train_feed_dict,
        dev_iterator, dev_feed_dict)

    with monitored_session(save_path, scaffold, hooks=hooks) as sess:
        while not sess.should_stop():
            sess.run([train_op, merged])

def get_scheduled_learning_rate(hparams, global_step):
    return tf.minimum(
        hparams.learning_rate,
        0.001 / tf.log(tf.cast(hparams.warmup_steps - 1, tf.float32))
        * tf.log(tf.cast(global_step, tf.float32) + 1))

if __name__ == '__main__':
    main()
