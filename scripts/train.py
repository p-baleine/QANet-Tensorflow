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

from qanet.model_utils import create_dataset, get_training_session_run_hooks
from qanet.model_utils import load_embedding, load_hparams
from qanet.model_utils import monitored_session
from qanet.preprocess import Preprocessor

logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)

@click.command()
@click.option('--data', type=click.Path(exists=True))
@click.option('--hparams', type=click.Path(exists=True), default=None)
@click.option('--save-path', type=click.Path(exists=True))
@click.option('--save-steps', type=click.INT, default=600)
def main(data, hparams, save_path, save_steps):
    processor = Preprocessor.restore(os.path.join(data, 'preprocessor.pickle'))
    hparams = load_hparams(
        save_path, hparams_path=hparams, preprocessor=processor)

    logger.info('Hyper parameters:')
    logger.info(json.dumps(json.loads(hparams.to_json()), indent=2))

    logger.info('Load embedding...')

    embedding = load_embedding(data)

    logger.info('Preparing model...')

    batch_size = hparams.batch_size
    train_file = os.path.join(data, 'train.tfrecord')
    dev_file = os.path.join(data, 'dev.tfrecord')

    train_dataset = create_dataset(train_file, batch_size, do_shuffle=True,
                                   repeat_count=hparams.epochs)
    dev_dataset = create_dataset(dev_file, batch_size, do_shuffle=False,
                                 repeat_count=-1)

    train_iterator = train_dataset.make_one_shot_iterator()
    dev_iterator = dev_dataset.make_one_shot_iterator()

    _, train_inputs, train_labels = train_iterator.get_next()
    _, dev_inputs, dev_labels = dev_iterator.get_next()

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
        model, train_inputs, train_labels, batch_size, training=True)

    dev_loss = qanet_model.loss_fn(
        model, dev_inputs, dev_labels, training=False)
    dev_acc = qanet_model.accuracy_fn(
        model, dev_inputs, dev_labels, batch_size, training=False)

    if hparams.ema_decay < 1.0:
        # Apply exponential moving average.
        ema = tf.train.ExponentialMovingAverage(
            hparams.ema_decay, global_step)
        with tf.control_dependencies([apply_gradient_op]):
            train_op = ema.apply(tf.trainable_variables())
    else:
        train_op = apply_gradient_op

    # Session run hooks.

    scaffold = tf.train.Scaffold()

    hooks = [
        tf.train.NanTensorHook(train_loss),
        tf.train.StepCounterHook(
            output_dir=save_path, every_n_steps=10),
        tf.train.LoggingTensorHook(
            {'train_loss': train_loss, 'dev_loss': dev_loss},
            every_n_iter=10),
        tf.train.CheckpointSaverHook(
            save_path,
            save_steps=save_steps,
            scaffold=scaffold),
        tf.train.SummarySaverHook(
            output_dir=save_path,
            save_steps=100,
            scaffold=scaffold),
    ]

    tf.summary.scalar('train_loss', train_loss)
    tf.summary.scalar('dev_loss', dev_loss)
    tf.summary.scalar('train_acc_p1', train_acc[0])
    tf.summary.scalar('train_acc_p2', train_acc[1])
    tf.summary.scalar('dev_acc_p1', dev_acc[0])
    tf.summary.scalar('dev_acc_p2', dev_acc[1])
    tf.summary.scalar('learning_rate', learning_rate)

    merged = tf.summary.merge_all()

    logger.info('Start training...')

    with monitored_session(save_path, scaffold, hooks=hooks) as sess:
        while not sess.should_stop():
            sess.run(train_op)

def get_scheduled_learning_rate(hparams, global_step):
    return tf.minimum(
        hparams.learning_rate,
        0.001 / tf.log(tf.cast(hparams.warmup_steps - 1, tf.float32))
        * tf.log(tf.cast(global_step, tf.float32) + 1))

if __name__ == '__main__':
    main()
