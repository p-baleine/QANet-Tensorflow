"""
学習する

動かし方:

  python -m scripts.train \
    --data /path/to/preprocessed_data_dir \
    --hparams /path/to/hparams.json \
    --save_path /path/to/save_dir
"""

import click
import json
import logging
import numpy as np
import tensorflow as tf

import qanet.model as qanet_model

from qanet.model_utils import create_iterator, get_training_session_run_hooks
from qanet.model_utils import load_data, load_embedding, load_hparams
from qanet.model_utils import monitored_session

logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)

@click.command()
@click.option('--data', type=click.Path(exists=True))
@click.option('--hparams', type=click.Path(exists=True), default=None)
@click.option('--save-path', type=click.Path(exists=True))
def main(data, hparams, save_path):
    hparams = load_hparams(save_path, hparams)

    logger.info('Hyper parameters:')
    logger.info(json.dumps(json.loads(hparams.to_json()), indent=2))

    logger.info('Loading data...')

    train_data, dev_data = load_data(data)
    embedding = load_embedding(data)

    _, _, train_iterator = create_iterator(train_data, hparams, True)
    _, _, dev_iterator = create_iterator(dev_data, hparams, False)

    logger.info('Preparing model...')

    train_inputs, train_labels = train_iterator.get_next()
    dev_inputs, dev_labels = dev_iterator.get_next()

    model = qanet_model.QANet(embedding, hparams)
    optimizer = tf.train.RMSPropOptimizer(
        learning_rate=hparams.learning_rate)
    train_loss = qanet_model.loss_fn(
        model, train_inputs, train_labels, training=True)
    grads = optimizer.compute_gradients(
        train_loss, colocate_gradients_with_ops=True)
    train_op = optimizer.apply_gradients(
        grads, global_step=tf.train.get_or_create_global_step())
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
    merged = tf.summary.merge_all()

    logger.info('Start training...')

    scaffold = tf.train.Scaffold()
    hooks = get_training_session_run_hooks(
        save_path, train_loss, dev_loss, scaffold)

    with monitored_session(save_path, scaffold, hooks=hooks) as sess:
        while not sess.should_stop():
            sess.run([merged, train_op])

if __name__ == '__main__':
    main()
