"""
学習する

動かし方:

  python -m scripts.train \
    --data /path/to/preprocessed_data_dir \
    --hparams /path/to/hparams.json \
    --save_path /path/to/save_dir
"""

# TODO em計算する時、contextが400越えるものは0, 0で答えを用意する

import click
import json
import logging
import os
import tensorflow as tf

from qanet import SQuADSequence
from qanet import create_or_load_model
from qanet import load_data, load_embedding, load_hparams

logger = logging.getLogger(__name__)

@click.command()
@click.option('--data', type=click.Path(exists=True))
@click.option('--hparams', type=click.Path(exists=True))
@click.option('--save-path', type=click.Path(exists=True))
def main(data, hparams, save_path):
    checkpoint_file_path = os.path.join(
        save_path, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')

    hparams = load_hparams(hparams)

    logger.info('Hyper parameters:')
    logger.info(json.dumps(json.loads(hparams.to_json()), indent=2))

    logger.info('Loading data...')

    train_data, dev_data = load_data(data)
    embedding = load_embedding(data)

    seq_params = dict(
        batch_size=hparams.batch_size,
        max_context_length=hparams.max_context_length,
        max_question_length=hparams.max_question_length,
        max_word_length=hparams.max_word_length)

    train_gen = SQuADSequence(train_data, sort=True, **seq_params)
    dev_gen = SQuADSequence(dev_data, sort=False, **seq_params)

    logger.info('Preparing model...')

    model = create_or_load_model(hparams, embedding, save_path)

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.RMSprop(hparams.learning_rate),
        loss='categorical_crossentropy',
        loss_weights=[.5, .5],
        metrics=['accuracy'])

    logger.info('Start training.')

    model.fit_generator(
        train_gen,
        epochs=hparams.epochs,
        verbose=2,
        validation_data=dev_gen,
        shuffle=False,
        callbacks=[
            tf.keras.callbacks.TerminateOnNaN(),
            tf.keras.callbacks.TensorBoard(save_path),
            tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_file_path,
                period=5,
                verbose=1),
        ])

if __name__ == '__main__':
    main()
