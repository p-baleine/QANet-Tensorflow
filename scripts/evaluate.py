"""
Evaluate by SQuAD's script.

Usage:

  python -m scripts.evaluate \
    --data /path/to/data \
    --raw-data-file /path/to/raw/squad/data/file \
    --save-path /path/to/save_dir
"""

import click
import json
import logging
import numpy as np
import os
import sys
import tensorflow as tf

sys.path.append(os.path.join(
    os.path.dirname(__file__), '..', 'SQuAD_scripts/'))

import squad

import qanet.model as qanet_model

from qanet.model_utils import DatasetInitializerHook
from qanet.model_utils import create_iterator, get_answer_spane
from qanet.model_utils import load_data, load_embedding, load_hparams
from qanet.model_utils import monitored_session
from qanet.preprocess import annotate
from qanet.preprocess import expand_article
from qanet.preprocess import Preprocessor

logger = logging.getLogger(__name__)

tf.logging.set_verbosity(tf.logging.INFO)

@click.command()
@click.option('--data', type=click.Path())
@click.option('--save-path', type=click.Path(exists=True))
@click.option('--raw-data-file', type=click.File())
@click.option('--use-ema', type=click.BOOL, default=True)
def main(data, save_path, raw_data_file, use_ema):
    hparams = load_hparams(save_path)

    logger.info('Hyper parameters:')
    logger.info(json.dumps(json.loads(hparams.to_json()), indent=2))

    logger.info('Loading data...')

    _, dev_data = load_data(data)
    embedding = load_embedding(data)

    id, _, iterator, feed_dict = create_iterator(
        dev_data, hparams, do_shuffle=False, repeat_count=None)
    inputs, _ = iterator.get_next()

    logger.info('Preparing model...')

    model = qanet_model.QANet(embedding, hparams)

    logger.info('Starting prediction...')

    prediction_op = model(inputs, training=False)

    if use_ema:
        logger.info('Using shadow variables.')
        ema = tf.train.ExponentialMovingAverage(decay=hparams.ema_decay)
        saver = tf.train.Saver(ema.variables_to_restore())
        scaffold = tf.train.Scaffold(saver=saver)
    else:
        scaffold = tf.train.Scaffold()

    starts = []
    ends = []

    with monitored_session(
            save_path, scaffold,
            hooks=[DatasetInitializerHook(iterator, feed_dict)]) as sess:
        while not sess.should_stop():
            start, end = sess.run(prediction_op)
            starts += start.tolist()
            ends += end.tolist()

    # TODO Restore without the filename.
    processor = Preprocessor.restore(os.path.join(data, 'preprocessor.pickle'))
    raw_dataset = json.load(raw_data_file)['data']

    print(json.dumps(squad.evaluate(
        raw_dataset, prediction_mapping(
            id, starts, ends, raw_dataset, processor))))

def prediction_mapping(id, starts, ends, raw_data, processor):
    data = sum([expand_article(a) for a in raw_data], [])
    context_mapping = dict((d.id, (d.context, annotate(d.context)))
                           for d in data)
    starts, ends, _ = zip(*[get_answer_spane(s, e) for s, e in
                            zip(starts, ends)])
    prediction_mapping = {}

    for id, start, end in zip(id, starts, ends):
        context, annotated_context = context_mapping[id]
        offset_begin = annotated_context[start].offset_begin
        offset_end = annotated_context[end].offset_end
        prediction_mapping[id] = context[offset_begin:offset_end+1]

    return prediction_mapping

if __name__ == '__main__':
    main()
