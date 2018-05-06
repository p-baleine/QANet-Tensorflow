"""
学習済モデルを評価する

動かし方:

  python -m scripts.evaluate \
    --data /path/to/data \
    --raw-data-file /path/to/raw/squad/data/file \
    --save-path /path/to/save_dir \
    --weights-file /path/to/saved/weights/file
"""

import click
import json
import logging
import numpy as np
import os
import sys

sys.path.append(os.path.join(
    os.path.dirname(__file__), '..', 'SQuAD_scripts/'))

import squad

from qanet import SQuADSequence
from qanet import create_or_load_model
from qanet import load_data, load_embedding, load_hparams
from qanet.preprocess import expand_article
from qanet.preprocess import Preprocessor

logger = logging.getLogger(__name__)

@click.command()
@click.option('--data', type=click.Path())
@click.option('--save-path', type=click.Path(exists=True))
@click.option('--weights-file', type=click.Path(exists=True))
@click.option('--raw-data-file', type=click.File())
def main(data, raw_data_file, save_path, weights_file):
    hparams = load_hparams(os.path.join(save_path, 'hparams.json'))

    logger.info('Hyper parameters:')
    logger.info(json.dumps(json.loads(hparams.to_json()), indent=2))

    logger.info('Loading data...')

    _, dev_data = load_data(data)
    embedding = load_embedding(data)

    dev_gen = SQuADSequence(
        dev_data,
        sort=False,
        batch_size=hparams.batch_size,
        max_context_length=hparams.max_context_length,
        max_question_length=hparams.max_question_length,
        max_word_length=hparams.max_word_length)

    # TODO ファイル名指定しなくてもrestoreできるようにする
    processor = Preprocessor.restore(os.path.join(data, 'preprocessor.pickle'))

    logger.info('Preparing model...')

    model = create_or_load_model(hparams, embedding, save_path,
                                 resume_from=weights_file)

    predictions = model.predict(dev_gen.valid_data.x)
    raw_dataset = json.load(raw_data_file)['data']

    print(json.dumps(squad.evaluate(
        raw_dataset, prediction_mapping(predictions, dev_gen, processor))))

def prediction_mapping(predictions, dev_gen, processor):
    predictions = zip(np.argmax(predictions[0], axis=1),
                      np.argmax(predictions[1], axis=1))
    prediction_mapping = dict(
        (id, ' '.join(processor.reverse_word_ids(x.context[start:end+1])))
         for id, x, (start, end) in zip(dev_gen.valid_data.ids,
                                        dev_gen.valid_data.raw_x,
                                        predictions))
    return prediction_mapping


if __name__ == '__main__':
    main()
