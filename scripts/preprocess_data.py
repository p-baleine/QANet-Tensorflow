"""
SQuADの{train,dev}.jsonから辞書と単語IDに変換したデータを作成して保存する

動かし方:

  python -m scripts.preprocess_data \
    --train-data /path/to/train.json \
    --dev-data /path/to/dev.json \
    --glove /path/to/glove.X.Y.word2vec.bin \
    --out /path/to/save_dir
"""

import click
import json
import logging
import os

from gensim.models import KeyedVectors

from qanet.preprocess import annotate
from qanet.preprocess import expand_article
from qanet.preprocess import normalize
from qanet.preprocess import Preprocessor

logger = logging.getLogger(__name__)

@click.command()
@click.option('--train-data', type=click.File())
@click.option('--dev-data', type=click.File())
@click.option('--glove', type=click.Path(exists=True))
@click.option('--out', type=click.Path(exists=True))
def main(train_data, dev_data, glove, out):
    logger.info('Loading glove file...')
    wv = KeyedVectors.load_word2vec_format(glove, binary=True)

    logger.info('Expanding train articles...')
    train_data = json.load(train_data)
    train_data = sum([expand_article(a) for a in train_data['data']], [])
    logger.info('Expanding dev articles...')
    dev_data = json.load(dev_data)
    dev_data = sum([expand_article(a) for a in dev_data['data']], [])

    processor = Preprocessor(wv, annotate=annotate, normalize=normalize)
    logger.info('Fitting preprocessor...')
    processor.fit(train_data)

    logger.info('Vocabulary size of words: {}'.format(
        len(processor.word_dict)))
    logger.info('Vocabulary size of characters: {}'.format(
        len(processor.word_dict)))

    logger.info('Transforming train data...')
    train_data = processor.transform(train_data)
    logger.info('Transforming dev data...')
    dev_data = processor.transform(dev_data)

    logger.info('Saving preprocessed data...')
    processor.save(out)

    with open(os.path.join(out, 'train.json'), 'wt') as f:
        json.dump(train_data, f)
    with open(os.path.join(out, 'dev.json'), 'wt') as f:
        json.dump(dev_data, f)

if __name__ == '__main__':
    main()
