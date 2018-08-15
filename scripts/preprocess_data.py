"""
Preprocess SQuAD's {train,dev}.json files.

Usage:

  python -m scripts.preprocess_data \
    --train-data /path/to/train.json \
    --dev-data /path/to/dev.json \
    --glove /path/to/glove.X.Y.word2vec.bin \
    --out /path/to/save_dir
"""

import click
import json
import logging
import numpy as np
import os
import tensorflow as tf

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
@click.option('--max-context-length', type=click.INT, default=400)
@click.option('--max-question-length', type=click.INT, default=30)
@click.option('--dev-max-context-length', type=click.INT, default=1000)
@click.option('--dev-max-question-length', type=click.INT, default=100)
@click.option('--max-word-length', type=click.INT, default=16)
@click.option('--char-count-threshold', type=click.INT, default=0)
def main(train_data, dev_data, glove, out, max_context_length,
         max_question_length, max_word_length, dev_max_context_length,
         dev_max_question_length, char_count_threshold):
    logger.info('Loading glove file...')
    wv = KeyedVectors.load_word2vec_format(glove, binary=True)

    logger.info('Expanding train articles...')
    train_data = json.load(train_data)
    train_data = sum([expand_article(a) for a in train_data['data']], [])
    logger.info('Expanding dev articles...')
    dev_data = json.load(dev_data)
    dev_data = sum([expand_article(a) for a in dev_data['data']], [])

    processor = Preprocessor(
        wv,
        annotate=annotate,
        normalize=normalize,
        max_word_length=max_word_length,
        char_count_threshold=char_count_threshold)

    logger.info('Fitting preprocessor...')
    processor.fit(train_data + dev_data)

    logger.info('Vocabulary size of words: {}'.format(
        len(processor.word_dict)))
    logger.info('Vocabulary size of characters: {}'.format(
        len(processor.char_dict)))

    logger.info('Saving preprocessor...')
    processor.save(out)

    logger.info('Transforming train data...')
    train_data = processor.transform(
        train_data,
        max_context_length=max_context_length,
        max_question_length=max_question_length)
    convert_to(train_data, os.path.join(out, 'train.tfrecord'))

    logger.info('Transforming dev data...')
    dev_data = processor.transform(
        dev_data,
        max_context_length=dev_max_context_length,
        max_question_length=dev_max_question_length)
    convert_to(dev_data, os.path.join(out, 'dev.tfrecord'))

def convert_to(data_set, file_path):
    logger.info('Writing {}...'.format(file_path))

    with tf.python_io.TFRecordWriter(file_path) as writer:
        for datum in data_set:
            id, _, x, y_list = datum
            answer_start, answer_end, _ = y_list[0]
            label_features = []

            example = tf.train.SequenceExample(
                context=tf.train.Features(feature={
                    'id': _byte_feature(id)
                }),
                feature_lists=tf.train.FeatureLists(feature_list={
                    'context': _int64_feature_list(x.context),
                    'context_unk_label': _int64_feature_list(x.context_unk_label),
                    'context_chars': _int64_feature_list(x.context_chars),
                    'question': _int64_feature_list(x.question),
                    'question_unk_label': _int64_feature_list(x.question_unk_label),
                    'question_chars': _int64_feature_list(x.question_chars),
                    'answer_start': _int64_feature_list(np.array([answer_start])),
                    'answer_end': _int64_feature_list(np.array([answer_end])),
                }))

            writer.write(example.SerializeToString())

def _byte_feature(value):
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))

def _int64_feature_list(values):
    is_nested = len(values.shape) == 2
    return tf.train.FeatureList(feature=[
        tf.train.Feature(int64_list=tf.train.Int64List(
            value=v if is_nested else [v])) for v in values])

if __name__ == '__main__':
    main()
