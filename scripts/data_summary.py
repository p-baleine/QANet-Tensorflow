"""
与えられたSQuAD形式のjsonファイルの統計情報を出力する

動かし方:

  python -m scripts.data_summary \
    /path/to/data.json \
    --glove /path/to/glove.X.Y.word2vec.bin
"""

import click
import json
import logging
import numpy as np

from collections import Counter
from gensim.models import KeyedVectors

from qanet.preprocess import annotate
from qanet.preprocess import expand_article
from qanet.preprocess import normalize

logger = logging.getLogger(__name__)

annotate_ = lambda x: [w for w, _, _ in annotate(x)]

@click.command()
@click.argument('data', type=click.File())
@click.option('--glove', type=click.Path(exists=True))
def main(data, glove):
    logger.debug('Loading glove file...')
    wv = KeyedVectors.load_word2vec_format(glove, binary=True)

    data = json.load(data)
    word_counter = Counter()
    unk_words = []
    context_length = []
    question_length = []
    context_longer_than_400 = []
    question_longer_than_30 = []

    logger.debug('Calcurate word count...')

    context_qa_pairs = sum([expand_article(a) for a in data['data']], [])

    for context_qa_pair in context_qa_pairs:
        annotated_context = annotate_(context_qa_pair.context)
        annotated_question = annotate_(context_qa_pair.question)
        for w in annotated_context + annotated_question:
            if normalize(w) not in wv:
                unk_words.append(w)
        word_counter.update(annotated_context + annotated_question)
        context_length.append(len(annotated_context))
        question_length.append(len(annotated_question))

        if len(annotated_context) > 400:
            context_longer_than_400.append(context_qa_pair.id)

        if len(annotated_question) > 30:
            question_longer_than_30.append(context_qa_pair.id)

    total_word_count = sum(val for _, val in word_counter.items())

    print('Data summary:')
    print('\tTotal article count:', len(data))
    print('\tTotal paragraph count:',
          sum([len(datum['paragraphs']) for datum in data['data']]))
    print('\tTotal qa count:', len(context_qa_pairs))
    print('\tTotal word count:', total_word_count)
    print('\tTotal unk word count:', len(unk_words))
    print('\tRatio of unk words:', len(unk_words) / total_word_count)
    print('\tAverage context length:', np.mean(context_length))
    print('\tSD of context length:', np.std(context_length))
    print('\tAverage context length:', np.mean(question_length))
    print('\tSD context length:', np.std(question_length))

    print(context_longer_than_400)
    print(question_longer_than_30)

if __name__ == '__main__':
    main()
