import json
import numpy as np
import os
import unittest

from collections import namedtuple, OrderedDict
from nose.tools import ok_, eq_

from ..annotator import annotate
from ..preprocessor import Preprocessor
from ..utils import ExpandedArticle, Answer
from ..utils import expand_article, normalize
from ..categorical_vocabulary import CategoricalVocabulary

np.random.seed(1234)

def tokenize(sentence):
    return sentence.split()

class TestPreprocessor(unittest.TestCase):
    def test_fit(self):
        data = [
            ExpandedArticle(
                id='12345',
                title='Super_Bowl_50',
                context='This is a test',
                question='Is this a test?',
                answers=[Answer(answer_start=1, text='This is a test')])
        ]

        dim = 10
        wv = _Word2VecMock(['This', 'is', 'a', 'test'], dim)
        processor = Preprocessor(wv, annotate=annotate, normalize=normalize)
        processor.fit(data)

        ok_(np.allclose(
            processor.vectors[processor.word_dict.get('test')],
            wv['test']))
        eq_(processor.vectors.shape[1], dim)

        all_chars = list(set(normalize(
            data[0].context + data[0].question).replace(' ', '')))

        ok_(all([processor.char_dict.get(x) != CategoricalVocabulary.UNK_ID
                 for x in all_chars]))

    def test_transform_x(self):
        data = load_fixture_data()
        data = sum([expand_article(a) for a in data], [])
        wv = _Word2VecMock(['super', 'bowl', 'was', '50',
                            'which', 'nfl', 'team'])
        processor = Preprocessor(wv, annotate=annotate, normalize=normalize)

        processor.fit(data)
        ok_(processor.word_dict.get('super') != CategoricalVocabulary.UNK_ID)
        ok_(processor.word_dict.get('bowl') != CategoricalVocabulary.UNK_ID)
        ok_(processor.word_dict.get('50') != CategoricalVocabulary.UNK_ID)
        ok_(processor.word_dict.get('was') != CategoricalVocabulary.UNK_ID)
        ok_(processor.word_dict.get('an') == CategoricalVocabulary.UNK_ID)
        ok_(processor.word_dict.get('which') != CategoricalVocabulary.UNK_ID)
        ok_(processor.word_dict.get('nfl') != CategoricalVocabulary.UNK_ID)
        ok_(processor.word_dict.get('team') != CategoricalVocabulary.UNK_ID)
        ok_(processor.word_dict.get('represented') == CategoricalVocabulary.UNK_ID)

        transformed = processor.transform(data)
        datum = transformed[0]

        eq_(datum.id, '56be4db0acb8001400a502ec')
        eq_(datum.title, 'Super_Bowl_50')
        # 小文字で登録されているはず
        eq_(datum.x.context[0], processor.word_dict.get('super'))
        eq_(datum.x.context[1], processor.word_dict.get('bowl'))
        eq_(datum.x.context[2], processor.word_dict.get('50'))
        eq_(datum.x.context[3], processor.word_dict.get('was'))
        eq_(datum.x.context[4], processor.word_dict.get('an'))
        # an以降はword2vecのモックに入れてない
        eq_(datum.x.context[4], CategoricalVocabulary.UNK_ID)
        eq_(datum.x.context_unk_label[:6],
            [0, 0, 0, 0, 1, 1])
        eq_(datum.x.question[0], processor.word_dict.get('which'))
        eq_(datum.x.question[1], processor.word_dict.get('nfl'))
        eq_(datum.x.question[2], processor.word_dict.get('team'))
        eq_(datum.x.question[3], processor.word_dict.get('represented'))
        # represented以降はword2vecのモックに入れていない
        eq_(datum.x.question[3], CategoricalVocabulary.UNK_ID)
        eq_(datum.x.question_unk_label[:5],
            [0, 0, 0, 1, 1])
        eq_(datum.x.context_chars[0][0], processor.char_dict.get('s'))
        eq_(datum.x.context_chars[0][1], processor.char_dict.get('u'))
        eq_(datum.x.context_chars[0][2], processor.char_dict.get('p'))
        eq_(datum.x.context_chars[0][3], processor.char_dict.get('e'))
        eq_(datum.x.context_chars[0][4], processor.char_dict.get('r'))
        eq_(datum.x.question_chars[1][0], processor.char_dict.get('n'))
        eq_(datum.x.question_chars[1][1], processor.char_dict.get('f'))
        eq_(datum.x.question_chars[1][2], processor.char_dict.get('l'))

    def test_transform_y(self):
        data = load_fixture_data()
        data = sum([expand_article(a) for a in data], [])
        wv = _Word2VecMock(['super', 'bowl', 'was', '50',
                            'which', 'nfl', 'team', 'denver', 'broncos',
                            'santa', ',', 'clara', 'california'])
        processor = Preprocessor(wv, annotate=annotate, normalize=normalize)

        processor.fit(data)

        transformed = processor.transform(data)
        datum = transformed[0]

        eq_(datum.id, '56be4db0acb8001400a502ec')
        eq_(datum.title, 'Super_Bowl_50')

        eq_(datum.y_list[0].raw_text, 'Denver Broncos')
        eq_(datum.y_list[0].answer_start, 33)
        eq_(datum.y_list[0].answer_end, 34)
        eq_(processor.word_dict.reverse(
            datum.x.context[datum.y_list[0].answer_start]),
            'denver')
        eq_(processor.word_dict.reverse(
            datum.x.context[datum.y_list[0].answer_end]),
            'broncos')

        datum = transformed[2]

        eq_(datum.id, '56be4db0acb8001400a502ee')
        eq_(datum.title, 'Super_Bowl_50')

        eq_(datum.y_list[0].raw_text, 'Santa Clara, California')
        eq_(datum.y_list[0].answer_start, 78)
        eq_(datum.y_list[0].answer_end, 81)
        eq_(processor.word_dict.reverse(
            datum.x.context[datum.y_list[0].answer_start]),
            'santa')
        eq_(processor.word_dict.reverse(
            datum.x.context[datum.y_list[0].answer_end]),
            'california')

class _Word2VecMock(object):
    class Vocab(namedtuple('Vocab', ['word', 'index', 'vector'])):
        pass

    def __init__(self, data, dim=10):
        self._dim = dim
        self._vocab = OrderedDict(
        (word, _Word2VecMock.Vocab(word, index, np.random.randn(dim)))
            for index, word in enumerate(data))

    @property
    def vocab(self):
        return self._vocab

    @property
    def index2word(self):
        return list(self._vocab.keys())

    @property
    def vector_size(self):
        return self._dim

    def __contains__(self, key):
        return key in self._vocab

    def __getitem__(self, key):
        return self._vocab[key].vector

_FIXTURE_DATA_PATH = os.path.join(
    os.path.dirname(__file__),
    'fixtures',
    'small-data.json')

def load_fixture_data():
    with open(_FIXTURE_DATA_PATH) as f:
        return json.load(f)['data']
