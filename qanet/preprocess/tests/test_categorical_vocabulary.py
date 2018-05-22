import unittest
import numpy as np

from collections import namedtuple, OrderedDict
from nose.tools import ok_, eq_

from ..categorical_vocabulary import CategoricalVocabulary

class TestCategoricalVocabulary(unittest.TestCase):
    def test_get(self):
        vocab = CategoricalVocabulary()
        eq_(vocab.get('This'), 2) # PADとUNKがあるから2から始まる
        eq_(vocab.get('is'), 3)
        eq_(vocab.get('a'), 4)
        eq_(vocab.get('pen'), 5)

    def test_freeze(self):
        vocab = CategoricalVocabulary()
        eq_(vocab.get('This'), 2)
        eq_(vocab.get('is'), 3)
        eq_(vocab.get('a'), 4)
        eq_(vocab.get('pen'), 5)
        vocab.freeze()
        eq_(vocab.get('This'), 2)

    def test_add(self):
        vocab = CategoricalVocabulary()
        vocab.add('This')
        vocab.add('is')
        vocab.add('a')
        vocab.add('pen')
        eq_(vocab.get('This'), 2)
        eq_(vocab.get('is'), 3)
        eq_(vocab.get('a'), 4)
        eq_(vocab.get('pen'), 5)
        vocab.freeze()
        vocab.add('pen')
        eq_(vocab.get('pen'), 5)

    def test_unk(self):
        vocab = CategoricalVocabulary()
        vocab.add('This')
        vocab.add('is')
        vocab.add('a')
        vocab.add('pen')
        vocab.freeze()
        eq_(vocab.get('ball'), CategoricalVocabulary.UNK_ID)

    def test_reverse(self):
        vocab = CategoricalVocabulary()
        vocab.add('This')
        vocab.add('is')
        vocab.add('a')
        vocab.add('pen')
        vocab.freeze()
        eq_(vocab.reverse(vocab.get('is')), 'is')
        eq_(vocab.reverse(CategoricalVocabulary.UNK_ID),
            CategoricalVocabulary.UNK_TOKEN)
        eq_(vocab.reverse(CategoricalVocabulary.PAD_ID),
            CategoricalVocabulary.PAD_TOKEN)

    def test_length(self):
        vocab = CategoricalVocabulary()
        vocab.add('This')
        vocab.add('is')
        vocab.add('a')
        vocab.add('pen')
        eq_(len(vocab), 6)

    def test_normalize(self):
        def normalize(x):
            return x.upper()
        vocab = CategoricalVocabulary(normalize=normalize)
        vocab.add('This')
        vocab.freeze()
        eq_(vocab.get('THIS'), 2)

    def test_move_to_unk(self):
        vocab = CategoricalVocabulary()
        vocab.add('This')
        vocab.add('is')
        vocab.add('a')
        vocab.add('pen')
        eq_(vocab.get('pen'), 5)
        vocab.move_to_unk('pen')
        vocab.freeze()
        eq_(vocab.get('pen'), CategoricalVocabulary.UNK_ID)

    def test_trim(self):
        vocab = CategoricalVocabulary()

        for _ in range(5):
            print('what??')
            vocab.add('This')
            vocab.add('is')
            vocab.add('a')

        vocab.add('pen')
        vocab.add('pen')
        vocab.add('pen')

        vocab.freeze()

        eq_(vocab.get('This'), 2)
        eq_(vocab.get('is'), 3)
        eq_(vocab.get('a'), 4)
        eq_(vocab.get('pen'), 5)

        vocab.trim(min_frequency=4)

        ok_(vocab.get('This') != CategoricalVocabulary.UNK_ID)
        eq_(vocab.get('pen'), CategoricalVocabulary.UNK_ID)
