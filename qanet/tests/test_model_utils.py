import numpy as np
import unittest

from nose.tools import ok_, eq_

from ..model_utils import get_answer_spane

class TestModelUtils(unittest.TestCase):
    def test_get_answer_span(self):
        # argmaxはidx=1とidx=0だけどspanとしてはidx=1とidx=3が正しい
        start_preds = [.1, .6, .1, .05, .05]
        end_preds = [.4, .2, .05, .3, .05]

        start, end, _ = get_answer_spane(start_preds, end_preds)

        eq_(start, 1)
        eq_(end, 3)
