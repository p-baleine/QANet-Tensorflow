import unittest

from nose.tools import ok_, eq_

from ..annotator import annotate

class TestAnnotator(unittest.TestCase):
    def test_annotate_output_fields(self):
        annotated = annotate('Hello world')
        ok_('surface' in annotated[0]._fields)
        ok_('offset_begin' in annotated[0]._fields)
        ok_('offset_end' in annotated[0]._fields)

    def test_annotate(self):
        annotated = annotate('Hello world.')
        eq_([list(x) for x in annotated],
            [['Hello', 0, 5], ['world', 6, 11], ['.', 11, 12]])

    def test_annotate_abbrev(self):
        annotated = annotate('You\'re testing now.')
        eq_([list(x) for x in annotated],
            [['You', 0, 3], ['\'', 3, 4], ['re', 4, 6],
             ['testing', 7, 14], ['now', 15, 18], ['.', 18, 19]])

    def test_annotate_abbrev2(self):
        annotated = annotate('I live in U.S.A.')
        eq_([list(x) for x in annotated],
            [['I', 0, 1], ['live', 2, 6], ['in', 7, 9], ['U.S.A.', 10, 16]])

    def test_annotate_multi_sentences(self):
        annotated = annotate('You\'re testing now. Hello world.')
        eq_([list(x) for x in annotated],
            [['You', 0, 3], ['\'', 3, 4], ['re', 4, 6],
             ['testing', 7, 14], ['now', 15, 18], ['.', 18, 19],
             ['Hello', 20, 25], ['world', 26, 31], ['.', 31, 32]])

    def test_annotate_hyphenated(self):
        annotated = annotate('This is a build-in function.')
        eq_([list(x) for x in annotated],
            [['This', 0, 4], ['is', 5, 7], ['a', 8, 9],
             ['build', 10, 15], ['-', 15, 16], ['in', 16, 18],
             ['function', 19, 27], ['.', 27, 28]])

    def test_annotate_multiplespaces(self):
        annotated = annotate('Hello     world')
        eq_([list(x) for x in annotated],
            [['Hello', 0, 5], ['world', 10, 15]])

    def test_annotate_newline(self):
        annotated = annotate('You\'re testing now. \n Hello world.')
        eq_([list(x) for x in annotated],
            [['You', 0, 3], ['\'', 3, 4], ['re', 4, 6],
             ['testing', 7, 14], ['now', 15, 18], ['.', 18, 19],
             ['Hello', 22, 27], ['world', 28, 33], ['.', 33, 34]])

    def test_annotate_sharp(self):
        annotated = annotate('#P is an important complexity')
        eq_([list(x) for x in annotated][:3],
            [['#', 0, 1], ['P', 1, 2], ['is', 3, 5]])

    def test_annotate_pond(self):
        annotated = annotate('Sky picked up the remaining four for £1.3bn.')
        eq_([list(x) for x in annotated],
            [['Sky', 0, 3], ['picked', 4, 10], ['up', 11, 13], ['the', 14, 17],
             ['remaining', 18, 27], ['four', 28, 32], ['for', 33, 36],
             ['£', 37, 38], ['1.3', 38, 41], ['bn', 41, 43],
             ['.', 43, 44]])
        annotated = annotate('from £18m to £34.002m')
        eq_([list(x) for x in annotated],
            [['from', 0, 4], ['£', 5, 6], ['18', 6, 8], ['m', 8, 9],
             ['to', 10, 12], ['£', 13, 14], ['34.002', 14, 20],
             ['m', 20, 21]])

    def test_annotate_celsius(self):
        annotated = annotate('temperature of −11.7 °C')
        eq_([list(x) for x in annotated],
            [['temperature', 0, 11], ['of', 12, 14], ['−', 15, 16],
             ['11.7', 16, 20], ['°', 21, 22], ['C', 22, 23]])

    def test_annotate_greater(self):
        annotated = annotate('>500 Da')
        eq_([list(x) for x in annotated],
            [['>', 0, 1], ['500', 1, 4], ['Da', 5, 7]])

    def test_annotate_eur(self):
        annotated = annotate('around €5,000')
        eq_([list(x) for x in annotated],
            [['around', 0, 6], ['€', 7, 8], ['5', 8, 9],
             [',', 9, 10], ['000', 10, 13]])

    def test_annotate_end_with_digits(self):
        annotated = annotate('It 1000.')
        eq_([list(x) for x in annotated],
            [['It', 0, 2], ['1000', 3, 7], ['.', 7, 8]])
