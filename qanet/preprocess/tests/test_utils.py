import unittest

from nose.tools import ok_, eq_

from ..utils import expand_article, normalize

class TestUtils(unittest.TestCase):
    def test_expand_article(self):
        data = {'title': 'Super_Bowl_50',
             'paragraphs': [
                 {'context': 'Super Bowl 50 was...',
                  'qas': [
                      {'id': '12345',
                       'question': 'Which NFL team...',
                       'answers': [
                           {'answer_start': 123,
                            'text': 'Denver Broncos'},
                           {'answer_start': 123,
                            'text': 'Denver Broncos'}]}]},
                 {'context': 'Superior Bowl 50 was...',
                  'qas': [
                      {'id': '1234567',
                       'question': 'Which OOO team...',
                       'answers': [
                           {'answer_start': 1234,
                            'text': 'Denver Broncos2'},
                           {'answer_start': 123,
                            'text': 'Denver Broncos'}]},
                      {'id': '12345678',
                       'question': 'Which OOO team...',
                       'answers': [
                           {'answer_start': 1234,
                            'text': 'Denver Broncos2'},
                           {'answer_start': 123,
                            'text': 'Denver Broncos'}]}]}]}

        expanded = expand_article(data)

        eq_(expanded[0].id, data['paragraphs'][0]['qas'][0]['id'])
        eq_(expanded[0].title, data['title'])
        eq_(expanded[0].context, data['paragraphs'][0]['context'])
        eq_(expanded[0].question, data['paragraphs'][0]['qas'][0]['question'])
        eq_(expanded[0].answers[0].answer_start,
            data['paragraphs'][0]['qas'][0]['answers'][0]['answer_start'])
        eq_(expanded[0].answers[0].text,
            data['paragraphs'][0]['qas'][0]['answers'][0]['text'])

        eq_(expanded[1].id, data['paragraphs'][1]['qas'][0]['id'])
        eq_(expanded[1].title, data['title'])
        eq_(expanded[1].context, data['paragraphs'][1]['context'])
        eq_(expanded[1].question, data['paragraphs'][1]['qas'][0]['question'])
        eq_(expanded[1].answers[0].answer_start,
            data['paragraphs'][1]['qas'][0]['answers'][0]['answer_start'])
        eq_(expanded[1].answers[0].text,
            data['paragraphs'][1]['qas'][0]['answers'][0]['text'])

        eq_(expanded[2].id, data['paragraphs'][1]['qas'][1]['id'])
        eq_(expanded[2].title, data['title'])
        eq_(expanded[2].context, data['paragraphs'][1]['context'])
        eq_(expanded[2].question, data['paragraphs'][1]['qas'][1]['question'])
        eq_(expanded[2].answers[0].answer_start,
            data['paragraphs'][1]['qas'][1]['answers'][0]['answer_start'])
        eq_(expanded[2].answers[0].text,
            data['paragraphs'][1]['qas'][1]['answers'][0]['text'])

    def test_normalize_uncase(self):
        eq_(normalize('Hello'), 'hello')

    def test_normalize_unicode(self):
        text = u'\u1E9B\u0323'
        eq_([ord(x) for x in text], [7835, 803])
        text = normalize(text)
        eq_([ord(x) for x in text], [383, 803, 775])
