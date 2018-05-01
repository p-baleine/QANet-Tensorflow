import unicodedata

from collections import namedtuple

class ExpandedArticle(namedtuple('ExpandedArticle', [
        'id',
        'title',
        'context',
        'question',
        'answers'])):
    __slots__ = ()

class Answer(namedtuple('Answer', ['answer_start', 'text'])):
    __slots__ = ()

def expand_article(article):
    """articleを受け取ってcontext-qaのペアに展開して返す

    引数:
      article: SQuADデータの1要素
    戻り値:
      ExpandedArticleのリスト
    """
    return [ExpandedArticle(
        id=qa['id'],
        title=article['title'],
        context=paragraph['context'],
        question=qa['question'],
        answers=[Answer(**a) for a in qa['answers']])
            for paragraph in article['paragraphs']
            for qa in paragraph['qas']]

def normalize(word):
    # glove.6B.300d.txtにおいては全部小文字で登録されている
    return unicodedata.normalize('NFD', word.lower())

identity = lambda x: x
