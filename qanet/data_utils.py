import numpy as np
import logging
import tensorflow as tf

from collections import namedtuple

from .preprocess import CategoricalVocabulary

logger = logging.getLogger(__name__)

class PaddedInput(namedtuple('PaddedInput', [
        'context',
        'context_unk_label',
        'context_chars',
        'context_mask',
        'question',
        'question_unk_label',
        'question_chars',
        'question_mask'])):
    __slots__ = ()

def create_transposed_data(
        raw_data,
        max_context_length,
        max_question_length,
        max_word_length):
    """学習用に加工したデータを返す

    qanet.preprocess.TransformedOutput形式のdataを処理する
    max_context_lengthよりcontextが長いデータは除外される
    max_question_lengthよりquestionが長いデータは除外される
    """
    valid_data = []

    # contextとquestionが閾値を越えていた場合これを除く
    for datum in raw_data:
        if len(datum.x.context) > max_context_length:
            logger.warn('Take away datum due to too long context'
                        ', {}, {}'.format(datum.title, datum.id))
            continue

        if len(datum.x.question) > max_question_length:
            logger.warn('Take away datum due to too long question'
                        ', {}, {}'.format(datum.title, datum.id))
            continue

        valid_data.append(datum)

    logger.info('{} data filtered, total data size: {}'.format(
        len(raw_data) - len(valid_data), len(valid_data)))

    # `x`をパディング
    valid_data = [d._replace(
        x=pad_input(d.x,
                    max_context_length=max_context_length,
                    max_question_length=max_question_length,
                    max_word_length=max_word_length)) for d in valid_data]

    id, title, x, y = list(zip(*valid_data))

    # 学習時には答えの先頭要素のみを用いる
    y = [(y_[0].answer_start, y_[0].answer_end) for y_ in y]

    return (
        id,
        title,
        PaddedInput._make([np.array(x_) for x_ in zip(*x)]),
        tuple(np.array(y_) for y_ in zip(*y)))

def pad_input(datum,
              max_context_length,
              max_question_length,
              max_word_length,
              pad_id=CategoricalVocabulary.PAD_ID):
    """datumのpaddingを行う
    datumはqanet.preprocess.Inputの形式を期待する

    max_context_length分datum.contextとdatum.context_unk_label、
    datum.context_charsにpaddingを行う
    max_context_lengthを越える長さを持っている場合エラーを投げる

    max_question_length分datum.questionとdatum.question_unk_label、
    datum.question_charsにpaddingを行う
    max_question_lengthを越える長さを持っている場合エラーを投げる

    datum.context_chars、datum.question_chars共に
    max_word_length分paddingを行う
    max_word_lengthを越えた要素は切り詰められる
    """
    def pad(x, max_length):
        return np.array(x[:max_length] + [pad_id] * (max_length - len(x)))

    def pad_chars(x, max_sentence_length):
        return np.array(
            [pad(x_[:max_sentence_length], max_word_length) for x_ in x]
            + [[pad_id] * max_word_length] * (max_sentence_length - len(x)))

    def mask(x, max_length):
        return np.array([1.] * len(x) + [0.] * (max_length - len(x)),
                        dtype=np.float32)

    assert len(datum.context) <= max_context_length
    assert len(datum.question) <= max_question_length

    return PaddedInput(
        context=pad(datum.context, max_context_length),
        context_unk_label=pad(datum.context_unk_label, max_context_length),
        context_chars=pad_chars(datum.context_chars, max_context_length),
        context_mask=mask(datum.context, max_context_length),
        question=pad(datum.question, max_question_length),
        question_unk_label=pad(datum.question_unk_label, max_question_length),
        question_chars=pad_chars(datum.question_chars, max_question_length),
        question_mask=mask(datum.question, max_question_length))
