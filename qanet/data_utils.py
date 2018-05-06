import numpy as np
import logging
import tensorflow as tf

from collections import namedtuple

from .preprocess import CategoricalVocabulary

logger = logging.getLogger(__name__)

class SQuADSequence(tf.keras.utils.Sequence):
    """SQuADデータのSequence

    qanet.preprocess.TransformedOutput形式のdataを処理する
    max_context_lengthよりcontextが長いデータは除外される
    max_question_lengthよりquestionが長いデータは除外される

    返却されるデータは、
      x: [
        context,
        context_unk_label,
        context_chars,
        context_mask,
        question,
        question_unk_label,
        question_chars,
        question_mask
      ]
      y: [
        answer_start,
        answer_end
      ]
    yについてはy答えのリストの先頭要素を返す
    """

    def __init__(self,
                 data,
                 batch_size,
                 max_context_length,
                 max_question_length,
                 max_word_length,
                 sort):
        self._batch_size = batch_size
        self._max_context_length = max_context_length
        self._max_question_length = max_question_length
        self._max_word_length = max_word_length
        self._sort = sort

        logger.info('Preprocessing data...')
        ids, self._x_set, self._y_set = self._preprocess(data)

        self._valid_data = _ListedData(ids, self._x_set, self._y_set)

    def __len__(self):
        return int(np.ceil(len(self._x_set) / float(self._batch_size)))

    def __getitem__(self, idx):
        batch_idx = slice(idx * self._batch_size, (idx + 1) * self._batch_size)
        x = self._x_set[batch_idx]
        y = self._y_set[batch_idx]

        batch_x = [np.array([getattr(x_, f) for x_ in x])
                   for f in x[0]._fields]
        batch_y = [
            tf.keras.utils.to_categorical(
                [y_[0] for y_ in y], num_classes=self._max_context_length),
            tf.keras.utils.to_categorical(
                [y_[1] for y_ in y], num_classes=self._max_context_length)]

        return batch_x, batch_y

    def _preprocess(self, data):
        valid_data = []

        # contextまたはquestionが長すぎるものはフィルタリングする
        for idx, datum in enumerate(data):
            if len(datum.x.context) > self._max_context_length:
                logger.warn('Take away datum due to too long context'
                            ', {}, {}'.format(datum.title, datum.id))
                continue

            if len(datum.x.question) > self._max_question_length:
                logger.warn('Take away datum due to too long question'
                            ', {}, {}'.format(datum.title, datum.id))
                continue

            valid_data.append(datum)

        logger.info('{} data filtered, total data size: {}'.format(
            len(data) - len(valid_data), len(valid_data)))

        if self._sort:
            # batch the examples by length
            valid_data = sorted(valid_data, key=lambda d: len(d.x.context))

        ids = [datum.id for datum in valid_data]

        x_set_ = [pad_datum(datum.x,
                            max_context_length=self._max_context_length,
                            max_question_length=self._max_question_length,
                            max_word_length=self._max_word_length)
                  for datum in valid_data]
        # yは、trainデータにおいて答えを一つしか含まないので、これを取り出して返す
        y_set_ = [[datum.y_list[0].answer_start, datum.y_list[0].answer_end]
                  for datum in valid_data]

        return ids, x_set_, y_set_

    @property
    def valid_data(self):
        """前処理でフィルタリングされなかったデータを返す
        idとの紐づけが欲しい評価時用のプロパティ
        """
        return self._valid_data

class PaddedDatum(namedtuple('PaddedDatum', [
        'context',
        'context_unk_label',
        'context_chars',
        'context_mask',
        'question',
        'question_unk_label',
        'question_chars',
        'question_mask'])):
    __slots__ = ()

def pad_datum(datum,
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
        return np.array([1.] * len(x) + [0.] * (max_length - len(x)))

    assert len(datum.context) <= max_context_length
    assert len(datum.question) <= max_question_length

    return PaddedDatum(
        context=pad(datum.context, max_context_length),
        context_unk_label=pad(datum.context_unk_label, max_context_length),
        context_chars=pad_chars(datum.context_chars, max_context_length),
        context_mask=mask(datum.context, max_context_length),
        question=pad(datum.question, max_question_length),
        question_unk_label=pad(datum.question_unk_label, max_question_length),
        question_chars=pad_chars(datum.question_chars, max_question_length),
        question_mask=mask(datum.question, max_question_length))

class _ListedData(object):
    def __init__(self, ids, x, y):
        self._ids = ids
        self._x = x
        self._y = y

    @property
    def ids(self):
        return self._ids

    @property
    def x(self):
        return [np.array([getattr(x_, f) for x_ in self._x])
                for f in self._x[0]._fields]

    @property
    def raw_x(self):
        return self._x
