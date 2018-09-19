import logging
import numpy as np
import os
import pickle

from collections import namedtuple

from .utils import identity
from .categorical_vocabulary import CategoricalVocabulary

logger = logging.getLogger(__name__)

class TransformedOutput(namedtuple('TransformedOutput', [
        'id',
        'title',
        'x',
        'y_list'])):
    __slots__ = ()

    @classmethod
    def from_raw_array(cls, arr):
        return TransformedOutput(
            id=arr[0],
            title=arr[1],
            x=Input.from_raw_array(arr[2]),
            y_list=[Label.from_raw_array(a) for a in arr[3]])

class Input(namedtuple('Input', [
        'context',
        'context_unk_label',
        'context_chars',
        'raw_context_words',
        'question',
        'question_unk_label',
        'question_chars'])):
    __slots__ = ()

    @classmethod
    def from_raw_array(cls, arr):
        return Input(*arr)

class Label(namedtuple('Label', [
        'answer_start',
        'answer_end',
        'raw_text'])):
    __slots__ = ()

    @classmethod
    def from_raw_array(cls, arr):
        return Label(*arr)

class PaddedInput(namedtuple('PaddedInput', [
        'context',
        'context_unk_label',
        'context_chars',
        'question',
        'question_unk_label',
        'question_chars'])):
    __slots__ = ()

class Preprocessor(object):
    """Preprocess `ExpandedArticle`.
    """

    def __init__(self, word2vec, annotate, max_word_length,
                 normalize=identity, char_count_threshold=0):
        self._word2vec = word2vec
        self._word_dict = CategoricalVocabulary(normalize=normalize)
        self._char_dict = CategoricalVocabulary(normalize=normalize)
        self._annotate = annotate
        self._vectors = None
        self._max_word_length = max_word_length
        self._char_count_threshold = char_count_threshold

    def fit(self, articles):
        """Fit words from the context and question in `articles`.
        Also save vectors which correspond to learned
        dictionary's entries.
        """
        for article in articles:
            sentences = [article.context, article.question]
            for w, _, _ in sum((self._annotate(s) for s in sentences), []):
                self._word_dict.add(w)
                for c in list(w):
                    self._char_dict.add(c)


        # Retrieve vectors from word2vec.
        self._vectors = np.zeros(
            (len(self._word_dict), self._word2vec.vector_size))

        for word, id in self._word_dict.items():
            if (id == CategoricalVocabulary.UNK_ID
                or id == CategoricalVocabulary.PAD_ID):
                continue
            if word in self._word2vec:
                self._vectors[id] = self._word2vec[word]
            else:
                logger.warn('glove dose not contain word "{}"'.format(word))
                # Make a ward that dose not exist in word2vec.
                self._word_dict.move_to_unk(word)

        logger.info('Triming character vocabulary by min frequency {}'.format(
            self._char_count_threshold))
        self._char_dict.trim(min_frequency=self._char_count_threshold)

        self._char_dict.freeze()
        self._word_dict.freeze()

    def transform(self, data, max_context_length, max_question_length,
                  max_answer_length):
        output = []

        def is_too_long(context, question):
            return (len(context) > max_context_length
                    or len(question) > max_question_length)

        for datum in data:
            id, title, context, question, answers = datum
            annotated_context = self._annotate(context)
            annotated_question = self._annotate(question)

            if is_too_long(annotated_context, annotated_question):
                logger.warn('Filter out too long datum, {}, {}'.format(id, title))
                continue

            y_list = self._transform_label(annotated_context, answers)

            if len(y_list) == 0:
                # Ignore the qas if it dose not have any answers.
                logger.warn('The question dose not have answers, '
                            'article_title: {}, id: {}'.format(title, id))
                continue

            if y_list[0].answer_end - y_list[0].answer_start > max_answer_length:
                logger.warn('Filter long answer, {}, {}'.format(
                    id, y_list[0].raw_text))
                continue

            x = self._transform_input(annotated_context, annotated_question)

            output.append(TransformedOutput(
                id=id,
                title=title,
                x=_padding(x, max_context_length, max_question_length,
                           self._max_word_length),
                y_list=y_list))

        return output

    def reverse_word_ids(self, word_ids):
        return [self._word_dict.reverse(id) for id in word_ids]

    @property
    def vectors(self):
        return self._vectors

    @property
    def word_dict(self):
        return self._word_dict

    @property
    def char_dict(self):
        return self._char_dict

    def save(self, save_dir):
        """インスタンスをsave_pathで指定されたpathにpickleで保存する
        """
        instance_file = os.path.join(save_dir, 'preprocessor.pickle')
        vector_file = os.path.join(save_dir, 'vectors')
        with open(instance_file, 'wb') as f:
            f.write(pickle.dumps(self))
        np.save(vector_file, self._vectors)

    @classmethod
    def restore(cls, file_path):
        """file_pathで指定されたpathからインスタンスを復帰する
        """
        with open(file_path, 'rb') as f:
            return pickle.loads(f.read())

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['_word2vec']
        del state['_vectors']
        return state

    def _transform_input(self, annotated_context, annotated_question):
        def to_word_ids(sentence):
            return [self._word_dict.get(w) for w, _, _ in sentence]

        def to_char_ids(sentence):
            return [[self._char_dict.get(c) for c in w]
                    for w, _, _ in sentence]

        def unk_label(ids):
            return [1 if id == CategoricalVocabulary.UNK_ID else 0 for id in ids]

        context_word_ids = to_word_ids(annotated_context)
        question_word_ids = to_word_ids(annotated_question)

        return Input(
            context=context_word_ids,
            context_unk_label=unk_label(context_word_ids),
            context_chars=to_char_ids(annotated_context),
            raw_context_words=[w for w, _, _ in annotated_context],
            question=question_word_ids,
            question_unk_label=unk_label(question_word_ids),
            question_chars=to_char_ids(annotated_question))

    def _transform_label(self, annotated_context, answers):
        labels = []

        # answer毎にcontextを走査してoffsetが一致する単語が
        # あったらそのanswerを答えとする
        for answer in answers:
            for idx, x in enumerate(annotated_context):
                if answer.answer_start == x.offset_begin:
                    labels.append(Label(
                        answer_start=idx,
                        answer_end=idx + len(self._annotate(answer.text)) - 1,
                        raw_text=answer.text))

        return labels

def _padding(datum,
             max_context_length,
             max_question_length,
             max_word_length,
             pad_id=CategoricalVocabulary.PAD_ID):
    """Padding `datum`
    """

    def pad(x, max_length):
        return np.array(x[:max_length] + [pad_id] * (max_length - len(x)))

    def pad_chars(x, max_sentence_length):
        return np.array(
            [pad(x_[:max_sentence_length], max_word_length) for x_ in x]
            + [[pad_id] * max_word_length] * (max_sentence_length - len(x)))

    assert len(datum.context) <= max_context_length
    assert len(datum.question) <= max_question_length

    return PaddedInput(
        context=pad(datum.context, max_context_length),
        context_unk_label=pad(
            datum.context_unk_label, max_context_length),
        context_chars=pad_chars(
            datum.context_chars, max_context_length),
        question=pad(datum.question, max_question_length),
        question_unk_label=pad(
            datum.question_unk_label, max_question_length),
        question_chars=pad_chars(
            datum.question_chars, max_question_length))
