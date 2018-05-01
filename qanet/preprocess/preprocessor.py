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

class Preprocessor(object):
    """ExpandedArticleのデータの変換を行う"""

    def __init__(self, word2vec, annotate, normalize=identity):
        self._word2vec = word2vec
        self._word_dict = CategoricalVocabulary(normalize=normalize)
        self._char_dict = CategoricalVocabulary(normalize=normalize)
        self._annotate = annotate
        self._vectors = None

    def fit(self, articles):
        """articlesのcontextとquestionから辞書を学習する
        また、辞書のエントリーに対応するword2vecのvectorを記憶しておく
        """
        for article in articles:
            sentences = [article.context, article.question]
            for w, _, _ in sum((self._annotate(s) for s in sentences), []):
                self._word_dict.add(w)
                for c in list(w):
                    self._char_dict.add(c)

        # word2vecのvectorを取り出す
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
                # word2vec側になかったらunkにする
                self._word_dict.move_to_unk(word)

        self._word_dict.freeze()

    def transform(self, data):
        """dataの各要素をTransformedOutputに変換したリストを返す
        """
        output = []

        for datum in data:
            id, title, context, question, answers = datum
            annotated_context = self._annotate(context)
            annotated_question = self._annotate(question)

            y_list = self._transform_label(annotated_context, answers)

            if len(y_list) == 0:
                # 答えが無かったら無視
                logger.warn('The question dose not have answers, '
                            'article_title: {}, id: {}'.format(title, id))

            output.append(TransformedOutput(
                id=id,
                title=title,
                x=self._transform_input(
                    annotated_context, annotated_question),
                y_list=y_list))

        return output

    def reverse_word_ids(self, word_ids):
        return [self._word_dict.reverse(id) for id in word_ids]

    @property
    def vectors(self):
        """fitの過程で得られた辞書に対応するword2vecのvector
        """
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
