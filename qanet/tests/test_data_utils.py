import numpy as np
import unittest

from nose.tools import ok_, eq_

from ..data_utils import SQuADSequence, pad_datum
from ..preprocess import Input, Label, TransformedOutput
from ..preprocess.categorical_vocabulary import CategoricalVocabulary

PAD_ID = CategoricalVocabulary.PAD_ID

np.random.seed(1234)

class TestDataUtils(unittest.TestCase):
    def test_pad_datum(self):
        datum = create_random_input()

        ok_(any([len(x) > 12 for x in datum.context_chars]))
        ok_(any([len(x) > 12 for x in datum.question_chars]))

        padded = pad_datum(datum,
                           max_context_length=150,
                           max_question_length=15,
                           max_word_length=12)

        # context
        ok_(all(np.isclose(
            padded.context[:100], datum.context)))
        ok_(all(np.isclose(
            padded.context[100:], [PAD_ID] * 50)))
        ok_(all(np.isclose(
            padded.context_unk_label[:100], datum.context_unk_label)))
        ok_(all(np.isclose(
            padded.context_unk_label[100:], [PAD_ID] * 50)))

        eq_(padded.context_chars.shape[0], 150)

        for idx, char_ids in enumerate(padded.context_chars):
            ok_(len(char_ids), 12)

            if idx < 100:
                ok_(all(np.isclose(
                    char_ids[:len(datum.context_chars[idx])],
                    datum.context_chars[idx][:12])))
                ok_(all(np.isclose(
                    char_ids[len(datum.context_chars[idx]):],
                    [PAD_ID] * (12 - len(datum.context_chars[idx])))))
            else:
                ok_(all(np.isclose(char_ids, np.zeros(12))))

        ok_(all(np.isclose(
            padded.context_mask,
            np.array([1.] * 100 + [0.] * 50))))

        # question
        ok_(all(np.isclose(
            padded.question[:10], datum.question)))
        ok_(all(np.isclose(
            padded.question[10:], [PAD_ID] * 5)))
        ok_(all(np.isclose(
            padded.question_unk_label[:10], datum.question_unk_label)))
        ok_(all(np.isclose(
            padded.question_unk_label[10:], [PAD_ID] * 5)))

        eq_(padded.question_chars.shape[0], 15)

        for idx, char_ids in enumerate(padded.question_chars):
            ok_(len(char_ids), 12)

            if idx < 10:
                ok_(all(np.isclose(
                    char_ids[:len(datum.question_chars[idx])],
                    datum.question_chars[idx][:12])))
                ok_(all(np.isclose(
                    char_ids[len(datum.question_chars[idx]):],
                    [PAD_ID] * (12 - len(datum.question_chars[idx])))))
            else:
                ok_(all(np.isclose(char_ids, np.zeros(12))))

        ok_(all(np.isclose(
            padded.question_mask,
            np.array([1.] * 10 + [0.] * 5))))

    def test_pad_datum_illegal_length(self):
        datum = create_random_input()

        err = None

        try:
            padded = pad_datum(datum,
                               max_context_length=90,
                               max_question_length=15,
                               max_word_length=12)
        except Exception as e:
            err = e

        ok_(err is not None)

        err2 = None

        try:
            padded = pad_datum(datum,
                               max_context_length=150,
                               max_question_length=5,
                               max_word_length=12)
        except Exception as e:
            err2 = e

        ok_(err2 is not None)

class TestDataUtilsSQuADSequence(unittest.TestCase):
    def test_getitem(self):
        size = 15
        data = [
            TransformedOutput(
            id='hoge',
            title='piyo',
            x=create_random_input(content_length=101),
            y_list=create_random_label_list(3))
            for _ in range(2)
        ] + [TransformedOutput(
            id='hoge',
            title='piyo',
            x=create_random_input(),
            y_list=create_random_label_list(3))
                for _ in range(size - 2)]

        batch_size = 4

        sequence = SQuADSequence(
            data,
            batch_size=batch_size,
            max_context_length=110,
            max_question_length=15,
            max_word_length=12,
            sort=True)

        eq_(len(sequence), size // batch_size + 1)

        batched_x, batched_y = sequence[0]

        eq_(batched_x[0].shape[0], batch_size)
        eq_(batched_y[0].shape[0], batch_size)

        # mask分2つ増えている
        eq_(len(batched_x), len(data[0].x) + 2)
        # パディング分増えている
        eq_(batched_x[0].shape[1], 110)
        # 一応中身も確認(ソートされているので先頭要素はdata[2])
        ok_(all(np.isclose(
            batched_x[0][0][:100], data[2].x.context)))
        ok_(all(np.isclose(
            batched_x[0][0][100:], [PAD_ID] * 10)))
        ok_(all(np.isclose(
            batched_x[1][0][:100], data[2].x.context_unk_label)))

        # answerはanswer_startとanswer_endだけ
        eq_(len(batched_y), 2)
        # SQuADSequenceはanswerリストの先頭要素を返す
        # (どうせ学習データはanswerリストの要素数は1)
        eq_(batched_y[0][0][data[2].y_list[0].answer_start], 1.)
        eq_(batched_y[1][0][data[2].y_list[0].answer_end], 1.)

    def test_getitem_sort(self):
        size = 15
        data = [
            TransformedOutput(
            id='hoge',
            title='piyo',
            x=create_random_input(content_length=101),
            y_list=create_random_label_list(3))
            for _ in range(2)
        ] + [TransformedOutput(
            id='hoge',
            title='piyo',
            x=create_random_input(),
            y_list=create_random_label_list(3))
                for _ in range(size - 2)]

        batch_size = 4

        sequence = SQuADSequence(
            data,
            batch_size=batch_size,
            max_context_length=110,
            max_question_length=15,
            max_word_length=12,
            sort=False)

        batched_x, batched_y = sequence[0]

        # 一応中身も確認(ソートされていないので先頭要素はdata[0])
        ok_(all(np.isclose(
            batched_x[0][0][:100], data[0].x.context[:100])))

    def test_getitem_filter_long_context_item(self):
        max_context_length = 100
        max_question_length = 15
        batch_size = 4
        size = 15

        data = [TransformedOutput(
            id='hoge',
            title='piyo',
            x=create_random_input(),
            y_list=create_random_label_list(3))
                for _ in range(size)]

        data[1] = data[1]._replace(x=create_random_input(content_length=110))
        ok_(len(data[1].x.context) > max_context_length)
        data[2] = data[2]._replace(x=create_random_input(question_length=17))
        ok_(len(data[2].x.question) > max_question_length)

        # これでidx=1と2の要素が除かれて、idx=3の要素が前につめられる

        sequence = SQuADSequence(
            data,
            batch_size=batch_size,
            max_context_length=max_context_length,
            max_question_length=max_question_length,
            max_word_length=12,
            sort=True)

        eq_(len(sequence), size // batch_size + 1)
        # 15個が13個になってbatch_size=4なので、最後のバッチには要素が1つ含まれる
        ok_(len(sequence[len(sequence) - 1]), 1)

        batched_x, batched_y = sequence[0]

        ok_(all(np.isclose(
            batched_x[0][1][:100], data[3].x.context)))

def create_random_input(content_length=100, question_length=10):
    return Input(
        context=np.random.randint(
            100, size=content_length).tolist(),
        context_unk_label=np.random.randint(
            2, size=content_length).tolist(),
        context_chars=[
            np.random.randint(100, size=np.random.randint(18)).tolist()
            for _ in range(content_length)],
        question=np.random.randint(
            100, size=question_length).tolist(),
        question_unk_label=np.random.randint(
            2, size=question_length).tolist(),
        question_chars=[
            np.random.randint(100, size=np.random.randint(18)).tolist()
            for _ in range(question_length)])

def create_random_label_list(num):
    return [Label(answer_start=np.random.randint(100),
                  answer_end=np.random.randint(100),
                  raw_text='hello') for _ in range(num)]
