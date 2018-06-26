import json
import logging
import numpy as np
import os
import pickle
import tensorflow as tf

from .model import create_model
from .preprocess import TransformedOutput

logger = logging.getLogger(__name__)

def create_or_load_model(hparams, embedding, save_dir, resume_from=None):
    """modelを新たに作成する。resume_fromが指定されている場合は
    resume_fromのweightsでをロードしたモデルを返す
    """
    model = create_model(embedding, hparams)

    if resume_from is not None:
        logger.info('Load weights from {}.'.format(resume_from))
        model.load_weights(resume_from)
    else:
        logger.info('Model created freshly.')
        # hparamsを保存しておく
        saved_hparams_file = os.path.join(save_dir, 'hparams.json')

        with open(saved_hparams_file, 'wt') as f:
            f.write(hparams.to_json())

    return model

def load_hparams(hparams_path):
    with open(hparams_path) as f:
        return tf.contrib.training.HParams(**json.load(f))

def load_data(data_path):
    with open(os.path.join(data_path, 'train.json')) as f:
        train_data = json.load(f)
    with open(os.path.join(data_path, 'dev.json')) as f:
        dev_data = json.load(f)
    return (
        [TransformedOutput.from_raw_array(d) for d in train_data],
        [TransformedOutput.from_raw_array(d) for d in dev_data])

def load_embedding(data_path):
    return np.load(os.path.join(data_path, 'vectors.npy'))

def get_answer_spane(start_preds, end_preds):
    best_score = 0.0
    best_start_idx = 0
    best_span = (0, 1)

    for i in range(len(start_preds)):
        start_score = start_preds[best_start_idx]

        if start_score < start_preds[i]:
            start_score = start_preds[i]
            best_start_idx = i

        end_score = end_preds[i]

        if start_score * end_score > best_score:
            best_span = (best_start_idx, i)
            best_score = start_score * end_score

    return best_span[0], best_span[1], best_score
