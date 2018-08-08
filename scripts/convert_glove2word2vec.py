"""
Convert the glove file to word2vec format and save it.
Save it with binary to speed up the loading time.

Usage:
  python -m scripts.convert_glove2word2vec \
    /path/to/glove.X.Y.txt \
    /path/to/save
"""

import click
import logging
import os
import tempfile

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models.keyedvectors import KeyedVectors

logger = logging.getLogger(__name__)

@click.command()
@click.argument('glove', type=click.Path(exists=True))
@click.argument('out', type=click.STRING)
def main(glove, out):
    ftmp, tmp_path = tempfile.mkstemp()

    logger.info('Converting glove format to word2vec format...')
    glove2word2vec(glove, tmp_path)

    logger.info('Load word2vec format file...')
    wv = KeyedVectors.load_word2vec_format(tmp_path, binary=False)

    logger.info('Saveing word2vec binary format file...')
    wv.save_word2vec_format(out, binary=True)

    os.remove(tmp_path)

if __name__ == '__main__':
    main()
