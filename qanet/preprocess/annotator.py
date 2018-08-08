from collections import namedtuple
from nltk.tokenize.regexp import RegexpTokenizer

# The pattern used for tokenize
# References: 『入門 自然言語処理』
PATTERN = r'''(?x)      # set flag to allow verbose regexps
    (?:[A-Z]\.)+        # abbreviations, e.g. U.S.A.
  | \d+\.?\d+           # digits
  | \w+                 # words with optional internal hyphens
  | \.\.\.              # ellipsis
  | [][.,;"'?():_`-]    # these are separate tokens; includes ], [
  | [#£\$%−°>€]         # characters that appeared in SQuAD data
'''

_tokenizer = RegexpTokenizer(PATTERN)

class AnnotateOutput(namedtuple('AnnotateOutput', [
        'surface',
        'offset_begin',
        'offset_end'])):
    __slots__ = ()

def annotate(sentence):
    return [AnnotateOutput(
        surface=surface,
        offset_begin=offset_begin,
        offset_end=offset_end) for surface, (offset_begin, offset_end)
            in zip(_tokenizer.tokenize(sentence),
                   _tokenizer.span_tokenize(sentence))]
