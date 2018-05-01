from collections import OrderedDict

from .utils import identity

class CategoricalVocabulary(object):
    """辞書"""

    PAD_ID = 0
    UNK_ID = 1
    PAD_TOKEN = '<pad>'
    UNK_TOKEN = '<unk>'

    def __init__(self, normalize=identity):
        self._normalize = normalize
        self._mapping = OrderedDict()
        self._mapping[self.PAD_TOKEN] = self.PAD_ID
        self._mapping[self.UNK_TOKEN] = self.UNK_ID
        self._reverse_mapping = {
            self.PAD_ID: self.PAD_TOKEN,
            self.UNK_ID: self.UNK_TOKEN}
        self._freeze = False
        self._unk_mapping = {}

    def get(self, category):
        category = self._normalize(category)

        if self._freeze:
            if category in self._mapping and category not in self._unk_mapping:
                return self._mapping[category]
            else:
                return self.UNK_ID
        else:
            self.add(category)
            return self._mapping[category]

    def add(self, category):
        category = self._normalize(category)

        if not category in self._mapping:
            category_id = len(self._mapping)
            self._mapping[category] = category_id
            self._reverse_mapping[category_id] = category

    def reverse(self, category_id):
        return self._reverse_mapping[category_id]

    def freeze(self):
        self._freeze = True

    def items(self):
        return self._mapping.items()

    def move_to_unk(self, category):
        if self._freeze:
            return
        self._unk_mapping[category] = 1
        # reverse_mappingからは消しちゃう
        del self._reverse_mapping[self._mapping[category]]

    def __len__(self):
        return len(self._mapping)

