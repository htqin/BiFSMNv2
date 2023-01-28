import logging
from collections import Counter
from .common import save_pickle
from .common import load_pickle

logger = logging.getLogger(__name__)


class Vocabulary(object):
    def __init__(self,
                 max_size=None,
                 min_freq=None,
                 pad_token="[PAD]",
                 unk_token="[UNK]",
                 cls_token="[CLS]",
                 sep_token="[SEP]",
                 mask_token="[MASK]",
                 add_unused=False):
        self.max_size = max_size
        self.min_freq = min_freq
        self.cls_token = cls_token
        self.sep_token = sep_token
        self.pad_token = pad_token
        self.mask_token = mask_token
        self.unk_token = unk_token
        self.word2idx = {}
        self.idx2word = None
        self.rebuild = True
        self.add_unused = add_unused
        self.word_counter = Counter()
        self.reset()

    def reset(self):
        ctrl_symbols = [
            self.pad_token, self.unk_token, self.cls_token, self.sep_token,
            self.mask_token
        ]
        for index, syb in enumerate(ctrl_symbols):
            self.word2idx[syb] = index

        if self.add_unused:
            for i in range(20):
                self.word2idx[f'[UNUSED{i}]'] = len(self.word2idx)

    def update(self, word_list):
        self.word_counter.update(word_list)

    def add(self, word):
        self.word_counter[word] += 1

    def has_word(self, word):
        return word in self.word2idx

    def to_index(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        if self.unk_token is not None:
            return self.word2idx[self.unk_token]
        else:
            raise ValueError("word {} not in vocabulary".format(word))

    def unknown_idx(self):
        if self.unk_token is None:
            return None
        return self.word2idx[self.unk_token]

    def padding_idx(self):
        if self.pad_token is None:
            return None
        return self.word2idx[self.pad_token]

    def to_word(self, idx):
        return self.idx2word[idx]

    def build_vocab(self):
        max_size = min(self.max_size, len(
            self.word_counter)) if self.max_size else None
        words = self.word_counter.most_common(max_size)
        if self.min_freq is not None:
            words = filter(lambda kv: kv[1] >= self.min_freq, words)
        if self.word2idx:
            words = filter(lambda kv: kv[0] not in self.word2idx, words)
        start_idx = len(self.word2idx)
        self.word2idx.update(
            {w: i + start_idx
             for i, (w, _) in enumerate(words)})
        self.build_reverse_vocab()
        self.rebuild = False

    def save(self, file_path):
        mappings = {"word2idx": self.word2idx, 'idx2word': self.idx2word}
        save_pickle(data=mappings, file_path=file_path)

    def save_bert_vocab(self, file_path):
        bert_vocab = [x for x, y in self.word2idx.items()]
        with open(str(file_path), 'w') as fo:
            for token in bert_vocab:
                fo.write(token + "\n")

    def load_from_file(self, file_path):
        mappings = load_pickle(input_file=file_path)
        self.idx2word = mappings['idx2word']
        self.word2idx = mappings['word2idx']

    def build_reverse_vocab(self):
        self.idx2word = {i: w for w, i in self.word2idx.items()}

    def read_data(self, data_path):
        if data_path.is_dir():
            files = sorted([f for f in data_path.iterdir() if f.exists()])
        else:
            files = [data_path]
        for file in files:
            f = open(file, 'r')
            lines = f.readlines()
            for line in lines:
                line = line.strip("\n")
                words = line.split(" ")
                self.update(words)

    def clear(self):
        self.word_counter.clear()
        self.word2idx = None
        self.idx2word = None
        self.rebuild = True
        self.reset()

    def __len__(self):
        return len(self.idx2word)
