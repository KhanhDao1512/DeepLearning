import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

IGNORE_INDEX = -100     # nhãn của vị trí padding, CrossEntropyLoss(ignore_index=...) sẽ bỏ qua

def read_conll(path: str) -> list[tuple[list[str], list[str]]]:
    '''
        Đọc file CoNLL của PhoNER_COVID19: mỗi dòng "token nhãn", câu cách nhau bằng dòng trống.
        Trả về list các (words, tags).
    '''
    sentences = []
    words, tags = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if words:
                    sentences.append((words, tags))
                    words, tags = [], []
                continue
            parts = line.split()
            words.append(parts[0])
            tags.append(parts[-1])
    if words:
        sentences.append((words, tags))
    return sentences

# gom các mẫu thành batch
def collate_fn(items: list[dict]) -> dict:
    input_ids = [item["input_ids"] for item in items]
    lengths = torch.tensor([ids.shape[0] for ids in input_ids], dtype=torch.long)
    max_len = int(lengths.max())

    input_ids = torch.stack([
        F.pad(ids, pad=(0, max_len - ids.shape[0]), mode="constant", value=0)
        for ids in input_ids
    ])
    labels = torch.stack([
        F.pad(item["labels"], pad=(0, max_len - item["labels"].shape[0]), mode="constant", value=IGNORE_INDEX)
        for item in items
    ])

    return {
        "input_ids": input_ids,     # (bs, max_len)
        "lengths": lengths,         # (bs, )
        "labels": labels            # (bs, max_len)
    }

class Vocab:
    def __init__(self, train_path: str):
        self.pad = "<pad>"
        self.unk = "<unk>"

        data = read_conll(train_path)
        all_words = set()
        all_tags = set()
        for words, tags in data:
            all_words.update(word.lower() for word in words)
            all_tags.update(tags)

        # 0 = <pad>, 1 = <unk>
        self.w2i = {self.pad: 0, self.unk: 1}
        for idx, word in enumerate(sorted(all_words), start=2):
            self.w2i[word] = idx
        self.i2w = {idx: word for word, idx in self.w2i.items()}

        self.l2i = {tag: idx for idx, tag in enumerate(sorted(all_tags))}
        self.i2l = {idx: tag for tag, idx in self.l2i.items()}

    @property
    def n_labels(self) -> int:
        return len(self.l2i)

    @property
    def len(self) -> int:
        return len(self.w2i)

    def encode_words(self, words: list[str]) -> torch.Tensor:
        unk_id = self.w2i[self.unk]
        return torch.tensor([self.w2i.get(w.lower(), unk_id) for w in words], dtype=torch.long)

    def encode_tags(self, tags: list[str]) -> torch.Tensor:
        return torch.tensor([self.l2i[t] for t in tags], dtype=torch.long)

class PhoNER(Dataset):
    def __init__(self, path: str, vocab: Vocab):
        super().__init__()
        self._data = read_conll(path)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> dict:
        words, tags = self._data[idx]
        return {
            "input_ids": self.vocab.encode_words(words),
            "labels": self.vocab.encode_tags(tags)
        }
