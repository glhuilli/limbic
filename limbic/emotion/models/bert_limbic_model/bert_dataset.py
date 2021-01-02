import torch
import transformers

_BERT_PATH = '../data/bert'
_TOKENIZER = transformers.BertTokenizer.from_pretrained(_BERT_PATH, do_lower_case=True)
_MAX_LEN = 64


class BERTDataset:
    def __init__(self, text, target):
        self.text = text
        self.target = target
        self.tokenizer = _TOKENIZER
        self.max_len = _MAX_LEN

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = str(self.text[item])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.target[item], dtype=torch.float),
        }
