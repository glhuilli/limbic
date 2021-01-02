import transformers
import torch.nn as nn

# TODO: Load these parameters from a config
_BERT_PATH = '../data/bert'
_NUM_LABELS = 4


class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(_BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, _NUM_LABELS)

    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        bo = self.bert_drop(o2)

        # TODO: review if this is the best option for evaluating the Softmax
        output = nn.functional.softmax(self.out(bo), dim=1)
        return output
