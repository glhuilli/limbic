from typing import Optional

import torch.nn as nn
import transformers

from limbic.limbic_types import BertBaseUncaseParams

_DEFAULT_DROPOUT = 0.3
_DEFAULT_IN_LINEAR_FEATURES = 768
_DEFAULT_NUM_LABELS = 4


class BERTBaseUncased(nn.Module):

    def __init__(self, bert_path: str, bert_params: Optional[BertBaseUncaseParams]):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        if bert_params:
            self.bert_drop = nn.Dropout(bert_params.dropout_probability)
            self.out = nn.Linear(bert_params.in_linear_features, bert_params.num_labels)
        else:
            self.bert_drop = nn.Dropout(_DEFAULT_DROPOUT)
            self.out = nn.Linear(_DEFAULT_IN_LINEAR_FEATURES, _DEFAULT_NUM_LABELS)

    def forward(self, ids, mask, token_type_ids):
        _, bert_output = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)
        drop_output = self.bert_drop(bert_output)

        # TODO: review if this is the best option for evaluating the Softmax
        return nn.functional.softmax(self.out(drop_output), dim=1)
