"""
 This model is based in Hugging Face pre-trained BERT base uncased.
 You can download the BERT files for python for free in https://huggingface.co/bert-base-uncased

 Files used in this package:
  - config.json
  - vocab.txt
  - pytorch_model.bin
"""
from typing import Any, Optional, Tuple
import transformers

import torch

from limbic.emotion.models.limbic_model import LimbicModel
from limbic.emotion.models.bert_limbic_model.bert_base_uncased import BERTBaseUncased
from limbic.limbic_types import ModelParams


_VERSION = '2021-01-01'
# TODO: Move _MAX_LEN and _EMOTIONS parameter to a config file associated to _VERSION
_MAX_LEN = 64
_DEVICE = 'cpu'
_MODEL_PATH = '../bert/path'
_BERT_PATH = ''
_TOKENIZER = transformers.BertTokenizer.from_pretrained(_BERT_PATH, do_lower_case=True)


class BertLimbicModel(LimbicModel):
    """
    Note that this is just an interface to use the model as input.
    The actual model and it's training details are available in emotion/models/tf_limbic_model/utils.py
    """
    def __init__(self, model_params: Optional[ModelParams] = None) -> None:
        LimbicModel.__init__(self, model_params)

    def get_max_len(self) -> int:
        return _MAX_LEN

    def load_model(self) -> Tuple[Any, Any]:
        model = BERTBaseUncased()
        model.load_state_dict(torch.load(_MODEL_PATH))
        model.to(_DEVICE)
        return None, model.eval()

    def sentence_prediction(self, sentence):
        # tokenizer = TOKENIZER
        # max_len = MAX_LEN
        review = str(sentence)
        review = " ".join(review.split())

        inputs = self.tokenizer.encode_plus(
            review, None, add_special_tokens=True, max_length=self.max_len)

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        padding_length = self.max_len - len(ids)
        ids = ids + ([0] * padding_length)
        mask = mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
        mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
        token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

        ids = ids.to(_DEVICE, dtype=torch.long)
        token_type_ids = token_type_ids.to(_DEVICE, dtype=torch.long)
        mask = mask.to(_DEVICE, dtype=torch.long)

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        return outputs

    def predict(self, sentence: str):  # TODO: add typing
        return self.sentence_prediction(sentence).detach().numpy()[0]
