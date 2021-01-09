from typing import Any, Optional, Tuple

import torch
import transformers

from limbic.emotion.models.bert_limbic_model.bert_base_uncased import BERTBaseUncased
from limbic.emotion.models.limbic_model import LimbicModel
from limbic.limbic_types import ModelParams

_VERSION = '2021-01-03'
_MAX_LEN = 64


class BertLimbicModel(LimbicModel):
    """
    Note that this is just an interface to use the model as input.
    The actual model and it's training details are available in emotion/models/tf_limbic_model/utils.py

    This model is based in Hugging Face pre-trained BERT base uncased.
    You can download the BERT files for python for free in https://huggingface.co/bert-base-uncased

    Files used in this package:
     - config.json
     - vocab.txt
     - pytorch_model.bin
    """
    def __init__(self,
                 model_path,
                 bert_path,
                 model_params: Optional[ModelParams] = None,
                 device='cpu') -> None:
        LimbicModel.__init__(self, model_params)
        self.bert_path = bert_path
        self.model_path = model_path
        self.device = device

    def get_max_len(self) -> int:
        return self.max_len or _MAX_LEN

    def load_model(self) -> Tuple[Any, Any]:
        model = BERTBaseUncased(self.bert_path)
        model.load_state_dict(torch.load(self.model_path))
        model.to(self.device)
        tokenizer = transformers.BertTokenizer.from_pretrained(self.bert_path, do_lower_case=True)
        return model.eval(), tokenizer

    def sentence_prediction(self, text: str):  # TODO: add typing
        text = " ".join(str(text).split())  # TODO: add edge cases

        inputs = self.tokenizer.encode_plus(text,
                                            None,
                                            add_special_tokens=True,
                                            max_length=self.get_max_len())

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

        ids = ids.to(self.device, dtype=torch.long)
        token_type_ids = token_type_ids.to(self.device, dtype=torch.long)
        mask = mask.to(self.device, dtype=torch.long)

        return self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)

    def predict(self, sentence: str):  # TODO: add typing
        return self.sentence_prediction(sentence).detach().numpy()[0]
