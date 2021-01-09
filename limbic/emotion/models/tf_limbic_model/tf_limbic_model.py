from typing import Any, Optional, Tuple

import numpy as np
import tensorflow as tf

from limbic.emotion.models.limbic_model import LimbicModel
from limbic.emotion.models.tf_limbic_model.utils import load_model as utils_load_model
from limbic.limbic_types import ModelParams

_VERSION = '2019-11-16'
# TODO: Move _MAX_LEN and _EMOTIONS parameter to a config file associated to _VERSION
_MAX_LEN = 150


class TfLimbicModel(LimbicModel):
    """
    Note that this is just an interface to use the model as input.
    The actual model and it's training details are available in emotion/models/tf_limbic_model/utils.py
    """
    def __init__(self, model_params: Optional[ModelParams] = None) -> None:
        LimbicModel.__init__(self, model_params)

    def get_max_len(self) -> int:
        return _MAX_LEN

    def load_model(self) -> Tuple[Any, Any]:
        return utils_load_model(_VERSION)

    def _process_input(self, sentence: str):  # TODO: Add typing
        tokenized_sentence = self.tokenizer.texts_to_sequences([sentence])
        padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokenized_sentence,
                                                                      maxlen=self.max_len)[0]
        return np.expand_dims(padded_tokens, 0)

    def predict(self, sentence: str):
        """
        Predicts emotions found in a sentence using a pre-loaded TensorFlow model and its parameters.

        The only consideration that the model needs is that the sentence pre-processing needs to be compatible
        with the one implemented here.
        TODO: refactor so you can pass a specific pre-processing step in case it's needed.
        """
        return self.model.predict(self._process_input(sentence))[0]
