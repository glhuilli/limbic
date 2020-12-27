from typing import List, Optional

import numpy as np
import tensorflow as tf

from limbic.emotion.models.tf_limbic_model.utils import load_model
from limbic.limbic_constants import AFFECT_INTENSITY_EMOTIONS
from limbic.limbic_types import EmotionValue, TfModelParams

_VERSION = '2019-11-16'
# TODO: Move _MAX_LEN and _EMOTIONS parameter to a config file associated to _VERSION
_MAX_LEN = 150


class TfLimbicModel:
    """
    Note that this is just an interface to use the model as input.
    The actual model and it's training details are available in emotion/models/tf_limbic_model/utils.py
    """
    def __init__(self, model_params: Optional[TfModelParams] = None) -> None:
        if model_params:
            self.tokenizer = model_params.tokenizer
            self.model = model_params.model
            self.max_len = model_params.max_len
            self.emotions = model_params.emotions
        else:
            self.model, self.tokenizer = load_model(_VERSION)
            self.max_len = _MAX_LEN

            # For the moment, this model will assume it was trained for Affect Intensity Emotions.
            # TODO: Make TfLimbicModel emotion type agnostic by refactoring it into a new TfTrainingParams namedtuple
            self.emotions = AFFECT_INTENSITY_EMOTIONS

    def _process_input(self, sentence: str):
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

    def get_sentence_emotions(self, sentence: str) -> List[EmotionValue]:
        """
        Predicts emotions found in a sentence using a pre-loaded TensorFlow model and its parameters.

        The only consideration that the model needs is that the sentence pre-processing needs to be compatible
        with the one implemented here.
        TODO: refactor so you can pass a specific pre-processing step in case it's needed.

        Note that the output is not associated to a specific term in the sentence, but to the overall set of words used.
        """
        prediction = self.predict(sentence)
        return [EmotionValue(category=k, value=v) for (k, v) in zip(self.emotions, prediction)]
