from typing import List
from typing import Any, Optional, Tuple
from abc import ABC, abstractmethod

from limbic.limbic_types import EmotionValue
from limbic.limbic_constants import AFFECT_INTENSITY_EMOTIONS as EMOTIONS
from limbic.limbic_types import ModelParams


_MAX_LEN = 150


class LimbicModel(ABC):
    def __init__(self, model_params: Optional[ModelParams]):
        if model_params:
            self.tokenizer = model_params.tokenizer
            self.model = model_params.model
            self.max_len = model_params.max_len
            self.emotions = model_params.emotions
        else:
            self.model, self.tokenizer = self.load_model()
            self.max_len = self.get_max_len()

        # For the moment, all LimbicModelS will assume it was trained for Affect Intensity Emotions.
        self.emotions = EMOTIONS

    @abstractmethod
    def get_max_len(self) -> int:
        pass

    @abstractmethod
    def load_model(self) -> Tuple[Any, Any]:
        pass

    @abstractmethod
    def predict(self, sentence: str):  # TODO: Add typing
        pass

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
