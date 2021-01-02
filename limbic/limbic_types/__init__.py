from typing import Any, Dict, List, NamedTuple, Optional, Set

from keras_preprocessing.text import Tokenizer as KerasTokenizer


class Emotion(NamedTuple):
    category: str
    value: float
    term: str


class EmotionValue(NamedTuple):
    category: str
    value: float


class TimeEmotion(NamedTuple):
    seconds: int
    emotion: Emotion


class Lexicon(NamedTuple):
    emotion_mapping: Dict[str, List[Emotion]]
    categories: Set[str]


class ProcessParams(NamedTuple):
    lexicon: Lexicon
    terms_mapping: Optional[Dict[str, str]]


class ModelParams(NamedTuple):
    model: Any
    tokenizer: Any
    max_len: int
    emotions: List[str]
