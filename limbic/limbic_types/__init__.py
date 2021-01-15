from typing import Any, Dict, List, NamedTuple, Optional, Set


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
    specific_params: Any


class BertBaseUncaseParams(NamedTuple):
    dropout_probability: float
    in_linear_features: int
    num_labels: int


class TrainBertParams(NamedTuple):
    emotions_train_file: str
    output_model_path: str
    bert_path: str
    device: str
    train_batch_size: int
    valid_batch_size: int
    epochs: int
    training_sample: int
    test_size: int
    weight_decay: float
    no_decay_components: List[str]
    bert_base_uncase_params: BertBaseUncaseParams
