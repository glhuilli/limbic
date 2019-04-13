from typing import Dict, List, NamedTuple, Optional, Set


class Emotion(NamedTuple):
    category: str
    value: float
    term: str


class TimeEmotion(NamedTuple):
    seconds: int
    emotion: Emotion


class Lexicon(NamedTuple):
    emotion_mapping: Dict[str, List[Emotion]]
    categories: Set[str]


class ProcessParams(NamedTuple):
    lexicon: Lexicon
    terms_mapping: Optional[Dict[str, str]]
