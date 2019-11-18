from typing import Dict, List, Optional

from limbic.utils.nlp import get_negated_words
from limbic.utils.text import process_content
from limbic.limbic_types import Emotion, Lexicon, ProcessParams
from limbic.emotion.plutchik_wheel import PLUTCHIK_EMOTIONS_OPPOSITE_MAPPING


class LexiconLimbicModel:

    def __init__(self, lexicon: Lexicon, terms_mapping: Optional[Dict[str, str]] = None) -> None:
        self.lexicon = lexicon
        self.terms_mapping = terms_mapping or {}
        self.process_params = ProcessParams(lexicon=self.lexicon, terms_mapping=terms_mapping)

    def get_term_emotions(self, term: str, is_negated: bool = False) -> List[Emotion]:
        """
        Compute the list of emotions for a given input term.

        In case the term is negated, then the emotions are converted according to
        Plutchik's emotions wheel.

        for example:
        - not joy = sadness
        - not anger = fear

        Note that negated terms will be denoted as "-term" in the output Emotion. Also, if the
        negated emotion is not in Plutchik's emotion wheel, the negated term will be skipped.
        """
        term = term.lower().strip()
        emotions: List[Emotion] = self.process_params.lexicon.emotion_mapping.get(term, [])
        if is_negated:
            negated_emotions = []
            for emotion in emotions:
                if emotion.category in PLUTCHIK_EMOTIONS_OPPOSITE_MAPPING:
                    negated_emotions.append(
                        Emotion(category=PLUTCHIK_EMOTIONS_OPPOSITE_MAPPING[emotion.category],
                                value=emotion.value, term=f'-{term}'))
            return negated_emotions
        return emotions

    def get_sentence_emotions(self, sentence: str) -> List[Emotion]:
        """
        Get list of all emotions in a given sentence.
        """
        sentence_emotions = []
        negated_terms = get_negated_words(sentence)
        for term in process_content(sentence, self.process_params.terms_mapping):
            for emotion in self.get_term_emotions(term, is_negated=term in negated_terms):
                sentence_emotions.append(emotion)
        return sentence_emotions
