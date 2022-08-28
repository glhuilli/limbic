from typing import Dict, List, Optional

from limbic.limbic_types import Emotion, Lexicon, ProcessParams
from limbic.utils.text import process_content


class LexiconLimbicEsModel:

    def __init__(self, lexicon: Lexicon, terms_mapping: Optional[Dict[str, str]] = None) -> None:
        self.lexicon = lexicon
        self.terms_mapping = terms_mapping or {}
        self.process_params = ProcessParams(lexicon=self.lexicon, terms_mapping=terms_mapping)

    def get_term_emotions(self, term: str) -> List[Emotion]:
        """
        Compute the list of emotions for a given input term.

        TODO: build negation strategy for spanish language (similar to lexicon_limbic_model)
        """
        term = term.lower().strip()
        emotions: List[Emotion] = self.process_params.lexicon.emotion_mapping.get(term, [])
        return emotions

    def get_sentence_emotions(self, sentence: str) -> List[Emotion]:
        """
        Get list of all emotions in a given sentence.

        TODO: build negation strategy for spanish language (similar to lexicon_limbic_model)
        """
        sentence_emotions = []
        for term in process_content(sentence, self.process_params.terms_mapping, 'es'):
            for emotion in self.get_term_emotions(term):
                sentence_emotions.append(emotion)
        return sentence_emotions
