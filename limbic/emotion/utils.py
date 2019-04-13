from collections import defaultdict
from typing import Dict, List

from limbic.limbic_types import Emotion, Lexicon


def load_lexicon(lexicon_path: str) -> Lexicon:
    """
    Generic lexicon file needs to be a CSV file with the following structure:
        term, emotion, score
    """
    data: Dict[str, List[Emotion]] = defaultdict(list)
    categories = set()
    with open(lexicon_path, 'r') as emotion_file:
        for line in emotion_file.readlines():
            if line.strip():
                term, emotion, score = line.strip().split('\t')
                data[term].append(Emotion(value=float(score), category=emotion, term=term))
                categories.add(emotion)
    return Lexicon(emotion_mapping=data, categories=categories)
