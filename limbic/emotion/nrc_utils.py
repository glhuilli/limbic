from collections import defaultdict
from typing import Dict, List

from limbic.emotion.utils import load_lexicon
from limbic.limbic_types import Emotion, Lexicon

# This application/product/tool makes use of the NRC Word-Emotion Association Lexicon,
# created by Saif M. Mohammad and Peter D. Turney at the National Research Council Canada,
# the NRC Valence, Arousal, Dominance Lexicon, created by Saif M. Mohammad
# at the National Research Council Canada, and the NRC Affect Intensity Lexicon,
# created by Saif M. Mohammad at the National Research Council Canada.

# You can check more details and get the emotion lexicon files in the following links
# http://saifmohammad.com/WebPages/AccessResource.htm
# and http://saifmohammad.com/WebPages/lexicons.html


class NotValidNRCLexiconException(Exception):
    pass


def load_nrc_lexicon(lexicon_path: str, lexicon_type: str) -> Lexicon:
    if lexicon_type == 'emotion':
        return _load_nrc_emotion(lexicon_path)
    if lexicon_type == 'affect_intensity':
        return _load_nrc_affect_intensity(lexicon_path)
    if lexicon_type == 'vad':
        return _load_nrc_vad(lexicon_path)
    raise NotValidNRCLexiconException


def _load_nrc_emotion(lexicon_file_path) -> Lexicon:
    return load_lexicon(lexicon_file_path)


def _load_nrc_affect_intensity(lexicon_file_path: str) -> Lexicon:
    """
    As described in https://saifmohammad.com/WebPages/AffectIntensity.htm
        The lexicon has close to 6,000 entries for four basic emotions: anger, fear, joy, and sadness.
    """
    data: Dict[str, List[Emotion]] = defaultdict(list)
    categories = set()
    skip = True  # Needed to skip a new disclaimer added to the lexicon file by the NRC.
    with open(lexicon_file_path, 'r') as intensity_file:
        for line in intensity_file.readlines():
            if skip:
                if line == 'term	score	AffectDimension\n':
                    skip = False
                continue
            term, score, affect_dimension = line.strip().split('\t')
            data[term].append(Emotion(value=float(score), category=affect_dimension, term=term))
            categories.add(affect_dimension)
    return Lexicon(emotion_mapping=data, categories=categories)


def _load_nrc_vad(lexicon_file_path) -> Lexicon:
    """
    As described in https://saifmohammad.com/WebPages/nrc-vad.html
        valence is the positive--negative or pleasure--displeasure dimension;
        arousal is the excited--calm or active--passive dimension; and
        dominance is the powerful--weak or 'have full control'--'have no control' dimension.
    """
    data: Dict[str, List[Emotion]] = defaultdict(list)
    with open(lexicon_file_path, 'r') as vad_file:
        for idx, line in enumerate(vad_file.readlines()):
            if idx > 0:
                term, valence, arousal, dominance = line.strip().split('\t')
                data[term].append(Emotion(value=float(valence), category='valence', term=term))
                data[term].append(Emotion(value=float(arousal), category='arousal', term=term))
                data[term].append(Emotion(value=float(dominance), category='dominance', term=term))
    return Lexicon(emotion_mapping=data, categories={'valence', 'arousal', 'dominance'})
