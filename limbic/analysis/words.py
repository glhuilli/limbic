from collections import Counter, defaultdict
from random import shuffle
from typing import Counter as CounterType, Dict, List, Optional, Set

import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
from wordcloud import WordCloud

from limbic.limbic_types import TimeEmotion


def plot_emotions_wordclouds(emotions: List[TimeEmotion],
                             categories: List[str],
                             unique: bool = False,
                             weighted: bool = False) -> None:
    fig = plt.figure()
    unique_terms = _unique_terms_for_categories(emotions, categories) if unique else {}
    for idx, emotion in enumerate(categories):
        if unique:
            title_prefix = 'Unique weighted word'
            emotion_terms = _weighted_emotions_terms(emotions, emotion, unique_terms[emotion])
        else:
            if weighted:
                title_prefix = 'Weighted word'
                emotion_terms = _weighted_emotions_terms(emotions, emotion)
            else:
                title_prefix = 'Word'
                emotion_terms = ' '.join(
                    [x.emotion.term for x in emotions if x.emotion.category == emotion])
        ax = fig.add_subplot(2, 2, idx + 1)
        ax.set_title(f'{title_prefix} cloud for "{emotion}"', fontsize=16)
        fig.set_figheight(10)
        fig.set_figwidth(18)
        word_cloud = WordCloud(
            background_color='white', max_words=200, max_font_size=40, scale=3,
            random_state=42).generate(emotion_terms)
        ax.imshow(word_cloud)
        ax.axis('off')
    plt.show()


def _weighted_emotions_terms(emotions: List[TimeEmotion], category: str,
                             unique_terms: Optional[Set[str]] = None) -> str:
    porter = PorterStemmer()
    total_emotion_per_term: Dict[str, float] = defaultdict(float)
    for e in emotions:
        if '-' in e.emotion.term:
            continue
        term_stem = porter.stem(e.emotion.term)
        if unique_terms:
            if term_stem not in unique_terms:
                continue
            if e.emotion.category == category:
                total_emotion_per_term[term_stem] += e.emotion.value
        else:
            if e.emotion.category == category:
                total_emotion_per_term[term_stem] += e.emotion.value
    weighted_terms = []
    for term, total_emotion in total_emotion_per_term.items():
        for _ in range(0, int(total_emotion)):
            weighted_terms.append(term)
    shuffle(weighted_terms)
    return ' '.join([x.strip() for x in weighted_terms])


def _unique_terms_for_categories(emotions: List[TimeEmotion],
                                 categories: List[str]) -> Dict[str, Set[str]]:
    porter = PorterStemmer()
    emotion_terms: Dict[str, Set[str]] = defaultdict(set)
    for category in categories:
        for e in emotions:
            if e.emotion.category == category and '-' not in e.emotion.term:
                emotion_terms[category].add(porter.stem(e.emotion.term))
    all_terms_c: CounterType[str] = Counter()
    for terms in emotion_terms.values():
        all_terms_c.update(terms)
    repeated_terms = {x for x, y in all_terms_c.items() if y > 1}
    emotion_unique_term_stems: Dict[str, Set[str]] = defaultdict(set)
    for category in categories:
        for e in emotions:
            if e.emotion.category == category and '-' not in e.emotion.term:
                term_stem = porter.stem(e.emotion.term)
                if term_stem not in repeated_terms:
                    emotion_unique_term_stems[category].add(term_stem)
    return emotion_unique_term_stems
