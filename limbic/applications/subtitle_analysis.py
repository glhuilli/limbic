import json
import os
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import srt
from srt import Subtitle

from limbic.emotion.models import LexiconLimbicModel
from limbic.limbic_types import Emotion, TimeEmotion
from limbic.utils.text import process_content

if 'ipykernel' in sys.modules:  # only use tqdm when inside a Jupyter notebook
    from tqdm import tqdm_notebook as tqdm
else:
    tqdm = lambda x: x


class SubtitleFileNotValidException(Exception):
    pass


class EmotionsFileNotAvailableException(Exception):
    pass


def load_subtitles(srt_file_path) -> List[Any]:
    """
    Given a subtitles file path, returns a list of subtitles using
    the srt python package.

    Note that encoding 'utf-8-sig' ignoring errors is necessary to load most
    of the subtitle files available online.
    """
    with open(srt_file_path, 'r', encoding='utf-8-sig', errors='ignore') as srt_file:
        return list(srt.parse(srt_file.read()))


def get_subtitles_emotions(subtitles: List[Subtitle],
                           limbic_model: LexiconLimbicModel) -> Iterable[TimeEmotion]:
    """
    Get emotions per second using TimeEmotion named tuple.
    Each term per subtitle line is processed, and all emotions within a term
    is listed independently.

    Note that a given emotion will be mapped to the start of the subtitle time.
    """
    for sub in subtitles:
        for emotion in limbic_model.get_sentence_emotions(sub.content):
            yield TimeEmotion(seconds=sub.start.seconds, emotion=emotion)


def get_emotions_for_series_folder(
        folder_path: str,
        limbic_model: LexiconLimbicModel,
        output_file_path: Optional[str] = None,
        fast: Optional[bool] = True) -> Dict[int, Dict[int, List[TimeEmotion]]]:
    """
    Given a folder_path with subtitles, compute all emotions and saves in output_file_path
    when given as input.

    If fast flag is set to false then every sentence will be analyzed using th NLP
    layer (which is slow), but some extra negations might be identified.
    """
    if fast:
        return _get_emotions_for_series_folder_fast(folder_path, limbic_model, output_file_path)
    return _get_emotions_for_series_folder(folder_path, limbic_model, output_file_path)


def _get_emotions_for_series_folder(
        folder_path: str,
        limbic_model: LexiconLimbicModel,
        output_file_path: Optional[str] = None) -> Dict[int, Dict[int, List[TimeEmotion]]]:
    """
    Iterate over all sentences in subtitles files in a folder and compute the emotions.
    """
    seasons_episodes_subtitles_emotions: Dict[int, Dict[int, List[TimeEmotion]]] = defaultdict(
        lambda: defaultdict(list))
    for srt_file in tqdm(os.listdir(folder_path)):
        srt_file_path = os.path.join(folder_path, srt_file)
        subtitles = load_subtitles(srt_file_path)
        season, episode = _get_season_episode(srt_file)
        seasons_episodes_subtitles_emotions[season][episode] = list(
            get_subtitles_emotions(subtitles, limbic_model))
    if output_file_path:
        _save_seasons_emotions_to_file(seasons_episodes_subtitles_emotions, output_file_path)
    return seasons_episodes_subtitles_emotions


def _get_emotions_for_series_folder_fast(
        folder_path: str,
        limbic_model: LexiconLimbicModel,
        output_file_path: Optional[str] = None) -> Dict[int, Dict[int, List[TimeEmotion]]]:
    """
    Given a folder (assuming all files have S*E* pattern), computes the
    emotions for each episode of each season

    If parameter "fast" is given as input, emotions will be computed with a
    proxy to identify negation sentences
    which might not guarantee to catch all linguistic negations. (e.g. "this is far from good").
    """
    seasons_episodes_subtitles_emotions: Dict[int, Dict[int, List[TimeEmotion]]] = defaultdict(
        lambda: defaultdict(list))
    season_episode_time_words: Dict[int, Dict[int, Dict[int, Set[Optional[str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set)))
    season_episode_time_sentence: Dict[int, Dict[int, Dict[int, List[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list)))
    subtitle_words = set()
    for srt_file in os.listdir(folder_path):
        subtitles = load_subtitles(os.path.join(folder_path, srt_file))
        season, episode = _get_season_episode(srt_file)
        for sub in subtitles:
            if _contains_negation(sub.content):
                season_episode_time_sentence[season][episode][sub.start.seconds].append(sub.content)
                season_episode_time_words[season][episode][sub.start.seconds].add(None)
            else:
                for term in process_content(sub.content, limbic_model.terms_mapping):
                    subtitle_words.add(term)
                    season_episode_time_words[season][episode][sub.start.seconds].add(term)

    _update_seasons_emotions_data(season_episode_time_words, seasons_episodes_subtitles_emotions,
                                  season_episode_time_sentence, subtitle_words, limbic_model)
    if output_file_path:
        _save_seasons_emotions_to_file(seasons_episodes_subtitles_emotions, output_file_path)
    return seasons_episodes_subtitles_emotions


def _save_seasons_emotions_to_file(seasons_episodes_subtitles_emotions: Dict[int, Dict[
    int, List[TimeEmotion]]], output_file_path: str) -> None:
    """
    Save dictionary seasons_episodes_subtitles_emotions to file
    """
    with open(output_file_path, 'w') as output_file:
        json.dump(_season_episode_emotions_to_dict(seasons_episodes_subtitles_emotions),
                  output_file)


def _update_seasons_emotions_data(season_episode_time_words: Dict[int, Dict[int, Dict[
    int, Set[Optional[str]]]]], seasons_episodes_subtitles_emotions: Dict[int,
                                                                          Dict[int,
                                                                               List[TimeEmotion]]],
                                  season_episode_time_sentence: Dict[int, Dict[int,
                                                                               Dict[int,
                                                                                    List[str]]]],
                                  sub_words: Set[str], limbic_model: LexiconLimbicModel) -> None:
    """
    Given the data in season_episode_time_words and season_episode_time_sentence
    updates the seasons_episodes_subtitles_emotions dictionary.

    1. get all emotions for every independent subtitle word (in sub_words)
    2. add emotions for words indexed by season, episode and time
    3. add emotions for sentences indexed by season, episode and time
    4. sort emotions by time

    Note that sentences in this case are selected by having a negation, in which case
    we want to compute the emotions using the NLP layer.

    TODO: Maybe reduce complexity of this method adding sub methods.
    """
    words_emotions = {word: limbic_model.get_term_emotions(word) for word in sub_words}
    for season, season_data in tqdm(season_episode_time_words.items(), 'processing sentences'):
        for episode, episode_data in season_data.items():
            for seconds, words in episode_data.items():
                # first we add the emotions from the set of words
                index = (season, episode, seconds)
                _add_emotions_from_words(words, words_emotions, seasons_episodes_subtitles_emotions,
                                         index)
                # then we add the emotions from the NLP layer
                _add_emotions_from_sentence(limbic_model, seasons_episodes_subtitles_emotions,
                                            season_episode_time_sentence, index)
    for season, season_data in tqdm(season_episode_time_words.items(), 'sorting emotions by time'):
        for episode in season_data.keys():
            sorted_emotions = sorted(seasons_episodes_subtitles_emotions[season][episode],
                                     key=lambda x: x.seconds)
            seasons_episodes_subtitles_emotions[season][episode] = sorted_emotions


def _add_emotions_from_words(words: Set[Optional[str]], words_emotions: Dict[str, List[Emotion]],
                             seasons_episodes_subtitles_emotions: Dict[int,
                                                                       Dict[int,
                                                                            List[TimeEmotion]]],
                             index: Tuple[int, int, int]) -> None:
    season, episode, seconds = index
    for word in [x for x in words if x]:
        for emotion in words_emotions[word]:
            seasons_episodes_subtitles_emotions[season][episode].append(
                TimeEmotion(seconds=seconds, emotion=emotion))


def _add_emotions_from_sentence(limbic_model: LexiconLimbicModel,
                                seasons_episodes_subtitles_emotions: Dict[int,
                                                                          Dict[int,
                                                                               List[TimeEmotion]]],
                                season_episode_time_sentence: Dict[int, Dict[int, Dict[int,
                                                                                       List[str]]]],
                                index: Tuple[int, int, int]) -> None:
    season, episode, seconds = index
    for sentence in season_episode_time_sentence[season][episode][seconds]:
        for emotion in limbic_model.get_sentence_emotions(sentence):
            seasons_episodes_subtitles_emotions[season][episode].append(
                TimeEmotion(seconds=seconds, emotion=emotion))


def _get_season_episode(file_name: str) -> Tuple[int, int]:
    """
    Gets season and episode number assuming pattern S*E* is in the file name.
    """
    match = re.search(r's\d*e\d*', file_name, re.I)
    if match:
        span = match.span()
        season, episode = tuple(
            map(int, re.split('e', file_name[span[0] + 1:span[1]], flags=re.IGNORECASE)))
        return season, episode  # decoupled from previous line of code to satisfy mypy
    raise SubtitleFileNotValidException


def _contains_negation(sentence: str) -> bool:
    """
    Soft strategy to verify if an input sentence has a negation.
    TODO: verify if some other cases could be missing
    """
    return re.search(r'\bnot\b|n\'t\b|\bno\b', sentence, flags=re.IGNORECASE) is not None


def _season_episode_emotions_to_dict(
        season_episode_emotions: Dict[int, Dict[int, List[TimeEmotion]]]) -> Dict[int, Any]:
    """
    Serialization method for season - episode TimeEmotions
    """
    output: Dict[int, Any] = defaultdict(lambda: defaultdict(list))
    for season, season_data in season_episode_emotions.items():
        for episode, episode_data in season_data.items():
            for emotion in episode_data:
                output[season][episode].append(_time_emotion_to_dict(emotion))
    return dict(output)


def _time_emotion_to_dict(time_emotion: TimeEmotion) -> Dict[str, Any]:
    """
    Serialization method for TimeEmotion
    TODO: move this to TimeEmotion (maybe create a class with tojson method)
    """
    return {
        'seconds': time_emotion.seconds,
        'category': time_emotion.emotion.category,
        'value': time_emotion.emotion.value,
        'term': time_emotion.emotion.term
    }


def load_emotions_for_series(
        emotions_data_file_path: str) -> Dict[int, Dict[int, List[TimeEmotion]]]:
    """
    Load existing file with emotions for series (indexed by season and episode).

    File needs to be in Json format with the following a specific Json schema
    TODO: validate json schema of input file
    """
    if os.path.isfile(emotions_data_file_path):
        with open(emotions_data_file_path, 'r') as emotions_data_file:
            season_episode_emotions_dict = json.load(emotions_data_file)
            return _season_episode_emotions_from_dict(season_episode_emotions_dict)
    raise EmotionsFileNotAvailableException


def _season_episode_emotions_from_dict(
        season_episode_emotions_dict: Dict[str, Any]) -> Dict[int, Dict[int, List[TimeEmotion]]]:
    """
    Utility method used to generate TimeEmotion from a dictionary with TimeEmotion data for a
    series with season and episodes.
    """
    output: Dict[int, Dict[int, List[TimeEmotion]]] = defaultdict(lambda: defaultdict(list))
    for season, season_data in season_episode_emotions_dict.items():
        for episode, episode_data in season_data.items():
            for emotion in episode_data:
                output[int(season)][int(episode)].append(
                    TimeEmotion(seconds=emotion.get('seconds'),
                                emotion=Emotion(category=emotion.get('category'),
                                                value=emotion.get('value'),
                                                term=emotion.get('term'))))
    return output
