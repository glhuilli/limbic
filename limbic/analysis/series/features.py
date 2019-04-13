from collections import defaultdict
from typing import Any, Dict, List

import pandas as pd

from limbic.limbic_types import TimeEmotion
from limbic.analysis import get_max_delta, get_mean, get_total, moving_window


def all_moving_windows(seasons_episodes_subtitles_emotions, categories: List[str],
                       window: int = 2) -> pd.DataFrame:
    data: Dict[str, List[Any]] = {
        'episode': [],
        'total': [],
        'time': [],
        'category': [],
        'season': []
    }
    for season in seasons_episodes_subtitles_emotions.keys():
        for category in categories:
            for episode, emotions in seasons_episodes_subtitles_emotions[season].items():
                episode_name = f'S0{season}E{episode}' if season < 10 else f'S{season}E{episode}'
                mw = moving_window(emotions, category)
                for idx, v in enumerate(mw):
                    data['total'].append(v)
                    data['episode'].append(episode_name)
                    data['category'].append(category)
                    data['time'].append(idx * window)
                    data['season'].append(season)
    return pd.DataFrame.from_dict(data)


def get_features(seasons_episodes_subtitles_emotions,
                 imdb_data,
                 categories: List[str],
                 min_threshold: float = 8.0,
                 max_threshold: float = 9.0) -> pd.DataFrame:
    seasons_episodes_data: Dict[str, Any] = defaultdict(list)
    for season, season_episodes in seasons_episodes_subtitles_emotions.items():
        for episode, episode_subtitle_emotions in season_episodes.items():
            features = _get_emotions_features(episode_subtitle_emotions, season, episode,
                                              categories)
            ratings = _get_rating(imdb_data, season, episode)
            votes = _get_votes(imdb_data, season, episode)
            _add_data(seasons_episodes_data, features, ratings, votes)
    df = pd.DataFrame.from_dict(seasons_episodes_data)
    df['rating_category'] = df.apply(
        lambda row: _rating_to_category(row['ratings'], min_threshold, max_threshold), axis=1)
    return df


def _add_data(episodes_data: Dict[str, Any], features: Dict[str, float], ratings, votes):
    for k, v in features.items():
        episodes_data[k].append(v)
    episodes_data['ratings'].append(ratings)
    episodes_data['votes'].append(votes)


def _get_rating(imdb_data, season, episode):
    return float(imdb_data[str(season)][episode - 1]['imdbRating'])


def _get_votes(imdb_data, season, episode):
    return int(imdb_data[str(season)][episode - 1]['imdbVotes'])


def _rating_to_category(rating: float, min_threshold: float, max_threshold: float) -> str:
    """
    TODO: compute automatically the min/max thresholds using quartile values.
    """
    if rating > max_threshold:
        return 'high'
    if min_threshold < rating <= max_threshold:
        return 'mid'
    return 'low'


def _get_emotions_features(subtitles_emotions: List[TimeEmotion], season: int, episode: int,
                           categories: List[str]) -> Dict[str, float]:
    # TODO: refactor so it uses the categories in the lexicon
    data: Dict[str, Any] = {'season': season, 'episode': episode}
    for e in categories:
        data[f'total_{e}'] = get_total(subtitles_emotions, e)
        data[f'avg_{e}'] = get_mean(subtitles_emotions, e)
        max_delta, time_max_delta = get_max_delta(subtitles_emotions, e)
        data[f'max_delta_{e}'] = max_delta
        data[f'time_max_delta_{e}'] = time_max_delta
    data['dir_joy'] = get_total(subtitles_emotions, 'joy') - get_total(
        subtitles_emotions, 'sadness')
    data['dir_fear'] = get_total(subtitles_emotions, 'fear') - get_total(
        subtitles_emotions, 'anger')
    return data
