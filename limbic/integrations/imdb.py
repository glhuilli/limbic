import json
import os
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

_OMDB_API_KEY = os.environ['OMDB_API_KEY']
_OMDB_API_BASE_URL = 'http://www.omdbapi.com/'
_OMDB_API_WAIT_TIME = 0.3
_IMDB_BASE_URL = 'https://www.imdb.com/title/'


def get_imdb_data(show_imdb_id: str, num_seasons: int,
                  output_path: Optional[str]) -> Dict[int, List[Any]]:
    """
    Given show_imdb_id and the number of seasons, get all metadata from IMDB, including
    ratings, viewers, etc..
    """
    seasons_episodes = _get_episodes_imdb_ids(show_imdb_id, num_seasons)
    seasons_data = _get_imdb_data(seasons_episodes)
    if output_path:
        with open(output_path, 'w') as imdb_file:
            json.dump(seasons_data, imdb_file)
    return seasons_data


def _get_episodes_imdb_ids(show_imdb_id: str, seasons: int) -> Dict[int, List[str]]:
    """
    Given a show imdb_id, gets all the IDs for the episodes of the show
    """
    base_url = f'{_IMDB_BASE_URL}{show_imdb_id}/'
    seasons_episodes: Dict[int, List[str]] = defaultdict(list)
    for i in tqdm(range(1, seasons + 1)):
        season_url = base_url + f'episodes?season={i}'
        soup = BeautifulSoup(requests.get(season_url).content, 'lxml')
        for element in soup.find_all('a'):
            if 'ttep' in element['href'] and 'ttep_ep_tt' not in element['href'] and element.get(
                    'itemprop') == 'url':
                seasons_episodes[i].append(element['href'].split('/')[2])
    return seasons_episodes


def _get_imdb_data(seasons_episodes: Dict[int, List[str]]) -> Dict[int, List[Any]]:
    """
    Using the OMDB API, get all episodes information given a
    """
    seasons_data: Dict[int, List[Any]] = defaultdict(list)
    for season, episodes in seasons_episodes.items():
        for episode_id in episodes:
            get_request = requests.get(
                _OMDB_API_BASE_URL, params={
                    'apikey': _OMDB_API_KEY,
                    'i': episode_id
                })
            seasons_data[season].append(json.loads(get_request.content.decode('utf-8')))
            time.sleep(_OMDB_API_WAIT_TIME)
    return seasons_data
