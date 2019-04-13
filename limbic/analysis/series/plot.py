from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from limbic.analysis import moving_window


def plot_emotion_all_episodes(subtitles_emotions, category):
    df_category_moving_window = _seasons_episodes_category_moving_window(
        subtitles_emotions, category)
    sns.set(style='ticks')
    pal = sns.color_palette()
    g = sns.FacetGrid(
        df_category_moving_window,
        row='episode',
        col='season',
        hue='episode',
        aspect=4,
        height=.8,
        palette=pal)
    g.map(plt.plot, 'time', f'total_{category}')
    g.map(plt.axhline, y=0, lw=1, clip_on=False)
    g.fig.subplots_adjust(hspace=-0.00)
    g.set_titles("")
    g.set_xlabels('')
    g.despine(bottom=True, left=True)
    plt.show()


def _seasons_episodes_category_moving_window(seasons_episodes_subtitles_emotions, category):
    data: Dict[str, List[Any]] = {'season': [], 'episode': [], f'total_{category}': [], 'time': []}
    max_mw = 0
    for season in seasons_episodes_subtitles_emotions.keys():
        for episode, emotions in seasons_episodes_subtitles_emotions[season].items():
            mw = moving_window(emotions, category)
            for idx, v in enumerate(mw):
                data[f'total_{category}'].append(v)
                data['episode'].append(f'E{episode}')
                data['season'].append(f'S{season}')
                data['time'].append(idx * 2)
            if max_mw < len(mw):
                max_mw = len(mw)
    return pd.DataFrame.from_dict(data)
