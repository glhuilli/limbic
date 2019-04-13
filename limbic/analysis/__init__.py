from typing import List, Tuple

import numpy as np

from limbic.limbic_types import TimeEmotion

# TODO: Consider moving limbic.utils.analysis -> limbic.analysis


def get_total(emotions: List[TimeEmotion], category: str) -> float:
    return sum([e.emotion.value for e in emotions if e.emotion.category == category])


def get_mean(emotions: List[TimeEmotion], category: str) -> np.ndarray:
    values = []
    for e in emotions:
        if e.emotion.category == category:
            values.append(e.emotion.value)
        else:
            values.append(0.0)
    return np.mean(values)


def moving_window(emotions: List[TimeEmotion], category: str, step: int = 120,
                  window: int = 300) -> List[float]:
    window_start = 0
    current_window_total = 0.0
    windows_total = []
    while window_start + window < max([x.seconds for x in emotions]):
        for e in emotions:
            if window_start <= e.seconds <= window_start + window:
                if e.emotion.category == category:
                    current_window_total += e.emotion.value
            elif e.seconds > window_start + window:
                windows_total.append(current_window_total)
                current_window_total = 0
                window_start += step
                break
    return windows_total


def get_max_delta(emotions: List[TimeEmotion], category: str, step: int = 120,
                  window: int = 300) -> Tuple[float, int]:
    window_totals = moving_window(emotions, category, step=step, window=window)
    previous_window_total = 0.0
    max_delta = 0.0
    time_max_delta = 0
    for idx, current_window_total in enumerate(window_totals):
        delta = current_window_total - previous_window_total
        if delta > 0 and delta > max_delta:
            max_delta = delta
            time_max_delta = idx * step
        previous_window_total = current_window_total
    return max_delta, time_max_delta
