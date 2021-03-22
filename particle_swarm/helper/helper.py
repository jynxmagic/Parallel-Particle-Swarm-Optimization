"""Module containing reused functions throughout the code-base."""
import numpy as np

from particle_swarm.configuration.constants import TARGET_SCORE


def current_score_is_better_than_best_score(current_score, best_score):
    """Determine if the current score is closer than the best score.

    Calculate the difference between current_score and best_score to TARGET_SCORE
    returns true if current_score has a smaller difference than best_score.

    Args:
        current_score (int): Score of the particles current position.
        best_score (int): Score of the particles best position.

    Returns:
        bool: True if current_score is closer to TARGET_SCORE.
    """
    if best_score is None:
        return True

    diff_current = np.abs(TARGET_SCORE - current_score)
    diff_best = np.abs(TARGET_SCORE - best_score)

    return diff_current < diff_best
