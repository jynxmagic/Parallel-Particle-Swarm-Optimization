"""Module containing reused functions throughout the code-base."""

import numpy as np

TARGET_SCORE = 0


def current_score_is_better_than_best_score(particles_curr_score, particles_best_score):
    """Determine if the current score is closer than the best score.

    Args:
        current_score (np.array): particles currrent scores
        best_score (int): particles best scores
    Returns:
        np.array(bool)*particle_amount: True if current score is closer to the target score.
    """
    cl_tl = _current_less_than_best_and_target_lower(
        particles_curr_score,
        particles_best_score,
    )
    cm_th = _current_more_than_best_and_target_higher(
        particles_curr_score,
        particles_best_score,
    )

    return np.logical_or(cl_tl, cm_th)


def _current_less_than_best_and_target_lower(
    particles_curr_score,
    particles_best_score,
):
    a = particles_curr_score < particles_best_score
    b = particles_curr_score >= TARGET_SCORE
    return np.logical_and(a, b)


def _current_more_than_best_and_target_higher(
    particles_curr_score,
    particles_best_score,
):
    a = particles_curr_score > particles_best_score
    b = particles_curr_score <= TARGET_SCORE
    return np.logical_and(a, b)
