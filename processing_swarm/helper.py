"""Module containing reused functions throughout the code-base."""

TARGET_SCORE = 0


def current_score_is_better_than_best_score(current_score, best_score):
    """Determine if the current score is closer than the best score.

    Args:
        current_score (int): Score of the particles current position.
        best_score (int): Score of the particles best position.

    Returns:
        bool: True if current score is closer to the target score.
    """
    if best_score is None:
        return True

    if _current_less_than_best_and_target_lower(current_score, best_score):
        return True

    if _current_more_than_best_and_target_higher(current_score, best_score):
        return True

    return False


def _current_less_than_best_and_target_lower(current_score, best_score):
    if TARGET_SCORE <= current_score < best_score:
        return True
    return False


def _current_more_than_best_and_target_higher(current_score, best_score):
    if best_score < current_score <= TARGET_SCORE:
        return True
    return False
