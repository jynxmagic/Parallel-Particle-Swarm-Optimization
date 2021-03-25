import numpy as np

from particle_swarm.configuration.constants import DIMENSIONS, PARTICLE_AMOUNT
from particle_swarm.data.particle import PARTICLE_DT, build_particles
from particle_swarm.helper.helper import current_score_is_better_than_best_score

SWARM_DT = np.dtype(
    [
        ("particles", PARTICLE_DT, PARTICLE_AMOUNT),  # shape is param 3
        ("swarm_best_pos", np.float32, DIMENSIONS),
        ("swarm_best_score", np.float32),
    ],
)


def build_swarm():
    """Build and return a swarm object.

    Args:
        min_pos (integer): Min Position of search space
        max_pos (type): Max position of search space

    Returns:
        dict: Instantiated swarm object
    """
    base_swarm = np.ones(1, dtype=SWARM_DT)

    base_swarm["particles"] = build_particles()

    # set best score to first particles' position
    base_swarm["swarm_best_pos"] = base_swarm["particles"][0][0]["curr_pos"]
    base_swarm["swarm_best_score"] = np.inf

    return base_swarm


def update_swarm_best_score(swarm):
    """Iterate through swarm and update best scores.

    Args:
        swarm ([np.array]): Swarm with all positions scored

    Returns:
        swarm [np.array]: Swarm with best scores updated
    """
    # update particles best scores
    for index, particle in enumerate(swarm["particles"][0]):
        if current_score_is_better_than_best_score(
            particle["curr_score"],
            particle["best_score"],
        ):
            particle["best_pos"] = particle["curr_pos"]
            particle["best_score"] = particle["curr_score"]
            swarm["particles"][0][index] = particle
            # update swarms best score, if it's better
            if current_score_is_better_than_best_score(
                particle["curr_score"],
                swarm["swarm_best_score"],
            ):
                swarm["swarm_best_score"] = particle["curr_score"]
                swarm["swarm_best_pos"] = particle["best_pos"]

    return swarm
