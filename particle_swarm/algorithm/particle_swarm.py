"""Program entry point."""

import numpy as np

from particle_swarm.algorithm import runner
from particle_swarm.configuration.constants import DIMENSIONS, PARTICLE_AMOUNT
from particle_swarm.helper import helper

random_generator = np.random.default_rng()


# data types
PARTICLE_DT = np.dtype(
    [
        ("particle_name", np.unicode_, 14),  # 14 char string
        ("curr_pos", np.float32, DIMENSIONS),
        ("curr_score", np.float32),
        ("best_score", np.float32),
        ("best_pos", np.float32, DIMENSIONS),
        ("velocity", np.float32, DIMENSIONS),
    ],
)
SWARM_DT = np.dtype(
    [
        ("particles", PARTICLE_DT, PARTICLE_AMOUNT),  # shape is param 3
        ("swarm_best_pos", np.float32, DIMENSIONS),
        ("swarm_best_score", np.float32),
    ],
)


def run(swarm_to_run):
    """Completes a full iteration over a particle swarm.

    Args:
        swarm_to_run (swarm): Swarm with particles initialized and scored

    Returns:
        initalized_swarm: swarm with iteration complete
    """
    swarm_with_updated_positions = runner.update_swarm_positions(swarm_to_run)
    swarm_with_particles_scored = runner.calculate_scores_for_swarm(
        swarm_with_updated_positions,
    )
    return update_swarm_current_best_score(
        swarm_with_particles_scored,
    )


def init_swarm():
    """Initialize a swarm object.

    Swarm object contains particles initialized and scored.

    Args:
        None

    Returns:
        dict: Initialized swarm object
    """
    initalized_swarm = build_swarm(1, 1000)

    # calculate particle scores for start pos
    runner.calculate_scores_for_swarm(initalized_swarm)

    # calculate swarm best for start pos
    update_swarm_current_best_score(initalized_swarm)

    return initalized_swarm


def build_swarm(min_pos, max_pos):
    """Builds and returns a swarm object.

    Args:
        min_pos (integer): Min Position of vector
        max_pos (type): Max position of vector

    Returns:
        dict: Instantiated swarm object
    """
    base_swarm = np.ones(1, dtype=SWARM_DT)

    base_swarm["particles"] = _build_particles(min_pos, max_pos)

    # set best score to first particles' position
    base_swarm["swarm_best_pos"] = base_swarm["particles"][0][0]["curr_pos"]
    base_swarm["swarm_best_score"] = np.inf

    return base_swarm


def _build_particles(min_pos, max_pos):
    """Builds and returns a particle object.

    Args:
        min_pos (float): minimum position of particle search space
        max_pos (float): max position of particle search space

    Returns:
        np.array: particle as array
    """
    particles = np.empty(PARTICLE_AMOUNT, dtype=PARTICLE_DT)

    particles["curr_pos"] = random_generator.uniform(
        size=(PARTICLE_AMOUNT, DIMENSIONS), low=min_pos, high=max_pos
    )
    particles["curr_score"] = np.inf
    particles["best_score"] = np.inf
    particles["best_pos"] = particles["curr_pos"]
    particles["velocity"] = particles["curr_pos"]

    return particles


def update_swarm_current_best_score(swarm_to_score):
    """Update the swarms current best score.

    Args:
        swarm_to_score (dict): Swarm with particles scored.

    Returns:
        dict: Swarm with swarm_best_score and swarm_best_pos updated.
    """
    best_score = swarm_to_score["swarm_best_score"]
    best_pos = swarm_to_score["swarm_best_pos"]

    curr_best_score = swarm_to_score["particles"][0][
        np.abs(
            swarm_to_score["particles"]["curr_score"] - helper.TARGET_SCORE,
        ).argmin()
    ]

    if helper.current_score_is_better_than_best_score(
        curr_best_score["curr_score"],
        best_score,
    ):
        best_score = curr_best_score["curr_score"]
        best_pos = curr_best_score["curr_pos"]

    swarm_to_score["swarm_best_score"] = best_score
    swarm_to_score["swarm_best_pos"] = best_pos

    return swarm_to_score
