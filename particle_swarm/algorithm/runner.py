"""Particle search runner.

Utilizes Ray.io to parralelize the search process.
"""

import numpy as np
import ray  # type: ignore

from particle_swarm.configuration.constants import (
    INDIVIDUAL_WEIGHT,
    INERTIA,
    NUM_CPUS,
    SOCIAL_WEIGHT,
)
from particle_swarm.data.swarm import update_swarm_best_score
from particle_swarm.test_cost_functions.sphere_function import sphere_np

ray.init(num_cpus=NUM_CPUS)


def run(swarm_to_run):
    """Completes a full iteration over a particle swarm.

    Args:
        swarm_to_run (swarm): Swarm with particles initialized and scored

    Returns:
        initalized_swarm: swarm with iteration complete
    """
    swarm_with_updated_positions = update_swarm_positions(swarm_to_run)
    return calculate_scores_for_swarm(
        swarm_with_updated_positions,
    )


@ray.remote
def _calculate_score(particle):
    return sphere_np(particle["curr_pos"])


@ray.remote
def _update_particle_position(particle, swarm_best_pos, r_1, r_2):

    current_position = particle["curr_pos"]
    best_position = particle["best_pos"]
    global_best = swarm_best_pos

    # vel_t. defines the distance a particle will move this iteration
    return (
        INERTIA * particle["velocity"]
        + ((INDIVIDUAL_WEIGHT * r_1) * (best_position - current_position))
        + ((SOCIAL_WEIGHT * r_2) * (global_best - current_position))
    )[0]


def calculate_scores_for_swarm(swarm):
    """Calculate the score for each particles' position in the particle swarm.

    Args:
        swarm (Swarm): Swarm containing particles to calculate the score of.

    Returns:
        Swarm: Swarm with calculated positions
    """

    particles = swarm["particles"][0]

    ray_refs = [_calculate_score.remote(particle) for particle in particles]

    scores = np.array(ray.get(ray_refs), dtype=float)

    swarm["particles"][0]["curr_score"] = scores

    return update_swarm_best_score(swarm)


def update_swarm_positions(swarm):
    """Updates all positions of the swarm ready for the next iteration.

    Note: function should be called after all particles have been scored.

    Args:
        swarm (Swarm): Swarm containing particles with positions to update.

    Returns:
        Swarm: Swarm with updated positions.
    """
    r_1 = np.random.default_rng().random()
    r_2 = np.random.default_rng().random()

    swarm_best_pos = swarm["swarm_best_pos"]
    ray_refs = [
        _update_particle_position.remote(particle, swarm_best_pos, r_1, r_2)
        for particle in swarm["particles"][0]
    ]

    velocity_tomorrow = np.copy(ray.get(ray_refs))

    swarm["particles"][0]["velocity"] = velocity_tomorrow
    swarm["particles"][0]["curr_pos"] += velocity_tomorrow

    return swarm
