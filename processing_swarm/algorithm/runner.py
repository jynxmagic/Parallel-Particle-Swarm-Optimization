"""Particle search runner.

Utilizes Ray.io to parralelize the search process.
"""

import random

import ray  # type: ignore

from processing_swarm.configuration.constants import (
    DIMENSIONS,
    INDIVIDUAL_WEIGHT,
    INERTIA,
    NUM_CPUS,
    SOCIAL_WEIGHT,
)
from processing_swarm.helper.helper import current_score_is_better_than_best_score
from processing_swarm.test_cost_functions.sphere_function import sphere_pp

ray.init(num_cpus=NUM_CPUS)


@ray.remote
def _calculate_score(particle):
    curr_pos = particle["curr_pos"]

    score = sphere_pp(curr_pos)
    particle["curr_score"] = score

    if current_score_is_better_than_best_score(
        score,
        particle["best_score"],
    ):
        particle["best_score"] = score
        particle["best_pos"] = curr_pos

    return particle


@ray.remote
def _update_particle_position(particle, swarm_best_pos, r_1, r_2):

    for dimension in range(DIMENSIONS):
        current_position = particle["curr_pos"][dimension]
        best_position = particle["best_pos"][dimension]
        global_best = swarm_best_pos[dimension]

        # vel_t defines the distance a particle will move this iteration
        vel_t = (
            INERTIA * particle["velocity"][dimension]
            + ((INDIVIDUAL_WEIGHT * r_1) * (best_position - current_position))
            + ((SOCIAL_WEIGHT * r_2) * (global_best - current_position))
        )

        particle["velocity"][dimension] = vel_t
        particle["curr_pos"][dimension] += vel_t

    return particle


def calculate_scores_for_swarm(swarm):
    """Calculate the score for each particles' position in the particle swarm.

    Args:
        swarm (Swarm): Swarm containing particles to calculate the score of.

    Returns:
        Swarm: Swarm with calculated positions
    """
    ray_refs = [_calculate_score.remote(particle) for particle in swarm["particles"]]

    scored_particles = ray.get(ray_refs)

    swarm["particles"] = scored_particles

    return swarm


def update_swarm_positions(swarm):
    """Updates all positions of the swarm ready for the next iteration.

    Note: function should be called after all particles have been scored.

    Args:
        swarm (Swarm): Swarm containing particles with positions to update.

    Returns:
        Swarm: Swarm with updated positions.
    """
    r_1 = random.random()
    r_2 = random.random()

    swarm_best_pos = swarm["swarm_best_pos"]
    ray_refs = [
        _update_particle_position.remote(particle, swarm_best_pos, r_1, r_2)
        for particle in swarm["particles"]
    ]

    updated_swarm_positions = ray.get(ray_refs)

    swarm["particles"] = updated_swarm_positions

    return swarm
