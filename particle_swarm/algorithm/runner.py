"""Particle search runner.

Utilizes Ray.io to parralelize the search process.
"""

import random

import numpy as np
import ray  # type: ignore

from particle_swarm.configuration.constants import (
    DIMENSIONS,
    INDIVIDUAL_WEIGHT,
    INERTIA,
    LEARNING_RATE,
    NUM_CPUS,
    SOCIAL_WEIGHT,
)
from particle_swarm.helper.helper import current_score_is_better_than_best_score
from particle_swarm.test_cost_functions.sphere_function import sphere_np

ray.init(num_cpus=NUM_CPUS)


@ray.remote
def _calculate_score(particle):
    curr_pos = particle["curr_pos"]

    score = sphere_np(curr_pos)
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

    current_position = particle["curr_pos"]
    best_position = particle["best_pos"]
    global_best = swarm_best_pos

    # vel_t defines the distance a particle will move this iteration
    vel_t =(
        INERTIA * particle["velocity"]
        + ((INDIVIDUAL_WEIGHT * r_1) * (best_position - current_position))
        + ((SOCIAL_WEIGHT * r_2) * (global_best - current_position))
    )
    
    return vel_t


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
    r_1 = np.random.default_rng().random()
    r_2 = np.random.default_rng().random()

    swarm_best_pos = swarm["swarm_best_pos"]
    ray_refs = [
        _update_particle_position.remote(particle, swarm_best_pos, r_1, r_2)
        for particle in swarm["particles"]
    ]

    velocity_tomorrow = np.copy(ray.get(ray_refs))

    for index, particle in enumerate(swarm["particles"]):
        particle["velocity"] = velocity_tomorrow[index]
        particle["curr_pos"] = particle["curr_pos"] + (LEARNING_RATE * particle["velocity"])

    return swarm
