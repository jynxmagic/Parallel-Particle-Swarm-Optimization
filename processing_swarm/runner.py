"""Particle search runner.

Utilizes Ray.io to parralelize the search process.
"""

import random

import ray  # type: ignore
import numpy as np

import helper
import sphere_function

ray.init(num_cpus=1)


INERTIA = 0.9
INDIVIDUAL_WEIGHT = random.random()
SOCIAL_WEIGHT = random.random()
LEARNING_RATE = random.random()
R1 = random.random()
R2 = random.random()


@ray.remote
def _calculate_score(swarm_mem_ref, particle_no):

    particle = swarm_mem_ref["particles"][0][particle_no]

    score = sphere_function.sphere_np(particle["curr_pos"])

    return score


@ray.remote
def _update_particle_position(swarm_mem_ref, particle_no):
    particle = swarm_mem_ref["particles"][0][particle_no]

    # vel_t defines the distance a particle will move this iteration
    vel_t = (
        INERTIA * particle["velocity"]
        + ((INDIVIDUAL_WEIGHT * R1) * (particle["best_pos"] - particle["curr_pos"]))
        + (
            (SOCIAL_WEIGHT * R2)
            * (swarm_mem_ref["swarm_best_pos"] - particle["curr_pos"])
        )
    )

    return vel_t[0]


def calculate_scores_for_swarm(swarm):
    """Calculate the score for each particles' position in the particle swarm.

    Args:
        swarm (Swarm): Swarm containing particles to calculate the score of.

    Returns:
        Swarm: Swarm with calculated positions
    """

    # update current scores
    swarm_mem_ref = ray.put(swarm)  # read-only
    scores_ref = [
        _calculate_score.remote(swarm_mem_ref, i - 1)
        for i in range(
            len(swarm["particles"][0])
        )  # spawn threads equal to particle amount
    ]

    scores = np.array(ray.get(scores_ref), dtype=float)

    swarm["particles"]["curr_score"] = scores

    # update best scores
    better_scores = helper.current_score_is_better_than_best_score(
        swarm["particles"]["curr_score"], swarm["particles"]["best_score"]
    )
    swarm["particles"][0]["best_score"] = np.where(
        better_scores,
        swarm["particles"][0]["curr_score"],
        swarm["particles"][0]["best_score"],
    )

    # update best pos
    # swarm["particles"][0]["best_pos"]
    swarm["particles"][better_scores]["best_pos"] = swarm["particles"][better_scores][
        "curr_pos"
    ]
    return swarm


def update_swarm_positions(swarm):
    """Updates all positions of the swarm ready for the next iteration.

    Note: function should be called after all particles have been scored.

    Args:
        swarm (Swarm): Swarm containing particles with positions to update.

    Returns:
        Swarm: Swarm with updated positions.
    """

    swarm_mem_ref = ray.put(swarm)  # read-only
    ray_refs = [
        _update_particle_position.remote(
            swarm_mem_ref,
            i - 1,
        )
        for i in range(len(swarm["particles"][0]))
    ]

    swarm_movement = ray.get(ray_refs)

    swarm["particles"][0]["velocity"] = swarm_movement

    swarm["particles"][0]["curr_pos"] += swarm_movement

    return swarm
