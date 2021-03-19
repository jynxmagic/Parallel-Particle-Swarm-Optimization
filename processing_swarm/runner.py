"""Particle search runner.

Utilizes Ray.io to parralelize the search process.
"""

import random

import ray  # type: ignore
import numpy as np

import helper
import sphere_function

ray.init(num_cpus=8)


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
    print(particle)

    rand_factor1 = random.randint(0, 1)
    rand_factor2 = random.randint(0, 1)
    rand_factor3 = random.randint(0, 1)

    x = rand_factor1 * (
        particle["velocity"]
        + rand_factor2 * (particle["best_pos"] - particle["curr_pos"])
        + rand_factor3 * (swarm_mem_ref["swarm_best_pos"] - particle["curr_pos"])
    )

    print(x)

    return x

    for dimension in range(0, len(particle["curr_pos"])):

        current_position = particle["curr_pos"][dimension]
        best_position = particle["best_pos"][dimension]
        global_best = swarm_mem_ref["swarm_best_pos"][dimension]

        # vel_t defines the distance a particle will move this iteration
        vel_t = (
            INERTIA * particle["velocity"][dimension]
            + ((INDIVIDUAL_WEIGHT * R1) * (best_position - current_position))
            + ((SOCIAL_WEIGHT * R2) * (global_best - current_position))
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

    updated_swarm_positions = ray.get(ray_refs)

    exit()

    swarm[0] = updated_swarm_positions

    return swarm
