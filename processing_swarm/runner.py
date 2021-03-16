"""Particle search runner.

Utilizes Ray.io to parralelize the search process.
"""

import random

import ray  # type: ignore
import helper
import sphere_function

ray.init(num_cpus=4)


INERTIA = 0.9
INDIVIDUAL_WEIGHT = random.random()
SOCIAL_WEIGHT = random.random()
LEARNING_RATE = random.random()
R1 = random.random()
R2 = random.random()

@ray.remote
def _calculate_score(particle):
    curr_pos = particle["curr_pos"]

    score = sphere_function.sphere_pp(curr_pos)
    particle["curr_score"] = score

    if helper.current_score_is_better_than_best_score(
        score,
        particle["best_score"],
    ):
        particle["best_score"] = score
        particle["best_pos"] = curr_pos
        
    return particle


@ray.remote
def _update_particle_position(particle, swarm_best_pos):

    for dimension in range(0, len(particle["curr_pos"])):

        current_position = particle["curr_pos"][dimension]
        best_position = particle["best_pos"][dimension]
        global_best = swarm_best_pos[dimension]

        # vel_t defines the distance a particle will move this iteration
        vel_t = INERTIA * particle["velocity"][dimension] \
            + ((INDIVIDUAL_WEIGHT * R1) * (best_position - current_position)) \
            + ((SOCIAL_WEIGHT * R2) * (global_best - current_position))

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

    """
    R1 and R2 are changed every iteration in some equations. see:
    https://www.intechopen.com/books/swarm-intelligence-recent-advances-new-perspectives-and-applications/particle-swarm-optimization-a-powerful-technique-for-solving-engineering-problems
    vs
    https://www.hindawi.com/journals/mpe/2015/931256/#EEq4
    R1 = random.random()
    R2 = random.random()
    """

    swarm_best_pos = swarm["swarm_best_pos"]
    ray_refs = [
        _update_particle_position.remote(particle, swarm_best_pos)
        for particle in swarm["particles"]
    ]

    updated_swarm_positions = ray.get(ray_refs)

    swarm["particles"] = updated_swarm_positions

    return swarm
