"""Particle search runner.

Utilizes Ray.io to parralelize the search process.
"""

import random

import ray  # type: ignore
import helper
import sphere_function

ray.init(num_cpus=4)


@ray.remote
def _calculate_score(particle):
    curr_pos = particle[1]

    score = sphere_function.sphere_np(curr_pos)

    particle[2] = score

    if helper.current_score_is_better_than_best_score(
        score,
        particle[3],
    ):
        particle[3] = score
        particle[4] = curr_pos

    return particle


@ray.remote
def _update_particle_position(particle, swarm_best_pos):

    for dimension in range(0, len(particle[1])):
        r_velocity = random.randint(-1, 1)

        particle[5] = r_velocity

        # todo rand1 should be value of social weight between 0,1
        rand_factor1 = random.randint(0, 1)
        rand_factor2 = random.randint(0, 1)
        rand_factor3 = random.randint(0, 1)
        current_position = particle[1][dimension]
        best_position = particle[4][dimension]
        global_best = swarm_best_pos[dimension]

        # vel_t defines the distance a particle will move this iteration
        vel_t = rand_factor1 * (
            particle[5]
            + rand_factor2 * (best_position - current_position)
            + rand_factor3 * (global_best - current_position)
        )

        particle["curr_pos"][dimension] += vel_t

    return particle


def calculate_scores_for_swarm(swarm):
    """Calculate the score for each particles' position in the particle swarm.

    Args:
        swarm (Swarm): Swarm containing particles to calculate the score of.

    Returns:
        Swarm: Swarm with calculated positions
    """
    ray_refs = [_calculate_score.remote(particle) for particle in swarm[0]]

    scored_particles = ray.get(ray_refs)

    swarm[0] = scored_particles

    return swarm


def update_swarm_positions(swarm):
    """Updates all positions of the swarm ready for the next iteration.

    Note: function should be called after all particles have been scored.

    Args:
        swarm (Swarm): Swarm containing particles with positions to update.

    Returns:
        Swarm: Swarm with updated positions.
    """
    swarm_best_pos = swarm[1]
    ray_refs = [
        _update_particle_position.remote(particle, swarm_best_pos)
        for particle in swarm[0]
    ]

    updated_swarm_positions = ray.get(ray_refs)

    swarm[0] = updated_swarm_positions

    return swarm
