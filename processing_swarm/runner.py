"""Particle search runner.

Utilizes Ray.io to parralelize the search process.
"""

import random

import ray  # type: ignore

import helper
import sphere_function

ray.init()


@ray.remote
def _calculate_score(particle):
    score = sphere_function.sphere_pp(particle["curr_pos"])
    particle["curr_score"] = score

    if helper.current_score_is_better_than_best_score(
        particle["curr_score"],
        particle["best_score"],
    ):
        particle["best_score"] = particle["curr_score"]
        particle["best_pos"] = particle["curr_pos"]
        print(particle)

    return particle


@ray.remote
def _update_particle_position(particle, swarm_best_pos):
    # move particles currPos
    # https://gyazo.com/b52c066fa8aa53bc68e9e161f650c289
    particle_current_position = particle["curr_pos"][0]
    particle_best_position = particle["best_pos"][0]

    random_int = random.randint(1, 3)
    self_movement = random_int * particle["velocity"]
    if particle_best_position != particle_current_position:
        self_movement += random_int * (
            particle_best_position - particle_current_position
        )
    else:
        self_movement += particle_current_position
    if particle_current_position != swarm_best_pos[0]:
        self_movement += random_int * (swarm_best_pos[0] - particle_current_position)
    particle_current_position = self_movement
    # calculate scores

    particle["curr_pos"] = [particle_current_position]

    return particle


def calculate_scores_for_swarm(swarm):
    """Calculate the score for each particles' position in the particle swarm.

    Args:
        swarm (Swarm): Swarm containing particles to calculate the score of.

    Returns:
        Swarm: Swarm with calculated positions
    """
    ray_refs = [_calculate_score.remote(particle) for particle in swarm.particles]

    scored_particles = ray.get(ray_refs)

    swarm.particles = scored_particles

    return swarm


def update_swarm_positions(swarm):
    """Updates all positions of the swarm ready for the next iteration.

    Note: function should be called after all particles have been scored.

    Args:
        swarm (Swarm): Swarm containing particles with positions to update.

    Returns:
        Swarm: Swarm with updated positions.
    """
    swarm_best_pos = swarm.swarm_best_pos
    ray_refs = [
        _update_particle_position.remote(particle, swarm_best_pos)
        for particle in swarm.particles
    ]

    updated_swarm_positions = ray.get(ray_refs)

    swarm.particles = updated_swarm_positions

    return swarm
