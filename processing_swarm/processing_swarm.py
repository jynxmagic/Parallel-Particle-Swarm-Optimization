"""Program entry point."""

import random

import helper
import runner

PARTICLE_AMOUNT = 10
DIMENSIONS = 3


def run(swarm_to_run):
    swarm_with_updated_positions = runner.update_swarm_positions(swarm_to_run)
    swarm_with_particles_scored = runner.calculate_scores_for_swarm(
        swarm_with_updated_positions,
    )
    return update_swarm_current_best_score(
        swarm_with_particles_scored,
    )


def _init_swarm():  # todo configuration

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
        Swarm: Instantiated swarm object
    """
    base_swarm = {"particles": []}

    for _ in range(PARTICLE_AMOUNT):
        particle_to_add = _build_particle(min_pos, max_pos)
        base_swarm["particles"].append(particle_to_add)

    # set best score to first particles' position
    base_swarm["swarm_best_pos"] = base_swarm["particles"][0]["curr_pos"]
    base_swarm["swarm_best_score"] = base_swarm["particles"][0]["curr_score"]

    return base_swarm


def _build_particle(min_pos, max_pos):
    pos = []
    vel = []
    for _ in range(DIMENSIONS):
        pos_d = random.uniform(min_pos, max_pos)
        pos.append(pos_d)

        vel_d = random.uniform(-1, 1)
        vel.append(vel_d)

    return {
        "name": "particle: " + str(random.randint(0, 99999)),
        "curr_pos": pos,
        "curr_score": None,
        "best_score": None,
        "best_pos": pos,
        "velocity": vel,
    }


def update_swarm_current_best_score(swarm_to_score):
    best_score = swarm_to_score["swarm_best_score"]
    best_pos = swarm_to_score["swarm_best_pos"]

    for particle in swarm_to_score["particles"]:
        if helper.current_score_is_better_than_best_score(
            particle["curr_score"],
            best_score,
        ):
            best_score = particle["curr_score"]
            best_pos = particle["curr_pos"]

    swarm_to_score["swarm_best_score"] = best_score
    swarm_to_score["swarm_best_pos"] = best_pos

    return swarm_to_score


if __name__ == "__main__":
    swarm = _init_swarm()

    # main loop
    RUN_COUNT = 1
    while swarm["swarm_best_score"] != helper.TARGET_SCORE:
        swarm = run(swarm)
        print("run: " + str(RUN_COUNT) + ", score: " + str(swarm["swarm_best_score"]))
        RUN_COUNT = RUN_COUNT + 1

    print(swarm["swarm_best_pos"])
    print(swarm["swarm_best_score"])
