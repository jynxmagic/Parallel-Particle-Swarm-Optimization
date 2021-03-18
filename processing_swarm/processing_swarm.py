"""Program entry point."""

import numpy as np

import helper
import runner

PARTICLE_AMOUNT = 10
DIMENSIONS = 3


# data types
SWARM_DT = np.dtype(
    [
        ("particles", np.object, PARTICLE_AMOUNT),  # shape is param 3
        ("swarm_best_pos", np.float, DIMENSIONS),
        ("swarm_best_score", np.float),
    ]
)
PARTICLE_DT = np.dtype(
    [
        ("particle_name", np.unicode_, 14),  # 14 char string
        ("curr_pos", np.float, 3),
        ("curr_score", np.float),
        ("best_score", np.float),
        ("best_pos", np.float, 3),
        ("velocity", np.int16),
    ]
)

# random generator
rsg = np.random.default_rng(1)


def run(swarm_to_run):
    swarm_with_updated_positions = runner.update_swarm_positions(swarm_to_run)
    swarm_with_particles_scored = runner.calculate_scores_for_swarm(
        swarm_with_updated_positions,
    )
    return update_swarm_current_best_score(
        swarm_with_particles_scored,
    )


def _init_swarm():  # todo configuration

    initalized_swarm = build_swarm()

    # calculate particle scores for start pos
    runner.calculate_scores_for_swarm(initalized_swarm)

    # calculate swarm best for start pos
    update_swarm_current_best_score(initalized_swarm)

    return initalized_swarm


def build_swarm():
    """Builds and returns a swarm object.

        x[0] = particles \n
        x[1] = swarm_best_pos \n
        x[2] = swarm_best_score \n
    Args:
        min_pos (integer): Min Position of vector
        max_pos (type): Max position of vector

    Returns:
        np.array: Instantiated swarm object
    """

    base_swarm = np.empty(
        1,
        dtype=SWARM_DT,
    )

    base_swarm["particles"] = build_particle()

    print(base_swarm["particles"])
    exit()

    for i in range(PARTICLE_AMOUNT):
        particle_to_add = build_particle()
        base_swarm[0][i - 1] = particle_to_add

    # set best score to first particles' position
    base_swarm[1] = base_swarm[0][0][1]
    base_swarm[2] = base_swarm[0][0][2]

    return base_swarm


def build_particle():
    """Builds and returns a particle object.

        x[0] = particle name \n
        x[1] = particle pos \n
        x[2] = particle current score \n
        x[3] = particle best score \n
        x[4] = particle best pos \n
        x[5] = velocity \n

    Returns:
        np.array: particle as array
    """
    pos = np.array(3, dtype=float)
    pos = rsg.integers(low=1, high=DIMENSIONS)  # np array

    print(pos)

    arr = np.empty(
        1,
        dtype=PARTICLE_DT,
    )

    particle_no = str(rsg.integers(low=0, high=9999, size=1)[0])
    as_str = "particle: " + particle_no

    arr["particle_name"] = as_str
    arr["curr_pos"] = pos
    arr["curr_score"] = None
    arr["best_score"] = None
    arr["best_pos"] = pos
    arr["velocity"] = rsg.integers(low=-1, high=1, size=1)[0]

    return arr


def update_swarm_current_best_score(swarm_to_score):
    best_score = swarm_to_score[2]
    best_pos = swarm_to_score[1]

    print(swarm_to_score[0])

    best_particle_current = np.amax(swarm_to_score[0], axis=[1, 2])

    exit()

    if helper.current_score_is_better_than_best_score(
        best_particle_current[3],
        best_score,
    ):
        best_score = best_particle_current[3]
        best_pos = best_particle_current[4]

    swarm_to_score[2] = best_score
    swarm_to_score[1] = best_pos

    return swarm_to_score


if __name__ == "__main__":
    swarm = _init_swarm()

    # main loop
    RUN_COUNT = 1
    while swarm[2] != helper.TARGET_SCORE:
        swarm = run(swarm)
        print("run: " + str(RUN_COUNT) + ", score: " + str(swarm[2]))
        RUN_COUNT = RUN_COUNT + 1
