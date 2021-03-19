"""Program entry point."""

import numpy as np

import helper
import runner

PARTICLE_AMOUNT = 10
DIMENSIONS = 3


# data types
PARTICLE_DT = np.dtype(
    [
        ("particle_name", np.unicode_, 14),  # 14 char string
        ("curr_pos", np.float32, DIMENSIONS),
        ("curr_score", np.float32),
        ("best_score", np.float32),
        ("best_pos", np.float32, DIMENSIONS),
        ("velocity", np.float32, DIMENSIONS),
    ]
)
SWARM_DT = np.dtype(
    [
        ("particles", PARTICLE_DT, PARTICLE_AMOUNT),  # shape is param 3
        ("swarm_best_pos", np.float32, DIMENSIONS),
        ("swarm_best_score", np.float32),
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

    base_swarm = np.ones(1, dtype=SWARM_DT)

    base_swarm["particles"] = build_particles()

    # set best score to first particles' position
    base_swarm["swarm_best_pos"] = base_swarm["particles"][0]["curr_pos"][0]
    base_swarm["swarm_best_score"] = np.inf

    return base_swarm


def build_particles():
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

    particles = np.empty(10, dtype=PARTICLE_DT)

    particles["particle_name"] = rsg.integers(low=0, high=9999, size=PARTICLE_AMOUNT)
    particles["curr_pos"] = (
        rsg.random(size=(PARTICLE_AMOUNT, DIMENSIONS), dtype=np.float32) * 1000
    )
    particles["curr_score"] = None
    particles["best_score"] = np.inf
    particles["best_pos"] = particles["curr_pos"]
    particles["velocity"] = rsg.integers(
        size=(PARTICLE_AMOUNT, DIMENSIONS), low=-1, high=1
    )

    return particles


def update_swarm_current_best_score(swarm_to_score):
    best_score = swarm_to_score["swarm_best_score"]
    best_pos = swarm_to_score["swarm_best_pos"]

    curr_best_score = swarm_to_score["particles"][0][
        np.abs(
            swarm_to_score["particles"]["best_score"] - helper.TARGET_SCORE,
        ).argmin()
    ]

    if helper.current_score_is_better_than_best_score(
        curr_best_score["best_score"],
        best_score,
    ):
        best_score = curr_best_score["best_score"]
        best_pos = curr_best_score["best_pos"]

    swarm_to_score["swarm_best_score"] = best_score
    swarm_to_score["swarm_best_pos"] = best_pos

    return swarm_to_score


if __name__ == "__main__":
    swarm = _init_swarm()

    # main loop
    RUN_COUNT = 1
    while swarm["swarm_best_score"] != helper.TARGET_SCORE:
        # while swarm["swarm_best_score"] != helper.TARGET_SCORE:
        swarm = run(swarm)
        print(swarm["particles"])
        print("run: " + str(RUN_COUNT) + ", score: " + str(swarm["swarm_best_score"]))
        RUN_COUNT = RUN_COUNT + 1
