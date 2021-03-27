import sys

import numpy as np
from numba import jit

NUM_CPUS = 8
# hyper-params
INERTIA = 0.9
r_1 = np.random.default_rng().random()
INDIVIDUAL_WEIGHT = 0.6
r_2 = np.random.default_rng().random()
SOCIAL_WEIGHT = 0.8
LEARNING_RATE = 0.7
PARTICLE_AMOUNT = 100
# search space
DIMENSIONS = 2
TARGET_SCORE = 0
MIN_POS = 0
MAX_POS = 20

PARTICLE_DT = np.dtype(
    [
        ("curr_pos", np.float32, DIMENSIONS),
        ("curr_score", np.float32),
        ("best_score", np.float32),
        ("best_pos", np.float32, DIMENSIONS),
        ("velocity", np.float32, DIMENSIONS),
    ],
)


def build_particles():
    i_particles = np.empty(PARTICLE_AMOUNT, dtype=PARTICLE_DT)

    position = np.random.default_rng().uniform(
        size=(PARTICLE_AMOUNT, DIMENSIONS),
        low=MIN_POS,
        high=MAX_POS,
    )

    i_particles["curr_pos"] = position
    i_particles["curr_score"] = np.inf
    i_particles["best_score"] = np.inf
    i_particles["best_pos"] = position
    i_particles["velocity"] = position
    return i_particles


@jit(nopython=True)
def current_score_is_better_than_best_score(current_score, best_score):
    if best_score is None:
        return True

    diff_current = np.abs(TARGET_SCORE - current_score)
    diff_best = np.abs(TARGET_SCORE - best_score)

    return diff_current <= diff_best


def pso(
    particles,
    gbest_p=None,
    gbest_s=None,
):
    particles["curr_score"] = np.sum((particles["curr_pos"]) ** 2)  # sphere func

    if np.any(particles["curr_score"] == TARGET_SCORE):
        return particles

    for index, particle in enumerate(particles):
        print(particle)
        if current_score_is_better_than_best_score(
            particle["curr_score"], particle["best_score"]
        ):
            particle["best_score"] = particle["curr_score"]
            particle["best_pos"] = particle["curr_pos"]
            if current_score_is_better_than_best_score(particle["curr_score"], gbest_s):
                gbest_s = particle["curr_score"]
                gbest_p = particle["curr_pos"]
        particles[index] = particle

    vel_t = (
        INERTIA * particles["velocity"]
        + INDIVIDUAL_WEIGHT * r_1 * (particles["best_pos"] - particles["curr_pos"])
        + SOCIAL_WEIGHT * r_2 * (gbest_p - particles["curr_pos"])
    )

    particles["curr_pos"] = particles["curr_pos"] + vel_t
    particles["velocity"] = vel_t

    pso(particles, gbest_p, gbest_s)


sys.setrecursionlimit(10000)
np.set_printoptions(suppress=True)
pos = pso(build_particles())
print(pos)
