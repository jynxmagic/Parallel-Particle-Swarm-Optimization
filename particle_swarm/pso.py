import time

import numpy as np
import ray
from numba import jit

from particle_swarm.tests.linear_regression import boston

NUM_CPUS = 8
PRECISION = 20
# hyper-params
INERTIA = 0.9
INDIVIDUAL_WEIGHT = 0.5
SOCIAL_WEIGHT = 0.3
LEARNING_RATE = 0.7
PARTICLE_AMOUNT = 100
# search space
DIMENSIONS = 1
TARGET_SCORE = 0
MIN_POS = 0
MAX_POS = 20
MAX_ITERATIONS = 500
# dtypes
PARTICLE_DT = np.dtype(
    [
        ("curr_pos", np.float64, (DIMENSIONS,)),
        ("curr_score", np.float64),
        ("best_score", np.float64),
        ("best_pos", np.float64, (DIMENSIONS,)),
        ("velocity", np.float64, (DIMENSIONS,)),
    ],
)

ray.init(num_cpus=NUM_CPUS)


def _build_particles():
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
def _current_score_is_better_than_best_score(current_score, best_score):
    if best_score is None:
        return True

    diff_current = np.abs(TARGET_SCORE - current_score)
    diff_best = np.abs(TARGET_SCORE - best_score)

    return diff_current <= diff_best


@ray.remote
def _score(pos):
    return boston(pos[0])


@ray.remote
def _calc_vel_t(particle, gbest_p, r_1, r_2):
    return (
        INERTIA * particle["velocity"]
        + INDIVIDUAL_WEIGHT * r_1 * (particle["best_pos"] - particle["curr_pos"])
        + SOCIAL_WEIGHT * r_2 * (gbest_p - particle["curr_pos"])
    )


def pso(particles):
    gbest_p = None
    gbest_s = np.inf

    run_count = 1
    while round(gbest_s, PRECISION) != TARGET_SCORE and run_count < MAX_ITERATIONS:
        # update scores
        particles["curr_score"] = ray.get(
            [_score.remote(particle["curr_pos"]) for particle in particles]
        )

        # updates best scores
        for index, particle in enumerate(particles):
            if _current_score_is_better_than_best_score(
                particle["curr_score"], particle["best_score"]
            ):
                particle["best_score"] = particle["curr_score"]
                particle["best_pos"] = particle["curr_pos"]
                if _current_score_is_better_than_best_score(
                    particle["curr_score"],
                    gbest_s,
                ):
                    gbest_s = particle["curr_score"]
                    gbest_p = particle["curr_pos"]
                particles[index] = particle

        # calculate velocity
        r_1 = np.random.default_rng().random()
        r_2 = np.random.default_rng().random()
        vel_t = ray.get(
            [_calc_vel_t.remote(particle, gbest_p, r_1, r_2) for particle in particles]
        )
        particles["curr_pos"] = LEARNING_RATE * (particles["curr_pos"] + vel_t)
        particles["velocity"] = vel_t

        run_count += 1

    return [particles, run_count, gbest_p, gbest_s]


def run():
    np.set_printoptions(precision=PRECISION, suppress=True)  # non-scientific notation
    start = time.time()

    res = pso(_build_particles())

    end = time.time()

    print("time taken: ", end - start)
    return res
