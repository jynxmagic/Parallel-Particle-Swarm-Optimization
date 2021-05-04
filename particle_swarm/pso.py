import numpy as np
import ray

from particle_swarm.tests.optimization_test_functions import sphere_np as fitness

NUM_CPUS = 8
PRECISION = 5
# hyper-params
INDIVIDUAL_WEIGHT = 2.8
SOCIAL_WEIGHT = 1.3
INERTIA = 0.9
LEARNING_RATE = 0.7
PARTICLE_AMOUNT = 30
# search space
DIMENSIONS = 20
TARGET_SCORE = 0
MIN_POS = -5
MAX_POS = 10
MAX_ITERATIONS = 100
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
    """Instantiate position and return particles defined by PARTICLE_DT.

    Returns:
        np.array: numpy array of PARTICLE_DT, size PARTICLE_AMOUNT
    """
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


def _current_score_is_better_than_best_score(current_score, best_score):
    """Calculate the difference between current_score and best_score to TARGET_SCORE.

    Args:
        current_score (float): current score
        best_score (float): best score

    Returns:
        bool: True if current_score is closer to TARGET_SCORE.
    """
    if best_score is None:
        return True

    diff_current = np.abs(TARGET_SCORE - current_score)
    diff_best = np.abs(TARGET_SCORE - best_score)

    return diff_current <= diff_best


@ray.remote
def _score(pos):
    return fitness(pos)


@ray.remote
def _calc_vel_t(particle, gbest_p, r_1, r_2):
    return (
        INERTIA * particle["velocity"]
        + INDIVIDUAL_WEIGHT * r_1 * (particle["best_pos"] - particle["curr_pos"])
        + SOCIAL_WEIGHT * r_2 * (gbest_p - particle["curr_pos"])
    )


def pso(particles):
    """Run the particle swarm until end conditions have been satisfied.

    Args:
        particles (np.array): particle swarm to iterate over

    Returns:
        array: returns swarm[0], total runs[1], swarm best position[2], swarm best score[3].
    """
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
                    print(run_count, gbest_s, gbest_p)
                particles[index] = particle

        # calculate velocity
        r_1 = np.random.default_rng().random()
        r_2 = np.random.default_rng().random()
        vel_t = ray.get(
            [_calc_vel_t.remote(particle, gbest_p, r_1, r_2) for particle in particles],
        )
        particles["curr_pos"] = LEARNING_RATE * (particles["curr_pos"] + vel_t)
        particles["velocity"] = vel_t

        run_count += 1

    return [run_count, gbest_p, gbest_s]


def run():
    """Method used as a standard entry point to call the script.

    Returns:
        array: results of the particle swarm
    """
    return pso(_build_particles())
