import numpy as np

from particle_swarm.configuration.constants import DIMENSIONS, PARTICLE_AMOUNT

PARTICLE_DT = np.dtype(
    [
        ("curr_pos", np.float32, DIMENSIONS),
        ("curr_score", np.float32),
        ("best_score", np.float32),
        ("best_pos", np.float32, DIMENSIONS),
        ("velocity", np.float32, DIMENSIONS),
    ],
)


def build_particles(min_pos, max_pos):
    """Builds and returns particles.

    Args:
        min_pos (float): minimum position of particle search space
        max_pos (float): max position of particle search space

    Returns:
        np.array: particles as np.array
    """
    particles = np.empty(PARTICLE_AMOUNT, dtype=PARTICLE_DT)

    position = np.random.default_rng().uniform(
        size=(PARTICLE_AMOUNT, DIMENSIONS), low=min_pos, high=max_pos
    )

    particles["curr_pos"] = position
    particles["curr_score"] = np.inf
    particles["best_score"] = np.inf
    particles["best_pos"] = position
    particles["velocity"] = position

    return particles
