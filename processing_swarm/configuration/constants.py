"""Configuration values can be changed in this module."""
import numpy as np

TARGET_SCORE = 10000
DIMENSIONS = 3
NUM_CPUS = 16

# https://numpy.org/doc/stable/reference/random/generated/numpy.random.Generator.random.html
INERTIA = 0.9
INDIVIDUAL_WEIGHT = 1 - 0.5 * np.random.default_rng().random(dtype=np.float32)
SOCIAL_WEIGHT = 1 - 0.5 * np.random.default_rng().random(dtype=np.float32)
LEARNING_RATE = 1 - 0.5 * np.random.default_rng().random(dtype=np.float32)
PARTICLE_AMOUNT = 10
