"""Program entry point."""
import time

from numba import jit

from particle_swarm.algorithm.particle_swarm import init_swarm, run
from particle_swarm.configuration.constants import (
    DIMENSIONS,
    INDIVIDUAL_WEIGHT,
    LEARNING_RATE,
    NUM_CPUS,
    PARTICLE_AMOUNT,
    SOCIAL_WEIGHT,
    TARGET_SCORE,
)


@jit(forceobj=True)
def main():
    """Main method for the program."""
    swarm = init_swarm()

    start = time.time()

    # main loop
    run_count = 1
    while swarm["swarm_best_score"] != TARGET_SCORE:
        swarm = run(swarm)
        print("run: ", run_count, ", score: ", swarm["swarm_best_score"])
        run_count = run_count + 1

    print(swarm["swarm_best_pos"])
    print(swarm["swarm_best_score"])

    end = time.time()

    print("time taken: ", end - start)


def print_config():
    """Prints the configuration values."""
    print(
        "############ CONFIGURATION: ##############\n",
        "Target Score: ",
        TARGET_SCORE,
        "\n",
        "Dimensions: ",
        DIMENSIONS,
        "\n",
        "Num CPUS: ",
        NUM_CPUS,
        "\n",
        "===Hyper-parameters===\n",
        "Individual Weight: ",
        INDIVIDUAL_WEIGHT,
        "\n",
        "Social Weight: ",
        SOCIAL_WEIGHT,
        "\n",
        "Learning Rate: ",
        LEARNING_RATE,
        "\n",
        "Particle Amount: ",
        PARTICLE_AMOUNT,
        "\n",
    )


if __name__ == "__main__":
    main()
    print_config()
