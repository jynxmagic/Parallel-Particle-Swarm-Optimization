"""Program entry point."""
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

if __name__ == "__main__":
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
    swarm = init_swarm()

    # main loop
    RUN_COUNT = 1
    while swarm["swarm_best_score"] != TARGET_SCORE:
        swarm = run(swarm)
        print("run: " + str(RUN_COUNT) + ", score: " + str(swarm["swarm_best_score"]))
        RUN_COUNT = RUN_COUNT + 1

    print(swarm["swarm_best_pos"])
    print(swarm["swarm_best_score"])
