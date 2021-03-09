"""Program entry point."""

import helper
import runner
import swarm_builder


def _init_swarm():  # todo configuration
    sbo = swarm_builder.SwarmBuilder()

    initalized_swarm = sbo.build_swarm(1, 1000)

    # calculate particle scores for start pos
    runner.calculate_scores_for_swarm(initalized_swarm)

    # calculate swarm best for start pos
    _update_swarm_current_best_score(initalized_swarm)

    return initalized_swarm


def _update_swarm_current_best_score(swarm_to_score):
    best_score = swarm_to_score.swarm_best_score
    best_pos = swarm_to_score.swarm_best_pos

    for particle in swarm_to_score.particles:
        if helper.current_score_is_better_than_best_score(
            particle["curr_score"],
            best_score,
        ):
            best_score = particle["curr_score"]
            best_pos = particle["curr_pos"]

    swarm_to_score.swarm_best_score = best_score
    swarm_to_score.swarm_best_pos = best_pos

    return swarm_to_score


def run(swarm_to_run):
    swarm_with_updated_positions = runner.update_swarm_positions(swarm_to_run)
    swarm_with_particles_scored = runner.calculate_scores_for_swarm(
        swarm_with_updated_positions,
    )
    return _update_swarm_current_best_score(
        swarm_with_particles_scored,
    )


if __name__ == "__main__":
    swarm = _init_swarm()

    # main loop
    for _loop in range(0, 100):
        swarm = run(swarm)
        print("run: " + str(_loop) + ", score: " + str(swarm.swarm_best_score))

    print(swarm.swarm_best_pos[0])
    print(swarm.swarm_best_score)
