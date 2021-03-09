"""Program entry point."""
import helper
import runner
import swarm_builder


def _init_swarm():  # todo configuration
    sb = swarm_builder.SwarmBuilder()

    swarm = sb.build_swarm(1, 1000)

    # calculate particle scores for start pos
    runner.calculate_scores_for_swarm(swarm)

    # calculate swarm best for start pos
    _update_swarm_current_best_score(swarm)

    return swarm


def _update_swarm_current_best_score(swarm):
    best_score = swarm.swarm_best_score
    best_pos = swarm.swarm_best_pos

    for particle in swarm.particles:
        if helper.current_score_is_better_than_best_score(
            particle.curr_score, best_score
        ):
            best_score = particle.curr_score
            best_pos = particle.curr_pos

    swarm.swarm_best_score = best_score
    swarm.swarm_best_pos = best_pos

    return swarm


def run(swarm):
    swarm = runner.update_swarm_positions(swarm)
    swarm = runner.calculate_scores_for_swarm(swarm)
    swarm = _update_swarm_current_best_score(swarm)


if __name__ == "__main__":
    swarm = _init_swarm()

    # main loop
    for _loop in range(0, 100):
        run(swarm)
        print("run: " + str(_loop) + ", score: " + str(swarm.swarm_best_score))

    print(swarm.swarm_best_pos[0])
    print(swarm.swarm_best_score)
