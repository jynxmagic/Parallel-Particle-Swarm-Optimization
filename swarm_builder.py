"""Module for holding the swarm builder."""

import random

from objects import particle_object as particle_obj
from objects import swarm_object as swarm_obj


class SwarmBuilder(object):
    """Object used for creating and returning swarms of set size."""

    # default values, can be changed
    particle_amount = 10

    # non-required args

    def build_swarm(self, min_pos, max_pos, particle_amount=None):
        """Builds and returns a swarm object.

        Args:
            min_pos (integer): Min Position of vector
            max_pos (type): Max position of vector
            particle_amount (intger, optional): Amount of particles in swarm.
                Defaults to None.

        Returns:
            Swarm: Instantiated swarm object
        """
        if particle_amount is not None:
            self.particle_amount = particle_amount

        swarm = swarm_obj.Swarm([], [], None)

        [
            swarm.add_particle(self._build_particle(min_pos, max_pos))
            for _i in range(self.particle_amount)
        ]

        swarm.swarm_best_pos = swarm.particles[0].curr_pos
        swarm.swarm_best_score = swarm.particles[0].curr_score

        return swarm

    def _build_particle(self, min_pos, max_pos):
        r_velocity = random.randint(-1, 1)
        pos = [random.randrange(min_pos, max_pos)]
        return particle_obj.Particle(pos, [], r_velocity)