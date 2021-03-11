"""Module for holding the swarm builder."""

import random

from objects import swarm_object as swarm_obj


class SwarmBuilder:
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

        for _ in range(self.particle_amount):
            particle_to_add = self._build_particle(min_pos, max_pos)
            swarm.add_particle(particle_to_add)

        swarm.swarm_best_pos = swarm.particles[0]["curr_pos"]
        swarm.swarm_best_score = swarm.particles[0]["curr_score"]

        return swarm

    @classmethod
    def _build_particle(cls, min_pos, max_pos):
        dimensions = 3
        r_velocity = random.randint(-1, 1)
        while r_velocity == 0:
            r_velocity = random.randint(-1, 1)

        pos = []
        for i in range(dimensions):
            r = random.randrange(min_pos, max_pos)
            pos.append(r)
        return {
            "name": "particle: " + str(random.randint(0, 99999)),
            "curr_pos": pos,
            "curr_score": None,
            "best_score": None,
            "best_pos": pos,
            "velocity": r_velocity,
        }
