"""Module containing the Swarm object."""


class Swarm(object):
    """Swarm containing all active particles."""

    def __init__(self, particles, swarm_best_pos, swarm_best_score):
        """Initialize the particle swarm.

        Args:
            particles (array): Particles within the swarm.
            swarm_best_pos (int): Best position of the swarm.
            swarm_best_score (int): Best score of the swarm.
        """
        self.particles = particles
        self.swarmBestPos = swarm_best_pos
        self.swarmBestScore = swarm_best_score

    def __str__(self):
        """Get dict of swarm data.

        Returns:
            string: description of swarm data.
        """
        return str(self.__dict__)

    def add_particle(self, particle):
        """Add a particle to the swarm.

        Args:
            particle (dict): particle to add.
        """
        self.particles.append(particle)
