"""Module containing particle object."""


class Particle(object):
    """Holds information about the particle."""

    def __init__(self, curr_pos, best_pos, velocity):
        """Initialize the particle object.

        Args:
            curr_pos (array): Vector describing the current particle
                position.
            best_pos (array): Vector describing the best particle position.
            velocity (int): Speed at which the search range of the particle
                changes each iteration.
        """
        self.curr_pos = curr_pos
        self.best_pos = best_pos
        self.velocity = velocity
        self.curr_score = None
        self.best_score = None

    def __str__(self):
        """Get dict of particle data.

        Returns:
            string: description of particle data.
        """
        return str(self.__dict__)
