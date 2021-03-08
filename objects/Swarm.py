class Swarm:
    def __init__(self, particles, swarmBestPos, swarmBestScore):
        self.particles = particles
        self.swarmBestPos = swarmBestPos
        self.swarmBestScore = swarmBestScore

    def __str__(self):
        return str(self.__dict__)

    
    def addParticle(self, particle):
        self.particles.append(particle)