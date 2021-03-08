import objects.Swarm as Swarm
import  objects.Particle as Particle
import random
import math

class SwarmBuilder:
    #default values, can be changed
    particleAmount = 10

    #non-required args
    def buildSwarm(self, minValue, maxValue, particleAmount=None):
        if particleAmount is not None:
            self.particleAmount = particleAmount

        swarm = Swarm.Swarm([], [], None)

        [swarm.addParticle(self.buildParticle(minValue, maxValue)) for i in range(self.particleAmount)]

        swarm.swarmBestPos = swarm.particles[0].currPos
        swarm.swarmBestScore = swarm.particles[0].currScore

        return swarm

    def buildParticle(self, minValue, maxValue):
        randVelocity = random.randint(-1, 1)
        pos = [random.randrange(minValue,maxValue)]
        particle = Particle.Particle(pos, [], randVelocity, None, None)
        return particle