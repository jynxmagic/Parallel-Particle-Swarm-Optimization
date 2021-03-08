from random import randint


class Particle:
    def __init__(self, currPos, bestPos,  velocity, currScore, bestScore):
        self.particleName = "particle: "+str(randint(0,90000))
        self.currPos = currPos
        self.bestPos = bestPos
        self.velocity = velocity
        self.currScore = currScore
        self.bestScore = bestScore

    def __str__(self):
        return str(self.__dict__)