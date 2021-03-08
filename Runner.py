import objects.Swarm as Swarm
import objects.Particle as Particle
import ray
import sphere_function
import random
import Functions

ray.init()

targetScore = 3000

@ray.remote
def calculateScore(particle):
    score = sphere_function.sphere_pp(particle.currPos)
    particle.currScore = score

    if(Functions.currentScoreIsBetterThanBestScore(particle.currScore, particle.bestScore)):
        particle.bestScore = particle.currScore
        particle.bestPos = particle.currPos
        print(particle)

    
    return particle

@ray.remote
def updateParticlePosition(particle, swarmBestPos):
    #move particles currPos # https://gyazo.com/b52c066fa8aa53bc68e9e161f650c289
    r = random.randint(1,3)
    e = (r*particle.velocity)
    if(particle.bestPos[0] != particle.currPos[0]):
        e += r* ( particle.bestPos[0] - particle.currPos[0] )
    else:
        e += particle.currPos[0]
    if(particle.currPos[0] != swarmBestPos[0]):
        e += r* (swarmBestPos[0] - particle.currPos[0])
    particle.currPos[0] = e
    #calculate scores

    return particle

def calculateScoresForSwarm(swarm):
    rayObjRefs = [calculateScore.remote(particle) for particle in swarm.particles]

    scoredParticles = ray.get(rayObjRefs)

    swarm.particles = scoredParticles

    return swarm

def updateSwarmPositions(swarm):
    swarmBestPos = swarm.swarmBestPos
    rayObjRefs = [updateParticlePosition.remote(particle, swarmBestPos) for particle in swarm.particles]

    updatedSwarmPositions = ray.get(rayObjRefs)

    swarm.particles = updatedSwarmPositions

    return swarm
