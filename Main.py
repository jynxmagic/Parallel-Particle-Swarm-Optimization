import SwarmBuilder
import objects.Swarm as Swarm
import objects.Particle as Particle
import Runner
import Functions

def initSwarm(): #todo configuration
    sb = SwarmBuilder.SwarmBuilder()

    swarm = sb.buildSwarm(1, 1000)

    #calculate particle scores for start pos
    Runner.calculateScoresForSwarm(swarm)

    #calculate swarm best for start pos
    updateSwarmCurrentBestScore(swarm)

    return swarm

def updateSwarmCurrentBestScore(swarm):
    bestScore = swarm.swarmBestScore
    bestPos = swarm.swarmBestPos

    for particle in swarm.particles:
        if (Functions.currentScoreIsBetterThanBestScore(particle.currScore, bestScore)):
            bestScore = particle.currScore
            bestPos = particle.currPos

    swarm.swarmBestScore = bestScore
    swarm.swarmBestPos = bestPos

    return swarm

def run(swarm):
    swarm = Runner.updateSwarmPositions(swarm)
    swarm = Runner.calculateScoresForSwarm(swarm)
    swarm = updateSwarmCurrentBestScore(swarm)


swarm = initSwarm()

#main loop
for i in range(0,100):
    run(swarm)

print(swarm.swarmBestPos[0])
print(swarm.swarmBestScore)