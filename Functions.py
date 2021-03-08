targetScore = 0

def currentScoreIsBetterThanBestScore(currentScore, bestScore):
    if(bestScore is None 
    or ((currentScore < bestScore and currentScore >= targetScore) 
    or (currentScore <= targetScore and currentScore > bestScore))):
        return True
    return False