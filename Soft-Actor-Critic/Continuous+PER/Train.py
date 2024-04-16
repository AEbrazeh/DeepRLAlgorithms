from Agent import *
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v1')#, render_mode="human")
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]
agent = softActorCritic(stateDim=stateDim,
                        actionDim=actionDim,
                        actionMin=-1,
                        actionMax=1,
                        hiddenDim=128,
                        numHiddenLayers=6,
                        criticLr=6e-4,
                        actorLr=1e-4,
                        alphaLr=1e-4,
                        alpha = 0.2)
#agent.load('last')

numEpisodes = 1000
batchSize = 8192
minibatchSize = 32
numMiniBatch = np.ceil(batchSize/minibatchSize).astype(int)
startSteps = 1024
learningFreq = 10

episodeLength = 1000
rewardHist = np.zeros((numEpisodes, episodeLength))

numSteps = 0
bestScore = -np.inf
for ii in range(numEpisodes):
    s_, _ = env.reset()
    d = False
    jj = 0
    while not d:
        numSteps += 1
        s = s_
        a = env.action_space.sample() if numSteps < startSteps else agent(s)
        s_, r, terminated, truncated, _ = env.step(a)
        d = max(truncated, terminated)
        agent.store(s, a, r, d, s_)
        rewardHist[ii, jj] = r
        jj += 1
        if jj % learningFreq == 0:
            if numSteps >= batchSize:
                agent.learn(minibatchSize, numMiniBatch)
            print("----------[ Playing Phase: Episode: {:04}/{:04}, Step: {:04}, Reward: {:04.3f}, Best Reward: {:04.3f} ]----------".format(ii+1, numEpisodes, jj, rewardHist[ii, :jj+1].sum(), bestScore), end='\r', flush=True)
    if rewardHist[ii].sum(-1) >= bestScore:
        agent.save('best')
        bestScore = rewardHist[ii].sum(-1)
    print("----------[ Playing Phase: Episode: {:04}/{:04}, Step: {:04}, Reward: {:04.3f}, Best Reward: {:04.3f} ]----------".format(ii+1, numEpisodes, jj, rewardHist[ii, :jj+1].sum(), bestScore))
    agent.save('last')
    
np.savetxt('rewardHist.csv', rewardHist)
env.close()