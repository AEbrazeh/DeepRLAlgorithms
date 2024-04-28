from Agent import *
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('CartPole-v1')#, render_mode="human")
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n
agent = deepQLearningDouble(stateDim, actionDim, hiddenDim=32, numHiddenLayers=2, lr=1e-3)

numEpisodes = 100
batchSize = 512
minibatchSize = 64
numMiniBatch = np.ceil(batchSize/minibatchSize).astype(int)
startSteps = 256
rewardHist = np.zeros(numEpisodes)

rewardHist = np.zeros(numEpisodes)

numSteps = 0
bestScore = -np.inf
for ii in range(numEpisodes):
    s_, _ = env.reset()
    d = False
    jj = 0
    while not d:
        numSteps += 1
        s = s_
        eps = np.clip((numSteps - startSteps)/1000, 0, 1)
        a = agent(s, eps)
        s_, r, terminated, truncated, _ = env.step(a)
        d = max(truncated, terminated)
        agent.store(s, a, r, d, s_)
        rewardHist[ii] += r
        jj += 1
        if numSteps >= batchSize:
            agent.learn(minibatchSize, numMiniBatch)
        print("----------[Episode: {:04}/{:04}, Step: {:04}, Reward: {:.3f}, Best Reward: {:04.3f} ]----------".format(ii+1, numEpisodes, jj, rewardHist[ii], bestScore), end='\r', flush=True)
    if rewardHist[ii].sum(-1) >= bestScore:
        agent.save('best')
        bestScore = rewardHist[ii].sum(-1)
    print("----------[Episode: {:04}/{:04}, Step: {:04}, Reward: {:.3f}, Best Reward: {:.3f} ]----------".format(ii+1, numEpisodes, jj, rewardHist[ii], bestScore))
    agent.save('last')
env.close()
    
np.savetxt('rewardHist.csv', rewardHist)
env.close()