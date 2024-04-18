from Agent import *
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('Pendulum-v1', render_mode="human")
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.shape[0]
agent = softActorCriticContinuousPER(stateDim=stateDim,
                                     actionDim=actionDim,
                                     actionMin=-1,
                                     actionMax=1,
                                     hiddenDim=128,
                                     numHiddenLayers=6,
                                     alpha = 0.2,
                                     criticLr=6e-4,
                                     actorLr=1e-4,
                                     alphaLr=1e-4)
agent.load('last')

episodeLength = 1001
rewardHist = np.zeros(episodeLength)

numSteps = 0
s_, _ = env.reset()
d = False
ii = 0
while not d:
    ii += 1
    s = s_
    a = agent(s)
    s_, r, terminated, truncated, _ = env.step(a)
    d = max(truncated, terminated)
    agent.store(s, a, r, d, s_)
    rewardHist[ii] = r
    print("----------[ Playing Phase: Step: {:03}, Reward: {:.3f}]----------".format(ii+1, rewardHist[:ii+1].sum()), end='\r', flush=True)