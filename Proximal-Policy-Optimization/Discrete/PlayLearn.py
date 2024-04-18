from Agent import *
import gymnasium as gym

env = gym.make('CartPole-v1')#, render_mode="human")
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n

agent = ppoClipDiscrete(stateDim=stateDim,
                        actionDim=actionDim,
                        hiddenDim=32,
                        numHiddenLayers=2,
                        eps=0.1,
                        policyLr=1e-4,
                        valueLr=1e-3)
#agent.load('best')
print("Model built successfully! Actor Parameters = {}, Critics Parameters = {}".format(sum(p.numel() for p in agent.actor.parameters() if p.requires_grad), 2 * sum(p.numel() for p in agent.critic.parameters() if p.requires_grad)))

tGame = 500
nGames = 100
minibatchSize = 32
learningFreq = 128
nEpoch = 20
rewardHistory = np.zeros(nGames)
bestScore = -np.inf
jj = 0
for ii in range(nGames):
    s_, _ = env.reset()
    d = False
    while not d:
        jj += 1
        s = s_
        a, p = agent(s)
        s_, r, truncated, terminated, _ = env.step(a)
        d = max(truncated, terminated)
        agent.store(s, a, r, s_, d, p, jj % learningFreq == 0)
        rewardHistory[ii] += r
        if jj % learningFreq == 0:
            agent.learn(nEpoch, minibatchSize)
            agent.clear()
            jj = 0
        print("*PLAYING* [Game #{:05d}] Total reward = {:.4f}".format(ii+1, rewardHistory[ii]), end='\r', flush=True)
    print("*PLAYING* [Game #{:05d}] Total reward = {:.4f}".format(ii+1, rewardHistory[ii]))
    if rewardHistory[ii] >= bestScore:
        bestScore = rewardHistory[ii]
        agent.save('best')
    agent.save('last')
