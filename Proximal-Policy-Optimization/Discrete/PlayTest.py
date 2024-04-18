from Agent import *
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n

agent = ppoClipDiscrete(stateDim=stateDim,
                        actionDim=actionDim,
                        hiddenDim=32,
                        numHiddenLayers=2,
                        eps=0.1,
                        policyLr=1e-4,
                        valueLr=1e-3)

agent.load('best')

print("Model built successfully! Actor Parameters = {}, Critics Parameters = {}".format(sum(p.numel() for p in agent.policyNet.parameters() if p.requires_grad), 2 * sum(p.numel() for p in agent.valueNet.parameters() if p.requires_grad)))

reward = 0
s_, _ = env.reset()
d = False
while not d:
    s = s_
    a, p = agent(s)
    s_, r, truncated, terminated, _ = env.step(a)
    reward += r
    d = max(truncated, terminated)
print("Total Reward : {}".format(reward))
