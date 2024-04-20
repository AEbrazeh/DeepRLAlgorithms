from Agent import *
import gymnasium as gym

env = gym.make('CartPole-v1', render_mode="human")
stateDim = env.observation_space.shape[0]
actionDim = env.action_space.n

agent = ppoRollbackDiscreteSampleReuse(stateDim=stateDim,
                                       actionDim=actionDim,
                                       hiddenDim=16,
                                       numHiddenLayers=2,
                                       eps=0.1,
                                       alpha=5e-2,
                                       policyLr=1e-4,
                                       valueLr=1e-3)
agent.load('best')
print("Model built successfully! Actor Parameters = {}, Critics Parameters = {}".format(sum(p.numel() for p in agent.actor.parameters() if p.requires_grad), 2 * sum(p.numel() for p in agent.critic.parameters() if p.requires_grad)))

#Playing and Learning 
s_, _ = env.reset()
d = False
while not d:
    s = s_
    a, p = agent(s)
    s_, r, truncated, terminated, _ = env.step(a)
    d = max(truncated, terminated)