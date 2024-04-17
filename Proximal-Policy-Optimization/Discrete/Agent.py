from Networks import *
from Utility import *
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class vanillaPPOClipDiscrete(nn.Module):
    def __init__(self, stateDim,
                 actionDim,
                 hiddenDim,
                 numHiddenLayers,
                 eps = 0.2,
                 gamma=0.99,
                 lambda_ = 0.9,
                 valueLr=2.5e-4,
                 policyLr=2.5e-4):
        super(vanillaPPOClipDiscrete, self).__init__()
        
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []        
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = eps
        
        self.valueNet = valueNetwork(stateDim, hiddenDim, numHiddenLayers).to(device)
        self.valueOptim = torch.optim.Adam(self.valueNet.parameters(), valueLr, eps=1e-5)        
        self.policyNet = policyNetwork(stateDim, actionDim, hiddenDim, numHiddenLayers).to(device)
        self.policyOptim = torch.optim.Adam(self.policyNet.parameters(), policyLr, eps=1e-5)

    def store(self, s, a, r, s_, d, p, isLastStep):
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(d)
        self.prob.append(p)
        if isLastStep or d:
            self.state.append(s_)

    def updatePolicy(self, s, a, p, adv):
        self.policyOptim.zero_grad(True)
        r = (self.policyNet(s).log_prob(a) - p).exp()
        L = -torch.where((r - 1).abs() <= self.eps, r * adv, r.clamp(1 - self.eps, 1 + self.eps) * adv).mean()
        L.backward()
        self.policyOptim.step()

    def updateValue(self, s, v, adv):
        self.valueOptim.zero_grad(True)
        rtg = adv + v
        value = self.valueNet(s).squeeze()
        L = (value - rtg).pow(2).mean()
        L.backward()
        self.valueOptim.step()
        
    @torch.no_grad()
    def __call__(self, s):
        state = torch.from_numpy(s).to(device)
        policy = self.policyNet(state)
        action = policy.sample()
        prob = policy.log_prob(action)
        return action.detach().cpu().numpy(), prob.detach().cpu().numpy()
    
    def learn(self, nEpoch, batchSize):
        state = np.array(self.state)
        action = np.array(self.action)
        reward = np.array(self.reward)
        #reward = (reward - reward.min()) / (reward.max() - reward.min() + 1e-12)
        done = np.array(self.done)
        prob = np.array(self.prob)
        
        last = np.zeros_like(done)
        last[-1] = 1
        
        with torch.no_grad():
            value = self.valueNet(torch.tensor(state, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
        
        doneIndex = np.where(np.max((done, last), axis=0)==1)[0]
        doneIndex = doneIndex + np.arange(len(doneIndex))
        value_ = np.delete(value, (doneIndex+2)%len(value), axis=0)
        value = np.delete(value, doneIndex+1, axis=0)
        state = np.delete(state, doneIndex+1, axis=0)
        
        onehot = done[::-1].cumsum()[::-1]
        onehot = onehot[0] - onehot
        onehot = onehotEncode(onehot)
        discount = (self.gamma * self.lambda_) ** ((onehot.cumsum(0)-1) * onehot)
        delta = (reward + self.gamma * value_ * (1-done) - value)[:, None] * onehot * discount
        
        advantage = (delta[::-1].cumsum(0)[::-1] * onehot / discount).sum(-1)
        
        numBatch = np.ceil(len(state)/batchSize).astype(int)
        
        for ii in range(nEpoch):
            index = np.arange(len(state))
            np.random.shuffle(index)
            for jj in range(numBatch):
                start = jj * batchSize
                end = (jj+1) * batchSize
                s = torch.tensor(state[start:end], dtype=torch.float32, device=device)
                a = torch.tensor(action[start:end], dtype=torch.int64, device=device)
                v = torch.tensor(value[start:end], dtype=torch.float32, device=device)
                p = torch.tensor(prob[start:end], dtype=torch.float32, device=device)
                adv = torch.tensor(advantage[start:end], dtype=torch.float32, device=device)
                #adv = (adv - adv.min()) / (adv.max() - adv.min())

                self.updateValue(s, v, adv)
                self.updatePolicy(s, a, p, adv)
    
    def clear(self):
        del(self.state[:],
            self.action[:],
            self.reward[:],
            self.prob[:],
            self.done[:])
        
    def save(self, file):
        self.policyNet.save(file+'Policy.pth')
        self.valueNet.save(file+'Value.pth')
        
    def load(self, file):
        self.policyNet = torch.load(file+'Policy.pth')
        self.valueNet = torch.load(file+'Value.pth')