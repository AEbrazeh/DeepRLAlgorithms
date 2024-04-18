from Networks import *
from Utility import *
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ppoClipDiscrete(nn.Module):
    """
    Proximal Policy Optimization with Clipped Surrogate (PPO-Clip) with Generalized Advantage Estimation (GAE) for environments with discrete action-space.
    Based on "Proximal Policy Optimization Algorithms" (https://arxiv.org/abs/1707.06347) by John Schulman et al. and "High-Dimensional Continuous Control Using Generalized Advantage Estimation" (https://arxiv.org/abs/1506.02438) by John Schulman et al.
    
    Args:
        stateDim (int): Dimensionality of the state space.
        actionDim (int): Dimensionality of the action space.
        hiddenDim (int): Number of hidden units in the neural networks.
        numHiddenLayers (int): Number of hidden layers in the neural networks.
        eps (float, optional): Clipping parameter for PPO. Defaults to 0.2.
        gamma (float, optional): Discount factor. Defaults to 0.99.
        lambda_ (float, optional): GAE (Generalized Advantage Estimation) parameter. Defaults to 0.9.
        valueLr (float, optional): Learning rate for the value network. Defaults to 2.5e-4.
        policyLr (float, optional): Learning rate for the policy network. Defaults to 2.5e-4.
    """
    def __init__(self, stateDim,
                 actionDim,
                 hiddenDim,
                 numHiddenLayers,
                 eps = 0.2,
                 gamma=0.99,
                 lambda_ = 0.9,
                 valueLr=2.5e-4,
                 policyLr=2.5e-4):
        super(ppoClipDiscrete, self).__init__()
        
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
        """
        Stores the transition information (state, action, reward, next state, done flag, and action probability).

        Args:
            s (numpy.ndarray): Current state.
            a (numpy.ndarray): Chosen action.
            r (float): Reward received.
            s_ (numpy.ndarray): Next state.
            d (bool): Whether the episode terminated.
            p (numpy.ndarray): Probability of choosing the action.
            isLastStep (bool): Whether the current step is the last step before the update.
        """
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(d)
        self.prob.append(p)
        if isLastStep or d:
            self.state.append(s_)

    def updatePolicy(self, s, a, p, adv):
        """
        Updates the policy network by computing the PPO loss and performing gradient descent.

        Args:
            s (numpy.ndarray): Current state.
            a (numpy.ndarray): Chosen action.
            p (numpy.ndarray): Old action probability.
            adv (numpy.ndarray): Generalized Advantage Estimation (GAE) advantage.
        """
        self.policyOptim.zero_grad(True)
        r = (self.policyNet(s).log_prob(a) - p).exp()
        L = -torch.where((r - 1).abs() <= self.eps, r * adv, r.clamp(1 - self.eps, 1 + self.eps) * adv).mean()
        L.backward()
        self.policyOptim.step()

    def updateValue(self, s, v, adv):
        """
        Updates the value network by computing the value loss and performing gradient descent.

        Args:
            s (numpy.ndarray): Current state.
            v (numpy.ndarray): Estimated state value.
            adv (numpy.ndarray): Generalized Advantage Estimation (GAE) advantage.
        """
        self.valueOptim.zero_grad(True)
        rtg = adv + v
        value = self.valueNet(s).squeeze()
        L = (value - rtg).pow(2).mean()
        L.backward()
        self.valueOptim.step()
        
    @torch.no_grad()
    def __call__(self, s):
        """
        Computes the action and log probability for a given state.

        Args:
            s (np.ndarray): Current state.
        Returns:
            np.ndarray: Chosen action.
            np.ndarray: Log probability of the chosen action.
        """
        state = torch.from_numpy(s).to(device)
        action, prob = self.policyNet.sample(state)
        return action.detach().cpu().numpy(), prob.detach().cpu().numpy()
    
    def learn(self, nEpoch, batchSize):
        """
        Performs training iterations for the PPO agent.

        Args:
            nEpoch (int): Number of training epochs.
            batchSize (int): Batch size for training.
        """
        state = np.array(self.state)
        action = np.array(self.action)
        reward = np.array(self.reward)
        #reward = (reward - reward.min()) / (reward.max() - reward.min() + 1e-12)
        done = np.array(self.done)
        prob = np.array(self.prob)
        
        last = np.zeros_like(done)
        last[-1] = 1
        
        # Calculate the value function for the next state
        with torch.no_grad():
            value = self.valueNet(torch.tensor(state, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
        
        # Identify the indices where an episode ends
        doneIndex = np.where(np.max((done, last), axis=0)==1)[0]
        doneIndex = doneIndex + np.arange(len(doneIndex))
        
        # Adjust the value function for the terminal states
        value_ = np.delete(value, (doneIndex+2)%len(value), axis=0)
        value = np.delete(value, doneIndex+1, axis=0)
        state = np.delete(state, doneIndex+1, axis=0)
        
        # Calculate the one-hot encoding for the done flags
        onehot = done[::-1].cumsum()[::-1]
        onehot = onehot[0] - onehot
        onehot = onehotEncode(onehot)
        
        # Calculate the discount factor for each step
        discount = (self.gamma * self.lambda_) ** ((onehot.cumsum(0)-1) * onehot)
        # Calculate the temporal difference error (delta)
        delta = (reward + self.gamma * value_ * (1-done) - value)[:, None] * onehot * discount
        # Calculate the advantage using the GAE formula
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
        """
        Clears all the stored data (states, actions, rewards, probabilities, and done flags).
        """
        del(self.state[:],
            self.action[:],
            self.reward[:],
            self.prob[:],
            self.done[:])
        
    def save(self, file):
        """
        Saves the policy and value networks to files.
        Args:
            file (str): The base file path to which 'Policy.pth' and 'Value.pth' will be appended.
        """
        self.policyNet.save(file+'Policy.pth')
        self.valueNet.save(file+'Value.pth')
        
    def load(self, file):
        """
        Loads the policy and value networks from files.
        Args:
            file (str): The base file path from which 'Policy.pth' and 'Value.pth' will be loaded.
        """
        self.policyNet = torch.load(file+'Policy.pth')
        self.valueNet = torch.load(file+'Value.pth')