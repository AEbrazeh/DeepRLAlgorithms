from Networks import *
from Utility import *
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
Proximal Policy Optimization with Clipped Surrogate (PPO-Clip) with Generalized Advantage Estimation (GAE) and Sample Reuse for environments with discrete action-space.
Based on "Proximal Policy Optimization Algorithms" (https://arxiv.org/abs/1707.06347) by John Schulman et al.,
"High-Dimensional Continuous Control Using Generalized Advantage Estimation" (https://arxiv.org/abs/1506.02438) by John Schulman et al.
"Generalized Proximal Policy Optimization with Sample Reuse" (https://arxiv.org/abs/2111.00072) by James Queeney et al.

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
class ppoClipDiscreteSampleReuse(nn.Module):
    def __init__(self, stateDim = 96,
                 actionDim = 19,
                 hiddenDim = 256,
                 numHiddenLayers=2,
                 eps = 0.2,
                 gamma=0.99,
                 lambda_ = 0.9,
                 valueLr=2.5e-4,
                 policyLr=2.5e-4):
        super(ppoClipDiscreteSampleReuse, self).__init__()
        
        self.state = []
        self.action = []
        self.reward = []
        self.done = []
        self.prob = []
        self.nUpdate = []
        
        self.gamma = gamma
        self.lambda_ = lambda_
        self.eps = eps
        
        self.critic = criticNetwork(stateDim, hiddenDim, numHiddenLayers).to(device)
        self.criticOptim = torch.optim.Adam(self.critic.parameters(), valueLr, eps=1e-5)        
        self.actor = actorNetwork(stateDim, actionDim, hiddenDim, numHiddenLayers).to(device)
        self.actorOptim = torch.optim.Adam(self.actor.parameters(), policyLr, eps=1e-5)

    def store(self, nu, s, a, r, s_, d, p, isLastStep):
        """
        Stores the transition information (state, action, reward, next state, done flag, and action probability).

        Args:
            nu (int): Number of updates done before this transition.
            s (numpy.ndarray): Current state.
            a (numpy.ndarray): Chosen action.
            r (float): Reward received.
            s_ (numpy.ndarray): Next state.
            d (bool): Whether the episode terminated.
            p (numpy.ndarray): Probability of choosing the action.
            isLastStep (bool): Whether the current step is the last step before the update.
        """
        self.nUpdate.append(nu)
        self.state.append(s)
        self.action.append(a)
        self.reward.append(r)
        self.done.append(d)
        self.prob.append(p)
        if isLastStep or d:
            self.state.append(s_)

    def updatePolicy(self, s, a, p, p_, adv):
        """
        Updates the policy network by computing the Generalized PPO loss and performing gradient descent.

        Args:
            s (torch.Tensor): Current state.
            a (torch.Tensor): Chosen action.
            p (torch.Tensor): Probability of choosing the action at the time of choosing.
            p_ (torch.Tensor): Probability of choosing the action right before the update.
            adv (torch.Tensor): Generalized Advantage Estimation (GAE) advantage.
        """
        self.actorOptim.zero_grad(True)
        r = (self.actor(s).log_prob(a) - p).exp()
        r_ = (p_ - p).exp()
        L = -torch.where((r - r_).abs() <= self.eps, r * adv, r.clamp(r_ - self.eps, r_ + self.eps) * adv).mean()
        L.backward()
        self.actorOptim.step()

    def updateValue(self, s, v, adv):
        """
        Updates the value network by computing the value loss and performing gradient descent.

        Args:
            s (torch.Tensor): Current state.
            v (torch.Tensor): Estimated state value.
            adv (torch.Tensor): Generalized Advantage Estimation (GAE) advantage.
        """
        self.criticOptim.zero_grad(True)
        rtg = adv + v
        value = self.critic(s).squeeze()
        L = (value - rtg).pow(2).mean()
        L.backward()
        self.criticOptim.step()
        
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
        action, prob = self.actor.sample(state)
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
        done = np.array(self.done)
        prob = np.array(self.prob)
        
        last = np.zeros_like(done)
        last[-1] = 1

        # Calculate the value function for the next state
        with torch.no_grad():
            value = self.critic(torch.tensor(state, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
        
        # Identify the indices where an episode ends
        doneIndex = np.where(np.max((done, last), axis=0)==1)[0]
        doneIndex = doneIndex + np.arange(len(doneIndex))
        
        # Adjust the value function for the terminal states
        value_ = np.delete(value, (doneIndex+2)%len(value), axis=0)
        value = np.delete(value, doneIndex+1, axis=0)
        state = np.delete(state, doneIndex+1, axis=0)
        
        with torch.no_grad():
            prob_ = self.actor(torch.tensor(state, dtype=torch.float32, device=device)).log_prob(torch.tensor(action, dtype=torch.float32, device=device)).cpu().numpy()
        
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
                p_ = torch.tensor(prob_[start:end], dtype=torch.float32, device=device)
                adv = torch.tensor(advantage[start:end], dtype=torch.float32, device=device)
                #adv = (adv - adv.min()) / (adv.max() - adv.min())

                self.updateValue(s, v, adv)
                self.updatePolicy(s, a, p, p_, adv)
                
    
    def clearEarliestUpdate(self):
        """
        Clears the oldest stored data (states, actions, rewards, probabilities, and done flags).
        """
        nUpdate = np.array(self.nUpdate)
        i = np.where(nUpdate == nUpdate.min())[0]
        s = np.array(self.done[:i.max()+1]).sum()
        del(self.state[:i.max()+1+s],
            self.action[:i.max()+1],
            self.reward[:i.max()+1],
            self.prob[:i.max()+1],
            self.done[:i.max()+1],
            self.nUpdate[:i.max()+1])
                
    def clear(self):
        """
        Clears all the stored data (states, actions, rewards, probabilities, and done flags).
        """
        del(self.state[:],
            self.action[:],
            self.reward[:],
            self.prob[:],
            self.done[:],
            self.nUpdate[:],)
    
    def save(self, file):
        """
        Saves the policy and value networks to files.
        Args:
            file (str): The base file path to which 'Policy.pth' and 'Value.pth' will be appended.
        """
        self.actor.save(file+'Policy.pth')
        self.critic.save(file+'Value.pth')
        
    def load(self, file):
        """
        Loads the policy and value networks from files.
        Args:
            file (str): The base file path from which 'Policy.pth' and 'Value.pth' will be loaded.
        """
        self.actor = torch.load(file+'Policy.pth')
        self.critic = torch.load(file+'Value.pth')