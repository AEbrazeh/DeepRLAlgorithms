from Networks import *
from ExperienceReplay import *
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class deepQLearningSingle(nn.Module):
    """
    Implementation of a deep Q-learning agent with a single Q-network based on "Playing Atari with Deep Reinforcement Learning" (https://arxiv.org/abs/1312.5602) by Minh et al.

    Args:
        stateDim (int): Dimensionality of the state space.
        actionDim (int): Dimensionality of the action space.
        hiddenDim (int): Number of units in each hidden layer of the Q-network.
        numHiddenLayers (int): Number of hidden layers in the Q-network.
        bufferSize (int, optional): Size of the experience replay buffer. Default is 100000.
        gamma (float, optional): Discount factor for future rewards. Default is 0.99.
        lr (float, optional): Learning rate for the Q-network optimizer. Default is 1e-4.
    """
    def __init__(self, stateDim,
                 actionDim,
                 hiddenDim,
                 numHiddenLayers,
                 bufferSize = 100000,
                 gamma=0.99,
                 lr = 1e-4):
        super(deepQLearningSingle, self).__init__()
        
        self.actionDim = actionDim
        self.gamma = gamma
        
        self.critic = qNetwork(stateDim, actionDim, hiddenDim, numHiddenLayers).to(device)
        self.criticOptim = torch.optim.Adam(self.critic.parameters(), lr)
        self.expReplay = ExperienceReplay(bufferSize, stateDim, 1)
        
    def updateCritic(self, s, a, r, d, s_):
        """
        Update the Q-network's parameters using a single step of gradient descent.

        Args:
            s (torch.Tensor): Current state tensor.
            a (torch.Tensor): Action tensor.
            r (torch.Tensor): Reward tensor.
            d (torch.Tensor): Done tensor.
            s_ (torch.Tensor): Next state tensor.
        """
        self.criticOptim.zero_grad(True)
        targetQ = (r + self.gamma * self.critic(s_).max(-1, keepdim=True)[0] * (1 - d)).squeeze()
        tde = (targetQ - self.critic(s)[torch.arange(len(a)), a.squeeze()]).abs()
        L = tde.pow(2).mean()
        L.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.criticOptim.step()
    
    @torch.no_grad()
    def __call__(self, s, eps = 0.0):
        """
        Select an action given the current state.

        Args:
            s (numpy.ndarray): Current state.
            eps (float, optional): Epsilon value for epsilon-greedy action selection. Default is 0.0.

        Returns:
            int: Selected action.
        """
        s = torch.from_numpy(s).to(device=device, dtype=torch.float32)
        a = self.critic(s).argmax()
        probs = (1-eps) * torch.ones(self.actionDim) / self.actionDim
        probs[a] += eps
        return torch.distributions.Categorical(probs=probs).sample().cpu().numpy()
    
    def store(self, s, a, r, d, s_):
        """
        Store a transition tuple in the experience replay buffer.

        Args:
            s (torch.Tensor): Current state tensor.
            a (torch.Tensor): Action tensor.
            r (torch.Tensor): Reward tensor.
            d (torch.Tensor): Done tensor.
            s_ (torch.Tensor): Next state tensor.
        """
        self.expReplay.store(s, a, r, d, s_)
    
    def learn(self, minibatchSize, numMiniBatch):
        """
        Sample transitions from the replay buffer and update the Q-network.

        Args:
            minibatchSize (int): Size of each mini-batch.
            numMiniBatch (int): Number of mini-batches to sample.
        """
        state, action, reward, done, state_ = self.expReplay.sample(minibatchSize, numMiniBatch)
        for jj in range(numMiniBatch):
            s = state[jj].to(device)
            a = action[jj].to(device)
            r = reward[jj].to(device)
            d = done[jj].to(device)
            s_ = state_[jj].to(device)
            self.updateCritic(s, a, r, d, s_)
            
    def save(self, file):
        """
        Save the Q-network's parameters to a file.
        Args:
            file (str): File path (without extension) to save the parameters.
        """
        self.critic.save(file+'.pth')
        
    def load(self, file):
        """
        Load the Q-network's parameters from a file.
        Args:
            file (str): File path (without extension) from which to load the parameters.
        """
        self.critic = torch.load(file+'.pth')