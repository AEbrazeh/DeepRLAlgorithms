from Networks import *
from PrioritizedExperienceReplay import *
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class deepQLearningDoubleDuellingNoisyPER(nn.Module):
    """
    Implementation of a deep Q-learning agent with a duelling double Q-network with Noisy Networks and Prioritized Experience Replay based on "Deep Reinforcement Learning with Double Q-learning" (https://arxiv.org/abs/1509.06461) by van Hasselt et al, "Dueling Network Architectures for Deep Reinforcement Learning" (https://arxiv.org/abs/1511.06581) by Wang et al and "Noisy Networks for Exploration" (https://arxiv.org/pdf/1706.10295) by Fortunato et al.

    Args:
        stateDim (int): Dimensionality of the state space.
        actionDim (int): Dimensionality of the action space.
        hiddenDim (int): Number of units in each hidden layer of the Q-network.
        numHiddenLayers (int): Number of hidden layers in the Q-network.
        bufferSize (int, optional): Size of the experience replay buffer. Default is 100000.
        gamma (float, optional): Discount factor for future rewards. Default is 0.99.
        tau (float, optional): Target update parameter for the critic network. Defaults to 0.01.
        lr (float, optional): Learning rate for the Q-network optimizer. Default is 1e-4.
        bufferAlpha (float, optional): PER parameter controlling the focus on higher priority transitions (defaults to 0.6). Values between 0 and 1.
        bufferBeta (float, optional): PER parameter controlling importance sampling (defaults to 0.4). Values between 0 and 1.
        bufferDeltaBeta (float, optional): PER parameter for updating beta over training (defaults to 1e-3).
        """
    def __init__(self, stateDim,
                 actionDim,
                 hiddenDim,
                 numHiddenLayers,
                 bufferSize = 100000,
                 gamma=0.99,
                 tau=1e-3,
                 lr = 1e-4,
                 bufferAlpha = 0.6,
                 bufferBeta = 0.4,
                 bufferDeltaBeta=1e-3):
        super(deepQLearningDoubleDuellingNoisyPER, self).__init__()
        
        self.actionDim = actionDim
        self.gamma = gamma
        self.tau = tau
        
        self.critic = qNetworkDuelling(stateDim, actionDim, hiddenDim, numHiddenLayers).to(device)
        self.critic_ = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.criticOptim = torch.optim.Adam(self.critic.parameters(), lr)
        self.expReplay = PrioritizedExperienceReplay(bufferSize, stateDim, 1, bufferAlpha, bufferBeta, bufferDeltaBeta)
        
    def updateCritic(self, s, a, r, d, s_, w):
        """
        Update the Q-network's parameters using a single step of gradient descent.

        Args:
            s (torch.Tensor): Current state tensor.
            a (torch.Tensor): Action tensor.
            r (torch.Tensor): Reward tensor.
            d (torch.Tensor): Done tensor.
            s_ (torch.Tensor): Next state tensor.
            w (torch.Tensor): Batch of importance sampling weights from the experience buffer (used in PER).
        """
        self.criticOptim.zero_grad(True)
        a_ = self.critic(s_).argmax(-1, keepdim=True)
        targetQ = (r + self.gamma * self.critic_(s_)[torch.arange(len(a_)), a_.squeeze()].unsqueeze(-1) * (1 - d)).squeeze()
        tde = (targetQ - self.critic(s)[torch.arange(len(a)), a.squeeze()]).abs()
        L = (w * tde.pow(2)).mean()
        L.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.criticOptim.step()
        return tde.unsqueeze(-1).detach().cpu()
    
    @torch.no_grad()
    def updateTarget(self):
        """
        Updates the target critic network using a polyak update (soft update).
        -------------------------------------
        θ'_t = τ * θ_t + (1 - τ) * θ'_τ, Where τ is the polyak update rate.
        """
        for p_, p in zip(self.critic_.parameters(), self.critic.parameters()):
            p_.data.copy_(p.data * self.tau + p_.data * (1 - self.tau))
    
    @torch.no_grad()
    def __call__(self, s):
        """
        Select an action given the current state.

        Args:
            s (numpy.ndarray): Current state.

        Returns:
            int: Selected action.
        """
        s = torch.from_numpy(s).to(device=device, dtype=torch.float32)
        a = self.critic(s).argmax()
        return a.cpu().numpy()
    
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
        Sample transitions from the replay buffer and update the Q-network based on prioritized experience replay (PER) samples minibatch transitions from the experience replay buffer, considering their importance weights.

        Args:
            minibatchSize (int): Size of each mini-batch.
            numMiniBatch (int): Number of mini-batches to sample.
        """
        state, action, reward, done, state_, weight = self.expReplay.sample(minibatchSize, numMiniBatch)
        for jj in range(numMiniBatch):
            s = state[jj].to(device)
            a = action[jj].to(device)
            r = reward[jj].to(device)
            d = done[jj].to(device)
            s_ = state_[jj].to(device)
            w = weight[jj].to(device)
            
            tde = self.updateCritic(s, a, r, d, s_, w)
            self.critic.resetNoise()
            self.critic_.resetNoise()
            self.expReplay.updateBuffer(tde, jj)
        self.updateTarget()
            
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
        self.critic = torch.load(file+'.pth').to(device)
        self.critic_ = copy.deepcopy(self.critic).requires_grad_(False).to(device)