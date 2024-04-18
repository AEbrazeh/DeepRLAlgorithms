from Networks import *
from ExperienceReplay import *
import copy

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class softActorCriticContinuous(nn.Module):
    def __init__(self, stateDim,
                 actionDim,
                 hiddenDim,
                 numHiddenLayers,
                 actionMin = -1,
                 actionMax = 1,
                 bufferSize = 100000,
                 gamma=0.99,
                 alpha=0.2,
                 tau=0.01,
                 criticLr = 1e-3,
                 actorLr=1e-4,
                 alphaLr=3e-5):
        super(softActorCriticContinuous, self).__init__()
        
        '''
        This code defines the softActorCritic class, which implements the Soft Actor Critic (SAC) algorithm for environments with continuous action-space.
        It is based on the paper "Soft Actor-Critic Algorithms and Applications" by Haarnoja et al. (https://arxiv.org/abs/1812.05905)
        
        Args:
        stateDim (int): Dimensionality of the agent's state space.
        actionDim (int): Dimensionality of the agent's action space.
        hiddenDim (int): Number of hidden units per layer in the actor and critic networks.
        numHiddenLayers (int): Number of hidden layers in the actor and critic networks.
        actionMin (float, optional): Minimum value allowed for actions (defaults to -1).
        actionMax (float, optional): Maximum value allowed for actions (defaults to 1).
        bufferSize (int, optional): Size of the experience replay buffer (defaults to 100,000).
        gamma (float, optional): Discount factor for future rewards (defaults to 0.99).
        alpha (float, optional): Initial value for the entropy coefficient (defaults to 0.2).
        tau (float, optional): Target update parameter for the critic networks (defaults to 0.01).
        criticLr (float, optional): Learning rate for the critic networks (defaults to 1e-3).
        actorLr (float, optional): Learning rate for the actor network (defaults to 1e-4).
        alphaLr (float, optional): Learning rate for the entropy coefficient (defaults to 3e-5).
        '''

        self.actionMin = actionMin
        self.actionMax = actionMax
        
        self.gamma = gamma
        self.tau = tau
        self.entropy_ = -actionDim
        
        self.critic = doubleCriticNetwork(stateDim, actionDim, hiddenDim, numHiddenLayers).to(device)
        self.critic_ = copy.deepcopy(self.critic).requires_grad_(False).to(device)
        self.criticOptim = torch.optim.Adam(self.critic.parameters(), criticLr)
        
        self.actor = actorNetwork(stateDim, actionDim, hiddenDim, numHiddenLayers).to(device)
        self.actorOptim = torch.optim.Adam(self.actor.parameters(), actorLr)
        
        self.logAlpha = nn.Parameter(torch.tensor([alpha]).log().to(device), requires_grad=True)
        self.alphaOptim = torch.optim.Adam([self.logAlpha], alphaLr)
        
        self.expReplay = ExperienceReplay(bufferSize, stateDim, actionDim)
        
        
    def updateCritic(self, s, a, r, d, s_):
        '''
        Updates the critic networks by minimizing the temporal difference error (TD error) between the predicted Q-values and the target Q-values.
        Args:
            s (torch.Tensor): Batch of states from the experience buffer.
            a (torch.Tensor): Batch of actions from the experience buffer.
            r (torch.Tensor): Batch of rewards from the experience buffer.
            d (torch.Tensor): Batch of done flags (0 for non-terminal, 1 for terminal) from the experience buffer.
            s_ (torch.Tensor): Batch of next states from the experience buffer.
        -------------------------------------
        v_ = min_i Q_i(s_, a_) - α * log(π(a_ | s_))
        Target Q-value: q_ = r + γ * v_ * (1 - d) (where γ is the discount factor)
        TD error: tde = |q - q_|
        Loss function: L = MSE(TDError)
        '''
        self.criticOptim.zero_grad(True)
        with torch.no_grad():
            a_, logProb_ = self.actor.sample(s_, self.actionMin, self.actionMax, grad=False)
            v_ = (self.critic_(s_, a_).min(-1, keepdims=True)[0] - self.logAlpha.exp() * logProb_)
            q_ = r + self.gamma * v_ * (1 - d)
        q = self.critic(s, a)
        tde = (q - q_).abs().mean(-1, keepdims=True)
        L = tde.pow(2).mean()
        L.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.criticOptim.step()
    
    def updateActor(self, s):
        '''
        Updates the actor network based on the current state (s).

        Args:
            s (torch.Tensor): State tensor.
        -------------------------------------
        L = (α * log(π(a | s)) - Q(s, a))
        '''
        self.actorOptim.zero_grad(True)
        a, logProb = self.actor.sample(s, self.actionMin, self.actionMax, grad=True)
        q = self.critic(s, a).min(-1)[0]
        L = ((self.logAlpha.exp() * logProb) - q).mean()
        L.backward()
        nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actorOptim.step()
    
    def updateAlpha(self, state):
        '''
        Updates the entropy coefficient (alpha) based on the current state (state).

        Args:
            state (torch.Tensor): State tensor.
        -------------------------------------
        L = -α * (log(π(a | s)) - H0)   Where H0 is target entropy (-actionDim for Continuous action spaces)
        
        '''
        self.alphaOptim.zero_grad(True)
        _, logProb = self.actor.sample(state, self.actionMin, self.actionMax, grad=False)
        L = -(self.logAlpha.exp() * (logProb + self.entropy_)).mean()
        L.backward()
        nn.utils.clip_grad.clip_grad_norm_([self.logAlpha], max_norm=1.0)
        self.alphaOptim.step()

    @torch.no_grad()
    def updateTargets(self):
        '''
        Updates the target critic networks using a polyak update (soft update).
        -------------------------------------
        θ'_t = τ * θ_t + (1 - τ) * θ'_τ, Where τ is the polyak update rate.
        '''
        for p_, p in zip(self.critic_.parameters(), self.critic.parameters()):
            p_.data.copy_(p.data * self.tau + p_.data * (1 - self.tau))
    
    @torch.no_grad()
    def __call__(self, s):
        '''
        Selects an action for the given state (s).
        Args:
            s (numpy.ndarray): State as a numpy array.
        Returns:
            numpy.ndarray: Selected action as a numpy array.    
        -------------------------------------
        a ~ π(s, θ) Where ~ means a sample from the distribution.
        '''
        s = torch.from_numpy(s[None]).to(device=device, dtype=torch.float32)
        a, _ = self.actor.sample(s, self.actionMin, self.actionMax, grad=False)
        return a[0].cpu().numpy()
    
    def store(self, s, a, r, d, s_):
        '''
        Stores a transition (s, a, r, d, s_) in the experience replay buffer.
        Args:
            s (numpy.ndarray): State as a numpy array.
            a (numpy.ndarray): Action as a numpy array.
            r (float): Reward.
            d (bool): Done flag.
            s_ (numpy.ndarray): Next state as a numpy array.
        '''
        self.expReplay.store(s, a, r, d, s_)
    
    def learn(self, minibatchSize, numMiniBatch):
        '''
        Performs a learning step by updating the actor, critic, alpha, and target networks based on sampled transitions from the experience replay buffer.
        Args:
            minibatchSize (int): Size of each minibatch.
            numMiniBatch (int): Number of minibatches to sample and learn from.
        '''
        state, action, reward, done, state_ = self.expReplay.sample(minibatchSize, numMiniBatch)
        for jj in range(numMiniBatch):
            s = state[jj].to(device)
            a = action[jj].to(device)
            r = reward[jj].to(device)
            d = done[jj].to(device)
            s_ = state_[jj].to(device)
            
            self.updateCritic(s, a, r, d, s_)
            self.updateActor(s)
            self.updateAlpha(s)
            self.updateTargets()
    
    def save(self, file):
        '''
        Saves the actor and critic network weights to the specified file.
        Args:
            file (str): Base filename for saving the networks.
        '''
        self.critic.save(file+'Critic.pth')
        self.actor.save(file+'Actor.pth')
        
    def load(self, file):
        '''
        Loads the actor and critic network weights from the specified file.
        Args:
            file (str): Base filename for loading the networks.
        '''
        self.critic = torch.load(file+'Critic.pth')
        self.actor = torch.load(file+'Actor.pth')
        