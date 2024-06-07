import torch
import numpy as np

class PrioritizedExperienceReplay:
    def __init__(self, bufferSize, stateDim, actionDim, alpha = 1.0, beta = 1.0, deltaBeta = 1e-3):
        """
        Implements prioritized experience replay (PER, based on paper "Priotorized Experience Replay" by Schaul et al. (https://arxiv.org/abs/1511.05952)) for storing and sampling transitions during agent training.
        This class maintains a buffer to store experience tuples (s, a, r, d, s_) encountered during agent interaction with the environment.
        These tuples represent state (s), action (a), reward (r), done flag (d), and next state (s_).
        PER prioritizes transitions based on their TD-errors, allowing the agent to focus on learning from informative experiences.
        
        This class prioritizes transitions based on their absolute TD-errors raised to the power of the `alpha` parameter.
        The `probs` tensor holds the probability of sampling each transition during minibatch creation, considering their priorities.
        The `isw` tensor holds importance sampling weights to compensate for the prioritization during experience replay.
        The `beta` parameter controls the strength of importance sampling. It's initially set to `beta` and gradually increases towards 1 during training (annealing).

        Attributes:
            bufferSize (int): Size of the experience replay buffer.
            bufferHead (int): Index pointing to the next insertion position in the buffer.
            stateDim (int): Dimensionality of the agent's state space.
            actionDim (int): Dimensionality of the agent's action space.
            alpha (float): PER parameter controlling the focus on higher priority transitions (defaults to 1.0). Values between 0 and 1.
            beta (float, optional): PER parameter controlling importance sampling (defaults to 1.0). Values between 0 and 1. Annealed over training.
            deltaBeta (float, optional): PER parameter for updating beta over training (defaults to 1e-3).
            state (torch.Tensor): Tensor to store states (shape: (bufferSize, stateDim)).
            action (torch.Tensor): Tensor to store actions (shape: (bufferSize, actionDim)).
            reward (torch.Tensor): Tensor to store rewards (shape: (bufferSize, 1)).
            done (torch.Tensor): Tensor to store done flags (shape: (bufferSize, 1)).
            state_ (torch.Tensor): Tensor to store next states (shape: (bufferSize, stateDim)).
            tdError (torch.Tensor): Tensor to store TD errors for each transition (shape: (bufferSize, 1)).
            probs (torch.Tensor): Tensor to store probabilities of sampling each transition based on priorities (shape: (bufferSize, 1)).
            isw (torch.Tensor): Tensor to store importance sampling weights for each transition (shape: (bufferSize, 1)).

        """
        self.alpha = alpha
        self.beta = beta
        self.deltaBeta = deltaBeta
        self.bufferSize = bufferSize
        self.bufferHead = 0
        
        self.actionDim = actionDim
        
        self.state = torch.zeros((bufferSize, stateDim), dtype=torch.float32)
        self.action = torch.zeros((bufferSize, actionDim), dtype=torch.int32)
        self.reward = torch.zeros((bufferSize, 1), dtype=torch.float32)
        self.done = torch.zeros((bufferSize, 1), dtype=torch.float32)
        self.state_ = torch.zeros((bufferSize, stateDim), dtype=torch.float32)
        self.tdError = torch.ones((bufferSize, 1), dtype=torch.float32) * 1e-9
        self.probs = torch.zeros((bufferSize, 1), dtype=torch.float32)
        self.isw = torch.zeros((bufferSize, 1), dtype=torch.float32)
        
    def updateProbs(self):
        '''
        Updates probabilities and importance sampling weights (ISWs) for transitions in the prioritized experience replay (PER) buffer.
        This function is called before sampling minibatch transitions to prioritize those with higher learning potential.
        
        Process:
        1. Calculates probabilities for transitions in the buffer up to the current buffer head (`self.bufferHead`) using the formula: `probs = td_error ^ alpha`.
        2. Normalizes the probabilities to sum to 1, ensuring they represent a valid probability distribution.
        3. Calculates importance sampling weights (ISWs) using the formula: `isw = 1 / (m * probs) ^ beta`, where `m` is the number of transitions in the buffer (or up to the buffer head).
        4. Normalizes the ISWs by dividing by their maximum value to prevent numerical issues.
        5. Updates the `beta` parameter by adding a small increment (`deltaBeta * (1 - beta)`) to gradually increase the importance of sampling from all transitions over time.
        '''
        m = min(self.bufferHead, self.bufferSize)
        rank = 1/(1 + torch.sort(self.tdError[:m], dim=0)[1])
        self.probs[:m] = rank**self.alpha
        self.probs = self.probs / self.probs.sum()
        
        self.isw[:m] = 1 / (m * self.probs[:m])**self.beta
        self.isw = self.isw / self.isw.max()
        
        self.beta += self.deltaBeta * (1 - self.beta)
        
    def store(self, state, action, reward, done, state_):
        """
        Stores a transition (experience tuple) in the experience replay buffer.
        Args:
            state (numpy.ndarray): Current state.
            action (numpy.ndarray): Action taken in the current state.
            reward (float): Reward received for taking the action.
            done (bool): Done flag indicating the end of an episode.
            state_ (numpy.ndarray): Next state reached after taking the action.
        """
        ii = self.bufferHead % self.bufferSize
        
        self.state[ii] = torch.from_numpy(state)
        self.action[ii] = torch.from_numpy(action)
        self.reward[ii] = reward
        self.done[ii] = done
        self.state_[ii] = torch.from_numpy(state_)
        self.tdError[ii] = self.tdError.max(0)[0]
        self.bufferHead += 1
        
    def sample(self, minibatchSize, numMinibatch):
        """
        Samples a minibatch of transitions from the prioritized experience replay (PER) buffer.
        This function prioritizes transitions during sampling based on their pre-calculated probabilities and importance sampling weights (ISWs).
        
        Probability-Proportional Sampling Transitions are sampled with a probability proportional to their calculated probabilities (updated by `updateProbs`). Transitions with higher probabilities have a greater chance of being selected.
        Importance Sampling Weights (ISWs): The returned minibatch also includes the ISWs for the sampled transitions. These weights are used during minibatch learning to compensate for the introduced bias from probability-proportional sampling and ensure unbiased learning.

        Args:
            minibatchSize (int): Size of each minibatch.
            numMinibatch (int): Number of minibatches to sample.

        Returns:
            tuple: A tuple containing the sampled minibatch of states, actions,
                rewards, done flags, next states and importance sampling weights. Each element is a 3D tensor
                with dimensions (numMinibatch, minibatchSize, *element_shape).
        """
        batchSize = int(minibatchSize * numMinibatch)
        self.updateProbs()
        m = min(self.bufferHead, self.bufferSize)
        ii = np.random.choice(m, batchSize, p=self.probs[:m, 0].numpy())
        self.lastSelected = ii.reshape((numMinibatch, minibatchSize))
        
        state = self.state[ii]
        action = self.action[ii]
        reward = self.reward[ii]
        done = self.done[ii]
        state_ = self.state_[ii]
        isw = self.isw[ii]
        return (state.reshape(numMinibatch, minibatchSize, *state.shape[1:]),
                action.reshape(numMinibatch, minibatchSize, *action.shape[1:]),
                reward.reshape(numMinibatch, minibatchSize, *reward.shape[1:]),
                done.reshape(numMinibatch, minibatchSize, *reward.shape[1:]),
                state_.reshape(numMinibatch, minibatchSize, *state_.shape[1:]),
                isw.reshape(numMinibatch, minibatchSize, *isw.shape[1:]))
    
    def updateBuffer(self, tdError, index):
        '''
        Updates the temporal difference errors (TD errors) for the transitions sampled in the latest minibatch.
        This function updates the TD errors in the prioritized experience replay (PER) buffer for the transitions that were most recently sampled for learning (using the `sample` function).
        '''
        self.tdError[self.lastSelected[index]] = tdError