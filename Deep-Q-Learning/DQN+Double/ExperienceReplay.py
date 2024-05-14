import torch
import numpy as np

class ExperienceReplay:
    """
    Stores transitions (experience tuples) for training a reinforcement learning agent.

    This class implements a circular buffer to store experience tuples (s, a, r, d, s_)
    encountered during agent interaction with the environment. These tuples represent
    state (s), action (a), reward (r), done flag (d), and next state (s_) and are used
    to train the agent's policy and value function.

    Attributes:
        bufferSize (int): Size of the experience replay buffer.
        bufferHead (int): Index pointing to the next insertion position in the buffer.
        stateDim (int): Dimensionality of the agent's state space.
        actionDim (int): Dimensionality of the agent's action space.
        state (torch.Tensor): Tensor to store states (shape: (bufferSize, stateDim)).
        action (torch.Tensor): Tensor to store actions (shape: (bufferSize, actionDim)).
        reward (torch.Tensor): Tensor to store rewards (shape: (bufferSize, 1)).
        done (torch.Tensor): Tensor to store done flags (shape: (bufferSize, 1)).
        state_ (torch.Tensor): Tensor to store next states (shape: (bufferSize, stateDim)).
        lastSelected (numpy.ndarray, optional): Stores the most recently sampled indices for minibatch creation (updated during `sample`).

    """
    def __init__(self, bufferSize, stateDim, actionDim):
        """
        Initializes the experience replay buffer.
        Args:
            bufferSize (int): Size of the experience replay buffer.
            stateDim (int): Dimensionality of the agent's state space.
            actionDim (int): Dimensionality of the agent's action space.
        """
        self.bufferSize = bufferSize
        self.bufferHead = 0
        
        self.actionDim = actionDim
        
        self.state = torch.zeros((bufferSize, stateDim), dtype=torch.float32)
        self.action = torch.zeros((bufferSize, actionDim), dtype=torch.int32)
        self.reward = torch.zeros((bufferSize, 1), dtype=torch.float32)
        self.done = torch.zeros((bufferSize, 1), dtype=torch.float32)
        self.state_ = torch.zeros((bufferSize, stateDim), dtype=torch.float32)
        
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
        self.bufferHead += 1
        
    def sample(self, minibatchSize, numMinibatch):
        """
        Samples a minibatch of transitions from the experience replay buffer.

        Args:
            minibatchSize (int): Size of each minibatch.
            numMinibatch (int): Number of minibatches to sample.

        Returns:
            tuple: A tuple containing the sampled minibatch of states, actions,
                rewards, done flags, and next states. Each element is a 3D tensor
                with dimensions (numMinibatch, minibatchSize, *element_shape).
        """
        batchSize = int(minibatchSize * numMinibatch)
        m = min(self.bufferHead, self.bufferSize)
        ii = np.random.choice(m, batchSize)
        self.lastSelected = ii.reshape((numMinibatch, minibatchSize))
        
        state = self.state[ii]
        action = self.action[ii]
        reward = self.reward[ii]
        done = self.done[ii]
        state_ = self.state_[ii]
        return (state.reshape(numMinibatch, minibatchSize, *state.shape[1:]),
                action.reshape(numMinibatch, minibatchSize, *action.shape[1:]),
                reward.reshape(numMinibatch, minibatchSize, *reward.shape[1:]),
                done.reshape(numMinibatch, minibatchSize, *reward.shape[1:]),
                state_.reshape(numMinibatch, minibatchSize, *state_.shape[1:]))