import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

class baseNet(nn.Module):
    """
    A simple feed-forward neural network class.

    This class defines a basic neural network architecture with multiple hidden layers
    using Leaky ReLU activations. It can be used as a building block for more complex
    models.

    Attributes:
        network (nn.Sequential): A sequential container holding the layers of the network.
    """
    def __init__(self, layersDim):
        """
        Initializes the neural network with the specified layer dimensions.
        Args:
            layersDim (list): A list containing the dimensions of each layer in the network.
            The first element represents the input dimension, and the last element
            represents the output dimension. The remaining elements represent the
            number of units in each hidden layer.
        """
        super(baseNet, self).__init__()
        self.network = nn.Sequential()
        for ii in range(len(layersDim) - 2):
            self.network.append(nn.Linear(layersDim[ii], layersDim[ii+1]))
            self.network.append(nn.LeakyReLU(0.01))
        self.network.append(nn.Linear(layersDim[-2], layersDim[-1]))
        
    def forward(self, x):
        """
        Performs a forward pass through the network.
        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Output of the network.
        """
        return self.network(x)
    
class doubleCriticNetwork(nn.Module):
    """
    A neural network architecture for a double Q-critic in some actor-critic algorithms such as TD3 and SAC.

    This class implements a double Q-critic network with two separate critic networks
    (critic1 and critic2) that share the same architecture. Both networks take the state
    and action as input and estimate the state-action value (Q-value).
    these twin critics improve stability during training.

    Attributes:
        critic1 (baseNet): The first critic network.
        critic2 (baseNet): The second critic network.
    """
    def __init__(self, stateDim, actionDim, hiddenDim, numHiddenLayers):
        """
        Initializes the double critic network.
        Args:
            stateDim (int): Dimensionality of the agent's state space.
            actionDim (int): Dimensionality of the agent's action space.
            hiddenDim (int): Number of units in each hidden layer.
            numHiddenLayers (int): Number of hidden layers in the network.
        """
        super(doubleCriticNetwork, self).__init__()
        layersDim = [stateDim + actionDim]
        layersDim += [hiddenDim for _ in range(numHiddenLayers)]
        layersDim.append(1)
        self.critic1 = baseNet(layersDim)
        self.critic2 = baseNet(layersDim)

    def forward(self, state, action):
        """
        Performs a forward pass through both critic networks.
        Args:
            state (torch.Tensor): State tensor.
            action (torch.Tensor): Action tensor.
        Returns:
            torch.Tensor: A concatenated tensor containing the outputs of both critic networks.
        """
        input = torch.cat((state, action), dim=-1)
        return torch.cat((self.critic1(input), self.critic2(input)), dim=-1)
    def save(self, file):
        """
        Saves the entire critic network (both critic1 and critic2) to a file.
        Args:
            file (str): Path to the file for saving the network.
        """
        torch.save(self, file)
        
class actorNetwork(nn.Module):
    """
    A neural network architecture for a stochastic actor in Soft Actor-Critic (SAC).

    This class implements an actor network that takes the state as input and samples
    an action according to a Gaussian stochastic policy.
    The network first outputs the mean (mu) and standard deviation (sigma) of the action distribution,
    which is a Gaussian distribution.
    Then, it samples an action from this distribution and applies a tanh function to constrain
    the action within the specified bounds (a_min, a_max).
    SAC uses this approach to explore the environment while learning the optimal policy.

    Attributes:
        actionDim (int): Dimensionality of the agent's action space.
        network (baseNet): The neural network that maps state to action distribution parameters.

    """
    def __init__(self, stateDim, actionDim, hiddenDim, numHiddenLayers):
        """
        Initializes the actor network.
        Args:
            stateDim (int): Dimensionality of the agent's state space.
            actionDim (int): Dimensionality of the agent's action space.
            hiddenDim (int): Number of units in each hidden layer.
            numHiddenLayers (int): Number of hidden layers in the network.
        """
        super(actorNetwork, self).__init__()
        self.actionDim = actionDim
        layersDim = [stateDim]
        layersDim += [hiddenDim for _ in range(numHiddenLayers)]
        layersDim.append(2 * actionDim)
        self.network = baseNet(layersDim)
    def forward(self, state):
        """
        Performs a forward pass through the network to get action distribution parameters.
        Args:
            state (torch.Tensor): State tensor.
        Returns:
            tuple: A tuple containing the mean (mu) and standard deviation (sigma) of the action distribution.
        """
        mu, logStd = self.network(state).split(self.actionDim, dim = -1)
        std = (logStd.clamp(-10, 1)).exp()
        return mu, std
    
    def sample(self, state, a_min, a_max, grad=True):
        """
        Samples an action from the actor's policy and calculates the log probability.
        Args:
            state (torch.Tensor): State tensor.
            a_min (float): Minimum value for the action range.
            a_max (float): Maximum value for the action range.
            grad (bool, optional): Whether to calculate the gradient of the sampling process.
                Defaults to True.
        Returns:
            tuple: A tuple containing the sampled action (torch.Tensor) and the log probability (torch.Tensor) of the action.
        """
        e = 1e-12
        N = Normal(*self(state))
        
        if grad: a = N.rsample()
        else: a = N.sample()
        
        logProb = (N.log_prob(a) - (1 - a.tanh()**2 + e).log()).sum(-1, keepdim=True)
        action = torch.tanh(a) * (a_max - a_min) / 2 + (a_max + a_min) / 2
        return action, logProb

    def save(self, file):
        """
        Saves the entire actor network to a file.
        Args:
            file (str): Path to the file for saving the network.
        """
        torch.save(self, file)
