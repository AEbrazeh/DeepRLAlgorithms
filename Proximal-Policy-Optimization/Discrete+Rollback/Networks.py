import torch
import torch.nn as nn
import numpy as np
from torch.functional import F

def orth(layer, weightStd=np.sqrt(2), biasValue=0.0):
    """
    Applies orthogonal initialization to the weights of a linear layer.

    Args:
        layer (nn.Linear): The linear layer to be initialized.
        weightStd (float, optional): Standard deviation for weight initialization. Defaults to sqrt(2).
        biasValue (float, optional): Value for initializing the bias term. Defaults to 0.0.

    Returns:
        nn.Linear: The initialized linear layer.
    """
    nn.init.orthogonal_(layer.weight, weightStd)
    nn.init.constant_(layer.bias, biasValue)
    return layer

class baseNetOrthogonal(nn.Module):
    """
    Neural network with orthogonal weight initialization for hidden layers.

    Args:
        layersDim (list): List of layer dimensions (input, hidden1, ..., hiddenN, output).
        finalLayerStd (float): Standard deviation for the final layer weights.
    """
    def __init__(self, layersDim, finalLayerStd):
        super(baseNetOrthogonal, self).__init__()
        self.network = nn.Sequential()
        for ii in range(len(layersDim) - 2):
            self.network.append(orth(nn.Linear(layersDim[ii], layersDim[ii+1])))
            self.network.append(nn.Tanh())
        self.network.append(orth(nn.Linear(layersDim[-2], layersDim[-1]), weightStd=finalLayerStd))
        
    def forward(self, x):
        return self.network(x)
    
class valueNetwork(nn.Module):
    """
    Value network with orthogonal initialization for state value estimation.

    Args:
        stateDim (int): Dimensionality of the state space.
        HiddenDim (int): Number of hidden units in the neural networks.
        numHiddenLayers (int): Number of hidden layers in the neural networks.
    """
    def __init__(self, stateDim, HiddenDim, numHiddenLayers):
        super(valueNetwork, self).__init__()
        layersDim = [stateDim]
        layersDim += [HiddenDim for _ in range(numHiddenLayers)]
        layersDim.append(1)
        self.network = baseNetOrthogonal(layersDim, 1.0)
    
    def forward(self, state):
        """
        Forward pass through the value network.

        Args:
            state (torch.Tensor): Current state.

        Returns:
            torch.Tensor: Estimated state value.
        """
        return self.network(state)
        
    def save(self, file):
        torch.save(self, file)
        
class policyNetwork(nn.Module):
    """
    Policy network with orthogonal initialization for action selection.

    Args:
        stateDim (int): Dimensionality of the state space.
        actionDim (int): Dimensionality of the action space.
        HiddenDim (int): Number of hidden units in the neural networks.
        numHiddenLayers (int): Number of hidden layers in the neural networks.
    """
    def __init__(self, stateDim, actionDim, HiddenDim, numHiddenLayers):
        super(policyNetwork, self).__init__()
        layersDim = [stateDim]
        layersDim += [HiddenDim for _ in range(numHiddenLayers)]
        layersDim.append(actionDim)
        self.network = baseNetOrthogonal(layersDim, 0.01)
        
    def forward(self, state):
        """
        Forward pass through the policy network.

        Args:
            state (torch.Tensor): Current state.

        Returns:
            torch.distributions.Categorical: Action distribution.
        """
        return torch.distributions.Categorical(logits=self.network(state))
    
    def sample(self, state):
        """
        Samples an action from the actor's policy and calculates the log probability.
        Args:
            state (torch.Tensor): State tensor.

        Returns:
            tuple: A tuple containing the sampled action (torch.Tensor) and the log probability (torch.Tensor) of the action.
        """
        dist = self(state)
        action = dist.sample()
        logProb = dist.log_prob(action)
        
        return action, logProb
    
    def save(self, file):
        torch.save(self, file)