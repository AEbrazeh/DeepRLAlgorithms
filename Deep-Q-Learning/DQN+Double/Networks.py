import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

class baseNet(nn.Module):
    def __init__(self, layersDim):
        """
        Initializes a simple feed forward neural network.

        Parameters:
            layersDim (list): The dimensions of the layers.
        Returns:
            None
        """
        super(baseNet, self).__init__()
        self.network = nn.Sequential()
        for ii in range(len(layersDim) - 2):
            self.network.append(nn.Linear(layersDim[ii], layersDim[ii+1]))
            self.network.append(nn.LeakyReLU(0.01))
        self.network.append(nn.Linear(layersDim[-2], layersDim[-1]))
        
    def forward(self, x):
        return self.network(x)
    
class qNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDim, numHiddenLayers):
        """
        Initialize the Q-network with the given state dimension, hidden dimension, and number of hidden layers.
        """
        super(qNetwork, self).__init__()
        layersDim = [stateDim]
        layersDim += [hiddenDim for _ in range(numHiddenLayers)]
        layersDim.append(actionDim)
        self.network = baseNet(layersDim)

    def forward(self, state):
        return self.network(state)
        
    def save(self, file):
        torch.save(self, file)