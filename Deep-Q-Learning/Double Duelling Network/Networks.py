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
    
class qNetworkDuelling(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDim, numHiddenLayers):
        """
        Initialize the Q-network with the given state dimension, hidden dimension, and number of hidden layers.
        """
        super(qNetworkDuelling, self).__init__()
        layersDim = [stateDim]
        layersDim += [hiddenDim for _ in range(numHiddenLayers)]
        layersDim.append(actionDim + 1)
        self.network = baseNet(layersDim)

    def forward(self, state, useMax=True):
        output = self.network(state)
        if state.dim() == 1:
            V, Adv = output[:1], output[1:]
        else:
            V, Adv = output[:, :1], output[:, 1:]
        if useMax:
            return V + Adv - Adv.max(-1, keepdim=True)[0]
        else:
            return V + Adv - Adv.mean(-1, keepdim=True)
        
    def save(self, file):
        torch.save(self, file)