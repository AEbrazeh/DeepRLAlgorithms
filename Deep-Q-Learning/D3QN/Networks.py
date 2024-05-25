import torch
import torch.nn as nn
import numpy as np
from torch.distributions.normal import Normal

class baseNet(nn.Module):
    def __init__(self, layersDim, lastLayerActivation = False):
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
        if lastLayerActivation: self.network.append(nn.LeakyReLU(0.01))
        
    def forward(self, x):
        return self.network(x)
    
class qNetworkDuelling(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDim, numHiddenLayers):
        """
        Initialize the Duelling Q-network with the given state dimension, hidden dimension, and number of hidden layers.
        """
        super(qNetworkDuelling, self).__init__()
        numHL1 = numHiddenLayers // 2
        numHL2 = numHiddenLayers - numHL1 + 1
        self.baseNetwork = baseNet([stateDim] + [hiddenDim for _ in range(numHL1)], lastLayerActivation=True)
        self.advNetwork = baseNet([hiddenDim for _ in range(numHL2)] + [actionDim])
        self.valueNetwork = baseNet([hiddenDim for _ in range(numHL2)] + [1])

    def forward(self, state):
        H = self.baseNetwork(state)
        V = self.valueNetwork(H)
        Adv = self.advNetwork(H)
        return V + Adv - Adv.mean(-1, keepdim=True)
        
    def save(self, file):
        torch.save(self, file)