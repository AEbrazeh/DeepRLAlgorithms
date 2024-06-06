import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import linear

def scaledNoise(size):
    noise = torch.randn(size)
    return noise.abs().sqrt() * noise.sign()
class noisyLinear(nn.Module):
    def __init__(self, inputDim, outputDim, stdInit=0.5):
        super(noisyLinear, self).__init__()
        
        self.inputDim = inputDim
        self.outputDim = outputDim
        self.stdInit = stdInit
        
        self.meanWeight = nn.Parameter(torch.Tensor(outputDim, inputDim))
        self.meanWeight.data.uniform_(-1 / np.sqrt(max(inputDim, outputDim)), 1 / np.sqrt(max(inputDim, outputDim)))
        
        self.meanBias = nn.Parameter(torch.Tensor(outputDim))
        self.meanBias.data.uniform_(-1 / np.sqrt(max(inputDim, outputDim)), 1 / np.sqrt(max(inputDim, outputDim)))
        
        self.stdWeight = nn.Parameter(torch.Tensor(outputDim, inputDim))
        self.stdWeight.data.fill_(stdInit / np.sqrt(max(inputDim, outputDim)))
        
        self.stdBias = nn.Parameter(torch.Tensor(outputDim))
        self.stdBias.data.fill_(stdInit / np.sqrt(max(inputDim, outputDim)))
        
        self.register_buffer('epsWeight', torch.Tensor(outputDim, inputDim))
        self.register_buffer('epsBias', torch.Tensor(outputDim))
        self.resetNoise()
        
    def resetNoise(self):
        self.epsWeight.copy_(scaledNoise(self.outputDim).outer(scaledNoise(self.inputDim)))
        self.epsBias.copy_(scaledNoise(self.outputDim))
        
    def forward(self, x):
        return linear(x, self.meanWeight + self.epsWeight * self.stdWeight, self.meanBias + self.epsBias * self.stdBias)
        
class noisyNet(nn.Module):
    def __init__(self, layersDim, lastLayerActivation = False):
        """
        Initializes a noisy feed forward neural network based on "Noisy Networks for Exploration" (https://arxiv.org/pdf/1706.10295) by Fortunato et al.

        Parameters:
            layersDim (list): The dimensions of the layers.
            lastLayerActivation (bool): if True the activation function is applied on the final layer too. Default is False.
        Returns:
            None
        """
        super(noisyNet, self).__init__()
        self.network = nn.Sequential()
        for ii in range(len(layersDim) - 2):
            self.network.append(noisyLinear(layersDim[ii], layersDim[ii+1]))
            self.network.append(nn.LeakyReLU(0.1))
        self.network.append(noisyLinear(layersDim[-2], layersDim[-1]))
        if lastLayerActivation: self.network.append(nn.LeakyReLU(0.1))
        
    def forward(self, x):
        return self.network(x)
    
    def resetNoise(self):
        for layer in self.network:
            if isinstance(layer, noisyLinear): layer.resetNoise()
    
class qNetworkDuelling(nn.Module):
    def __init__(self, stateDim, actionDim, hiddenDim, numHiddenLayers):
        """
        Initialize the Duelling Q-network with the given state dimension, hidden dimension, and number of hidden layers.
        """
        super(qNetworkDuelling, self).__init__()
        numHL1 = numHiddenLayers // 2
        numHL2 = numHiddenLayers - numHL1 + 1
        self.baseNetwork = noisyNet([stateDim] + [hiddenDim for _ in range(numHL1)], lastLayerActivation=True)
        self.advNetwork = noisyNet([hiddenDim for _ in range(numHL2)] + [actionDim])
        self.valueNetwork = noisyNet([hiddenDim for _ in range(numHL2)] + [1])

    def forward(self, state):
        H = self.baseNetwork(state)
        V = self.valueNetwork(H)
        Adv = self.advNetwork(H)
        return V + Adv - Adv.mean(-1, keepdim=True)
    
    def resetNoise(self):
        self.baseNetwork.resetNoise()
        self.valueNetwork.resetNoise()
        self.advNetwork.resetNoise()
        
    def save(self, file):
        torch.save(self, file)