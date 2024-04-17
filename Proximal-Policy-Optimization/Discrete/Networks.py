import torch
import torch.nn as nn
import numpy as np
from torch.functional import F

def orth(layer, weightStd=np.sqrt(2), biasValue=0.0):
    nn.init.orthogonal_(layer.weight, weightStd)
    nn.init.constant_(layer.bias, biasValue)
    return layer

class baseNetOrthogonal(nn.Module):
    def __init__(self, layersDim, finalLayerStd):
        super(baseNetOrthogonal, self).__init__()
        self.network = nn.Sequential()
        for ii in range(len(layersDim) - 2):
            self.network.append(orth(nn.Linear(layersDim[ii], layersDim[ii+1])))
            #self.network.append(nn.Linear(layersDim[ii], layersDim[ii+1]))
            self.network.append(nn.Tanh())
        self.network.append(orth(nn.Linear(layersDim[-2], layersDim[-1]), weightStd=finalLayerStd))
        #self.network.append(nn.Linear(layersDim[-2], layersDim[-1]))
        
    def forward(self, x):
        return self.network(x)
    
class valueNetwork(nn.Module):
    def __init__(self, stateDim, HiddenDim, numHiddenLayers):
        super(valueNetwork, self).__init__()
        layersDim = [stateDim]
        layersDim += [HiddenDim for _ in range(numHiddenLayers)]
        layersDim.append(1)
        self.network = baseNetOrthogonal(layersDim, 1.0)
    
    def forward(self, state):
        return self.network(state)
        
    def save(self, file):
        torch.save(self, file)
        
class policyNetwork(nn.Module):
    def __init__(self, stateDim, actionDim, HiddenDim, numHiddenLayers):
        super(policyNetwork, self).__init__()
        layersDim = [stateDim]
        layersDim += [HiddenDim for _ in range(numHiddenLayers)]
        layersDim.append(actionDim)
        self.network = baseNetOrthogonal(layersDim, 0.01)
        
    def forward(self, state):
        return torch.distributions.Categorical(logits=self.network(state))
    
    def save(self, file):
        torch.save(self, file)