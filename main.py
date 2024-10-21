# Initial Implementation, Probo pretty shit
import torch
from torch import tensor

device = torch.device('mps') # MACOS ONLY, CHANGE TO CUDA OR CPU

class BigVirtualBrain:
    def __init__(self, inputsize=):
        self.inputLayer = BrainLayer()
        self.layers = [] 

    def addHiddenLayer(self, size=32):
        self.layers.append(BrainLayer(size=size))

class BrainLayer:
    def __init__(self, size=32, _type=1): # t:type 0:input, 1:hidden, 2:output
        self.nodes = torch.ones(size, dtype=torch.float32, device=device)
        self._type = _type
        # self.unweighted_nodes = self.nodes
        self.relu = torch.nn.ReLU()

    def updateLayer(self, prevMatrix):
        self.nodes = self.relu(prevMatrix) @ self.nodes


