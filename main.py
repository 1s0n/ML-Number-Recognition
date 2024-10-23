# Initial Implementation, Probo pretty shit
import torch
from torch import tensor
from os.path import join
from load import MnistDataloader, show_images
import numpy as np

device = torch.device('mps') # MACOS ONLY, CHANGE TO CUDA OR CPU

class BigVirtualBrain:
    def __init__(self, inputsize=784, layerSize=32):
        self.inputLayer = BrainLayer(size=inputsize, _type=0)
        self.outputLayer = BrainLayer(size=10, _type=2, prevsize=layerSize)

        self.layers = [] 
        
        self.addHiddenLayer(layerSize, inputsize)
        self.addHiddenLayer(layerSize)    
        # print(len(self.inputLayer.nodes))

    def addHiddenLayer(self, size=32, prevSize=32):
        self.layers.append(BrainLayer(size=size, prevsize=prevSize))
    
    def save(self):
        pass

    def load(self, data):
        pass
    
    def input(self, inputmatrix):
        
        if type(inputmatrix) != torch.Tensor:
            inp = tensor(inputmatrix, dtype=torch.float32, device=device)
        else:
            inp = inputmatrix
        
        if len(inputmatrix) == 28:
            inp = torch.ravel(inp)

        
        self.inputLayer.setLayer(inp) 
    
    def process(self):  
        # TODO: Fix this bit
        index = 0
        for layer in self.layers:
            if index == 0:
                # print(len(self.inputLayer.nodes))
                layer.updateLayer(self.inputLayer.nodes, debugindex=index)
            else:
                layer.updateLayer(self.layers[index - 1].nodes, debugindex=index)
            index += 1

        self.outputLayer.updateLayer(self.layers[index - 1].nodes) 

class BrainLayer:
    def __init__(self, size=32, _type=1, prevsize=32): # t:type 0:input, 1:hidden, 2:output
        self.nodes = torch.ones(size, dtype=torch.float32, device=device)
         
        # print(len(self.nodes))

        self.previous_size=prevsize
        
        self.biases = torch.zeros(size, dtype=torch.float32, device=device) 
        

        if _type != 0:
            # Height has to be the same as prevsize, width same as size
            self.weights = torch.ones(size=(prevsize, size), dtype=torch.float32, device=device)

        self._type = _type
        
        # self.activFunc = torch.nn.ReLU()
        self.activFunc = torch.nn.Sigmoid()
        
        

    def updateLayer(self, prevMatrix, debugindex=0):
        print(len(prevMatrix))
        print(debugindex)
        if type(prevMatrix) == torch.Tensor:
           self.nodes = self.activFunc(torch.add(prevMatrix @ self.weights, self.biases))
        else:
            self.nodes = self.activFunc(torch.add(tensor(prevMatrix) @ self.weights, self.biases))
        

    def setLayer(self, mat):
        self.nodes = mat

    def toList(self):
        return [self._type, self.nodes, self.weights, self.biases]

    def fromList(self, d):
        self._type = d[0]
        self.nodes = d[1]
        self.weights = d[2]
        self.biases = d[3]

    def yieldNodes(self):
        return self.nodes

def mse(y_true, y_pred):
    return torch.mean(torch.square(torch.sub(y_true, y_pred)))

class BigLearningBrain(BigVirtualBrain):
    def __init__(self):

        input_path = './data/'
        training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
        training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
        test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
        test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

        mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
        (self.x_train, self.y_train), (self.x_test, self.y_test) = mnist_dataloader.load_data()

        brain = BigVirtualBrain()
        brain.input(self.x_train[0])
        print(self.x_train[0][0])

        brain.process()
        output = brain.outputLayer.yieldNodes()

        print(output)
        print(self.y_train[0])
        print(self.cost(output, self.y_train[0]))

    def cost(self, output, predictionVal):
        prediction = [0 for a in range(10)]
        prediction[predictionVal] = 1
        prediction = tensor(prediction, dtype=torch.float32, device=device)

        loss = mse(output, prediction) 
        return loss

if __name__ == "__main__":
    brain = BigLearningBrain()

