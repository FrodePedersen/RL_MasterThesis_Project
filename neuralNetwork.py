import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import Quarto_Game as QG
import TDLambdaAgent as TDLA

class relu_Model_endSigmoid(nn.Module):

    def __init__(self):
        super(relu_Model_endSigmoid, self).__init__()
        #LAYERS
        #self.conv1 = nn.Conv2d(in_channels=7, out_channels=28,kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=28, out_channels=56,kernel_size=2, padding=1)

        #self.fc1 = nn.Linear(28 * 3 * 3, 4 * 4 * 7)
        self.fc1 = nn.Linear(3 * 4 * 4, 9 * 4 * 4)
        self.fc2 = nn.Linear(9*4*4, 16)
        self.fc3 = nn.Linear(16, 1)

        nn.init.xavier_uniform_(self.fc1.weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc3.weight, nn.init.calculate_gain('sigmoid'))
        #self.conv2 = nnConv2d

    def forward(self, input):
        x = input.view(-1, 3 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x


class softmax_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()
        # LAYERS
        # self.conv1 = nn.Conv2d(in_channels=7, out_channels=28,kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=28, out_channels=56,kernel_size=2, padding=1)

        # self.fc1 = nn.Linear(28 * 3 * 3, 4 * 4 * 7)
        self.fc1 = nn.Linear(7 * 4 * 4, 4 * 4 * 4)
        self.fc2 = nn.Linear(4 * 4 * 4, 16)
        # MASK LAYER

        nn.init.xavier_uniform_(self.fc1.weight, nn.init.calculate_gain('relu'))
        nn.init.xavier_uniform_(self.fc2.weight, nn.init.calculate_gain('relu'))


class CNN_Model(nn.Module):

    def __init__(self):
        super(CNN_Model, self).__init__()
        #LAYERS
        #self.conv1 = nn.Conv2d(in_channels=7, out_channels=28,kernel_size=3, padding=1)
        #self.conv2 = nn.Conv2d(in_channels=28, out_channels=56,kernel_size=2, padding=1)

        #self.fc1 = nn.Linear(28 * 3 * 3, 4 * 4 * 7)
        self.fc1 = nn.Linear(7 * 4 * 4, 4 * 4 * 4)
        self.fc2 = nn.Linear(4*4*4, 16)
        self.fc3 = nn.Linear(16, 1)

        nn.init.xavier_uniform_(self.fc1.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc2.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc3.weight, nn.init.calculate_gain('sigmoid'))
        #self.conv2 = nnConv2d

    def forward(self, input):
        #FORWARD PASS
        #print(f'(INPUT) {input}')
        #x = self.conv1(input)
        #print(f'conv1.size: {x.size()}')
        #x = F.relu(x)
        #print(f'relu1.size: {x.size()}')
        #x = F.max_pool2d(x, kernel_size=2, padding=1)
        #print(f'Maxpool1.size: {x.size()}')

        #x = self.conv2(x)
        #print(f'conv2.size: {x.size()}')
        #x = F.relu(x)
        #print(f'relu2.size: {x.size()}')
        #x = F.max_pool2d(x,2)
        #print(f'maxpool2.size: {x.size()}')
        #x = x.view(-1, self.num_flat_features(x))

        #x = x.view(-1, 28 * 3 * 3)
        x = input.view(-1, 7 * 4 * 4)
        x = torch.sigmoid(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #print(f'relu4: {x}')
        #print(f'before sig: {x}')
        x = torch.sigmoid(self.fc3(x))
        return x

    def SGD(self, loss, z_params, agent):
        for param, z in zip(self.parameters(), z_params):
            param[:] = param + agent.lparams['alpha'] * z * loss


    def saveModel(self):
        #Save Model params
        pass

    def loadModel(self):
        #load model params
        pass



def main():

    m = CNN_Model()
    qG = QG.GameBoard()
    agent = TDLA.TDLambdaAgent()
    input = agent.translateComplexToTensors(qG.translateSimpleToComplex(qG.getBoardStateSimple()))
    #input = torch.randn(128,20, requires_grad=True)
    #input.requires_grad = True
    m.zero_grad()
    output = m(input)
    #print(f'output: {output.size()}')
    #print(f'output!!!! {output}')
    #print(f'OUTPUT: {output.view(-1)}')
    #print(f'OUTPUT SIZE: {output.size()}')
    print(f'GRAD?!?! {input.grad}')
    grad = output.backward()
    print(f'GRAD: {input.grad}')
    #output.backward()
    #print(f'INPUT: {input.grad.data}')
    #print(f'backward: {output}')



if __name__ == "__main__":
    main()