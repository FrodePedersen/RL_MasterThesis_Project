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
        #self.fc1batch = nn.BatchNorm1d(num_features=9 * 4 * 4)
        self.fc2 = nn.Linear(9*4*4, 16)
        #self.fc2batch = nn.BatchNorm1d(num_features=16)
        self.fc3 = nn.Linear(16, 1)

        self.drop = nn.Dropout(p=0.1) # 0.5 As according to Hinton et al.
        self.sigmoid = nn.Sigmoid()

        #self.leakyRelu = nn.LeakyReLU()

        nn.init.xavier_uniform_(self.fc1.weight, nn.init.calculate_gain('sigmoid'))
        #self.fc1.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc2.weight, nn.init.calculate_gain('sigmoid'))
        #self.fc2.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.fc3.weight, nn.init.calculate_gain('sigmoid'))
        #self.fc3.bias.data.fill_(0)

        #print(f'fc2: {self.fc2.weight}')
        #print(f'fc3: {self.fc3.weight}')
        #self.conv2 = nnConv2d

    def forward(self, input):
        #i = (input.view(-1) / 16) > 1
        '''
        if len(i.nonzero()) > 0:
            print(f'LARGER THAN NORMALIZED: i')
            print(f'LARGER THAN NORMALIZED: i')
            print(f'LARGER THAN NORMALIZED: i')
            print(f'LARGER THAN NORMALIZED: i')
            print(f'LARGER THAN NORMALIZED: i')
            print(f'LARGER THAN NORMALIZED: i')
            print(f'LARGER THAN NORMALIZED: i')
        '''
        x = input.view(-1, 3 * 4 * 4)
        x = self.sigmoid(self.fc1(self.drop(x)))
        x = self.sigmoid(self.fc2(self.drop(x)))
        #print(f'fc2: {x}')
        #print(f'self.fc2.weight: {self.fc2.weight}')
        #print(f'self.fc2.bias: {self.fc2.bias}')
        x = self.drop(x)
        #print(f'x_drop?? {x} ')
        x = self.sigmoid(self.fc3(x))
        #print(f'fc3: {x}')
        return x


class softmax_Model(nn.Module):

    def __init__(self):
        super().__init__()
        # LAYERS
        # self.conv1 = nn.Conv2d(in_channels=7, out_channels=28,kernel_size=3, padding=1)
        # self.conv2 = nn.Conv2d(in_channels=28, out_channels=56,kernel_size=2, padding=1)

        # self.fc1 = nn.Linear(28 * 3 * 3, 4 * 4 * 7)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(3 * 4 * 4, 9 * 4 * 4)
        self.fc2 = nn.Linear(9 * 4 * 4, 17*17)
        self.softMax = nn.LogSoftmax(dim=0)
        self.sigmoid = nn.Sigmoid()
        #self.relu = nn.ReLU()

        # MASK LAYER

        nn.init.xavier_uniform_(self.fc1.weight, nn.init.calculate_gain('sigmoid'))
        nn.init.xavier_uniform_(self.fc2.weight, nn.init.calculate_gain('sigmoid'))

    def forward(self, *input):
        #print(f'LEN OF INPUT: {len(input)}')
        #print(f'INPUT: {input}')
        x = input[0].view(-1, 3 * 4 * 4) #input[0] is the input tensor input[1] is the mask
        #print(f'x: {x}')
        mask = input[1].view(-1, 17*17) #
        if(x.size()[0] == 1):
            x = x.view(-1)
        if(mask.size()[0] == 1):
            mask = mask.view(-1)
        #print(f'INPUT AFTER: {x,mask}')
        x = self.drop(x)
        x = self.sigmoid(self.fc1(x))
        x = self.drop(x)
        x = self.sigmoid(self.fc2(x))
        j = None
        if len(x.size()) > 1:       #For batch
            for i in range(x.size()[0]):
                print(f'TYPES MASK, X: {mask[i], x[i]}')
                x_i = (mask[i] * (x[i] + 2.0) - 1.0)

                if j is None:
                    j = [self.softMax(x_i)]
                else:
                    j.append(self.softMax(x_i))
            j = torch.stack(j)

        else:
            print(f'TYPES MASK, X: {mask, x}')
            x_i = (mask * (x + 2.0) - 1.0)
            j = self.softMax(x_i)

        #print(f'after MASK:: {x} ')


        return j


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