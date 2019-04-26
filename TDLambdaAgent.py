import neuralNetwork as NN
import random

class TDLambdaAgent():

    def __init__(self):
        nn = NN.NN_Model()
        self.qG = None

    def act(self):
        validMoves = self.qG.collectValidMoves()  # List containing all permutations of (piece to give, valid placement of piece given)
        return random.choice(validMoves)  # tuple: (index to place, piece to give)

    def setBoard(self, qG):
        self.qG = qG

    def __str__(self):
        return f'TDLambdaAgent'