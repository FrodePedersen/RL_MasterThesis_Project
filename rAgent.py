import main
import Quarto_Game as QG
import torch
import random

class rAgent():

    def __init__(self, qG, seed):
        self.qG = qG
        #random.seed(seed)


    def act(self):
        validMoves = self.qG.collectValidMoves() #List containing all permutations of (piece to give, valid placement of piece given)
        return random.choice(validMoves) #tuple: (index to place, piece to give)
