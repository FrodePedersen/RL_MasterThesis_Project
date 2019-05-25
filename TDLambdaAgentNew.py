import neuralNetwork as NN
import random
import torch
import Quarto_Game as QG
import copy


class TDLambdaAgent():

    def __init__(self, functionAproxModel):
        self.currentNN = functionAproxModel
        self.targetNN = functionAproxModel
        self.qG = None
        self.lparams = {}
        self.trainingAgent = False

    def act(self):
        validMoves = self.qG.collectValidMoves()  # List containing all permutations of (piece to give, valid placement of piece given)
        bestMove = None

        if random.random() > self.lparams['epsilon'].item():
            bestMove = self.findBestMove(validMoves)
        else:
            bestMove = random.choice(validMoves)

        return bestMove  # tuple: (index to place, piece to give)

    def findBestMove(self, validMoves):
        bestScore = torch.tensor([0])
        worstScore = torch.tensor([1])
        bestMove = None
        # print(f'## AMOUNT OF VALID MOVES: {len(validMoves)}')
        placementPieceIndex = self.qG.pickedPieceRep.view(-1).nonzero()

        if placementPieceIndex.size()[0] > 0:
            placementPieceIndex = placementPieceIndex.item() + 1

        #print(f'LLLEEEENNN::: {len(validMoves)}')
        for (placement, pieceIdx) in validMoves:
            newBoardRep = self.qG.boardRep
            newPiecePool = self.qG.piecePoolRep
            newPickedPieceRep = self.qG.pickedPieceRep

            if placement != None:
                placement = (placement[0], placement[1])
                newBoardRep = self.qG.placePieceAt(placementPieceIndex, placement)

            if pieceIdx != None:
                newPiecePool, newPickedPieceRep = self.qG.takePieceFromPool(pieceIdx)
                placementPieceIndex = pieceIdx

            self.currentNN.zero_grad()
            self.currentNN.eval()
            #print(f'BEFORE SYMS: {torch.stack([newBoardRep, newPiecePool, newPickedPieceRep])}')
            inputTens = self.qG.calculateSymmetries(torch.stack([newBoardRep, newPiecePool, newPickedPieceRep]))
            if torch.cuda.is_available():
                #print(f'ARE WEHERERERE?!?!?!?!?!?')
                inputTens = inputTens.cuda()
            #print(f'IS CUDA? {torch.cuda.is_available()}')
            #print(f'INPUT TENS: {inputTens}')
            #print(f'DID WE GET SYMS? {inputTens}')
            moveScore = self.currentNN(inputTens)

            # TODO: ALWAYS SAME MoveScore?!?!?!
            # print(f'MoveScore: {moveScore}')
            if moveScore.item() >= bestScore.item():
                bestScore = moveScore
                bestMove = (placement, pieceIdx)

            if moveScore.item() <= worstScore.item():
                worstScore = moveScore

        # print(f'Best Move: {bestMove}')
        #if self.trainingAgent:
        #    print(f'Best Score: {bestScore}, worstScore: {worstScore}, difference: {bestScore - worstScore}')

        return bestMove

    def setBoard(self, qG):
        self.qG = qG

    def set_learningParams(self, **lparams):
        for k, v in lparams:
            self.lparams[k] = v

    def __str__(self):
        return f'TDLambdaAgentNEW'