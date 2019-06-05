import neuralNetwork as NN
import random
import torch
import Quarto_Game as QG
import copy


class DQNAgent():

    def __init__(self, functionAproxModel):
        self.currentNN = functionAproxModel
        self.targetNN = functionAproxModel
        self.qG = None
        self.lparams = {}
        self.trainingAgent = False

    def act(self):
        validMoves = self.qG.collectValidMoves()  # List containing all permutations of (piece to give, valid placement of piece given)
        bestMove = self.findBestMove(validMoves)

        return bestMove  # tuple: (index to place, piece to give)

    def findBestMove(self, validMoves):
        self.currentNN.eval()
        bestScore = None
        worstScore = None
        #bestMove = random.choice(validMoves) #Remove once done
        # print(f'## AMOUNT OF VALID MOVES: {len(validMoves)}')
        #placementPieceIndex = self.qG.pickedPieceRep.view(-1).nonzero()

        mask_vector = torch.zeros((17*17)) #(coord, piece) (16, 16). (None, None).
        #Map valid moves to mask

        for (placementIdx, pieceIdx) in validMoves: #pieceIdx is 1-indexed

            #print(f'placementIdx: {placementIdx}')
            #print(f'pieceIdx: {pieceIdx}')
            if placementIdx is None:
                placementIdx = 16
            else:
                placementIdx = placementIdx[0] * 4 + placementIdx[1]
            if pieceIdx is None:
                pieceIdx = 16
            else:
                pieceIdx -= 1
            mask_vector[placementIdx*17+pieceIdx] = 1 #because piexeIdx is 1-indexed, minus one.
            #print(f'placementIdx: {placementIdx}')
            #print(f'pieceIdx: {pieceIdx}')


        #Get input
        newBoardRep = self.qG.boardRep
        newPiecePool = self.qG.piecePoolRep
        newPickedPieceRep = self.qG.pickedPieceRep
        inputTens = self.qG.calculateSymmetries(torch.stack([newBoardRep, newPiecePool, newPickedPieceRep]))
        inputTens = inputTens #Normalize input
        if torch.cuda.is_available():
            inputTens = inputTens.cuda()

        moveScores = self.currentNN(inputTens, mask_vector)
        #if self.trainingAgent:
        #    print(f'bestScore: {moveScores.max()}, worstScore: {moveScores.min()}, difference: {moveScores.max() - moveScores.min()}')
        return moveScores, mask_vector

    def find_mask_vector(self, state):
        masks = None
        for i in range(state.size()[0]):
            valid_moves = self.qG.findMovesForState(state[i])
            mask_vector = torch.zeros((17 * 17))  # (coord, piece) (16, 16). (None, None).
            # Map valid moves to mask

            for (placementIdx, pieceIdx) in valid_moves:  # pieceIdx is 1-indexed

                # print(f'placementIdx: {placementIdx}')
                # print(f'pieceIdx: {pieceIdx}')
                if placementIdx is None:
                    placementIdx = 16
                else:
                    placementIdx = placementIdx[0] * 4 + placementIdx[1]
                if pieceIdx is None:
                    pieceIdx = 16
                else:
                    pieceIdx -= 1
                mask_vector[placementIdx * 17 + pieceIdx] = 1  # because piexeIdx is 1-indexed, minus one.

            if masks is None:
                masks = [mask_vector]
            else:
                masks.append(mask_vector)

        masks = torch.stack(masks)
        return masks

    def translateScoresToMove(self, moveIdx):
        #print(f'move: {moveIdx}')
        coord = int(moveIdx / 17) #set the coordinate entry
        if coord == 16:
            coord = None
        else:
            coord = (int(coord / 4), coord % 4)

        pieceIdx = moveIdx % 17 + 1

        if pieceIdx == 17:
            pieceIdx = None
        #print(f'TranslateScoresToMove: {(coord, pieceIdx)}')
        return (coord, pieceIdx)

    def setBoard(self, qG):
        self.qG = qG

    def set_learningParams(self, **lparams):
        for k, v in lparams:
            self.lparams[k] = v

    def __str__(self):
        return f'DQNAgent'