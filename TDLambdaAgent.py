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
    '''
    def act(self):
        validMoves = self.qG.collectValidMoves() #List containing all permutations of (piece to give, valid placement of piece given)
        return random.choice(validMoves) #tuple: (index to place, piece to give)
    '''
    
    def act(self):
        validMoves = self.qG.collectValidMoves()  # List containing all permutations of (piece to give, valid placement of piece given)
        bestMove = None

        if random.random() > self.lparams['epsilon'].item():
            bestMove = self.findBestMove(validMoves)
        else:
            bestMove = random.choice(validMoves)

        if self.lparams['epsilon'] > 0.1:
            self.lparams['epsilon'] *= self.lparams['epsilon_decay']

        return bestMove  # tuple: (index to place, piece to give)

    def findBestMove(self, validMoves):
        bestScore = torch.tensor([0])
        worstScore = torch.tensor([1])
        bestMove = None
        #print(f'## AMOUNT OF VALID MOVES: {len(validMoves)}')
        placementIndex = self.qG.getBoardStateSimple()[2].view(-1).nonzero()
        placementPiece = None

        if placementIndex.size()[0] > 0:
            placementIndex = placementIndex.item() + 1
            placementPiece = self.qG.indexToPiece[placementIndex]

        for (placement, piece) in validMoves:
            #testBoard = copy.deepcopy(self.qG)
            newBoardMats = self.qG.getBoardMatrices()
            newSimpleBoardRep = self.qG.getBoardStateSimple()[0]
            if placement != None:
                placement = (placement[0], placement[1])
                newSimpleBoardRep, newBoardMats = self.qG.placePieceAt(placementPiece, placement)

            newPieceRep = self.qG.simplePieceRep
            newPickedPieceRep = self.qG.simplePickedPieceRep
            if piece != None:
                placementPiece, newPiecePool, newPieceRep, newPickedPieceRep = self.qG.takePieceFromPool(piece)



            self.currentNN.zero_grad()
            self.currentNN.eval()
            inputTens = self.qG.translateComplexToTensors(self.qG.translateSimpleToComplex(self.qG.calculateSymmetries((newSimpleBoardRep, newPieceRep, newPickedPieceRep))))
            #print(f'#'*25)
            #print(f'INPUTTENS: {inputTens}')
            #print(f'ARE WE IN HERE AT ALL?!')
            #print(f'INPUT tENS.SIZE: {inputTens.size()}')
            #print(f'input tens: {inputTens}')
            moveScore = self.currentNN(inputTens)


            #TODO: ALWAYS SAME MoveScore?!?!?!
            #print(f'MoveScore: {moveScore}')
            if moveScore.item() >= bestScore.item():
                bestScore = moveScore
                bestMove = (placement, piece)

            if moveScore.item() <= worstScore.item():
                worstScore = moveScore

        #print(f'Best Move: {bestMove}')
        #print(f'Best Score: {bestScore}, worstScore: {worstScore}, difference: {bestScore - worstScore}')

        return bestMove


    def setBoard(self, qG):
        self.qG = qG

    def set_learningParams(self, **lparams):
        for k, v in lparams:
            self.lparams[k] = v

    def translateComplexToTensors(self, complexRep):
        boardMats = complexRep[0]
        piecePool = complexRep[1]
        pickedPiece = complexRep[2]

        tens = torch.zeros((7,4,4))#,  dtype=torch.float64, requires_grad=True)
        #print(f'tens shape: {tens.size()}')
        tens[0] = boardMats['OCCUPIED'].getMatrix().clone()
        tens[1] = boardMats['HEIGHT'].getMatrix().clone()
        tens[2] = boardMats['COLOR'].getMatrix().clone()
        tens[3] = boardMats['SHAPE'].getMatrix().clone()
        tens[4] = boardMats['INDENTATION'].getMatrix().clone()
        tens[5] = piecePool.clone()
        tens[6] = pickedPiece.clone()

        outputTensor = torch.zeros((1,7,4,4))
        outputTensor[0] = tens
        outputTensor.requires_grad = True
        return outputTensor


    def __str__(self):
        return f'TDLambdaAgent'