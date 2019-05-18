from distutils.command.check import check

#from bitarray import bitarray
import torch
import copy
import random



class GameBoard(object):

    def __init__(self):
        self.ID = 'NEW'
        self.boardRep = torch.zeros((4, 4))
        self.piecePoolRep = torch.ones((4,4))
        self.pickedPieceRep = torch.zeros((4,4))

        self.firstMove = True

        self.verticalSwap = False
        self.horizontalSwap = False

        #Translation between index rep to torch.tensor
        self.indexToPieceTensor = {}

        self.indexToPieceTensor[0] = None
        self.indexToPieceTensor[1] = torch.tensor([0,0,0,0])
        self.indexToPieceTensor[2] = torch.tensor([0,0,0,1])
        self.indexToPieceTensor[3] = torch.tensor([0,0,1,0])
        self.indexToPieceTensor[4] = torch.tensor([0,0,1,1])
        self.indexToPieceTensor[5] = torch.tensor([0,1,0,0])
        self.indexToPieceTensor[6] = torch.tensor([0,1,0,1])
        self.indexToPieceTensor[7] = torch.tensor([0,1,1,0])
        self.indexToPieceTensor[8] = torch.tensor([0,1,1,1])
        self.indexToPieceTensor[9] = torch.tensor([1,0,0,0])
        self.indexToPieceTensor[10] = torch.tensor([1,0,0,1])
        self.indexToPieceTensor[11] = torch.tensor([1,0,1,0])
        self.indexToPieceTensor[12] = torch.tensor([1,0,1,1])
        self.indexToPieceTensor[13] = torch.tensor([1,1,0,0])
        self.indexToPieceTensor[14] = torch.tensor([1,1,0,1])
        self.indexToPieceTensor[15] = torch.tensor([1,1,1,0])
        self.indexToPieceTensor[16] = torch.tensor([1,1,1,1])

        # Index to String mapping
        self.indexToString = {}

        self.indexToString[1] = 'SHORT WHITE ROUND FILLED'
        self.indexToString[2] = 'SHORT WHITE ROUND INDENTED'
        self.indexToString[3] = 'SHORT WHITE SQUARE FILLED'
        self.indexToString[4] = 'SHORT WHITE SQUARE INDENTED'
        self.indexToString[5] = 'SHORT BLACK ROUND FILLED'
        self.indexToString[6] = 'SHORT BLACK ROUND INDENTED'
        self.indexToString[7] = 'SHORT BLACK SQUARE FILLED'
        self.indexToString[8] = 'SHORT BLACK SQUARE INDENTED'
        self.indexToString[9] = 'TALL WHITE ROUND FILLED'
        self.indexToString[10] = 'TALL WHITE ROUND INDENTED'
        self.indexToString[11] = 'TALL WHITE SQUARE FILLED'
        self.indexToString[12] = 'TALL WHITE SQUARE INDENTED'
        self.indexToString[13] = 'TALL BLACK ROUND FILLED'
        self.indexToString[14] = 'TALL BLACK ROUND INDENTED'
        self.indexToString[15] = 'TALL BLACK SQUARE FILLED'
        self.indexToString[16] = 'TALL BLACK SQUARE INDENTED'

        self.stringToIndex = {}

        self.stringToIndex['SHORT WHITE ROUND FILLED'] = 1
        self.stringToIndex['SHORT WHITE ROUND INDENTED'] = 2
        self.stringToIndex['SHORT WHITE SQUARE FILLED'] = 3
        self.stringToIndex['SHORT WHITE SQUARE INDENTED'] = 4
        self.stringToIndex['SHORT BLACK ROUND FILLED'] = 5
        self.stringToIndex['SHORT BLACK ROUND INDENTED'] = 6
        self.stringToIndex['SHORT BLACK SQUARE FILLED'] = 7
        self.stringToIndex['SHORT BLACK SQUARE INDENTED'] = 8
        self.stringToIndex['TALL WHITE ROUND FILLED'] = 9
        self.stringToIndex['TALL WHITE ROUND INDENTED'] = 10
        self.stringToIndex['TALL WHITE SQUARE FILLED'] = 11
        self.stringToIndex['TALL WHITE SQUARE INDENTED'] = 12
        self.stringToIndex['TALL BLACK ROUND FILLED'] = 13
        self.stringToIndex['TALL BLACK ROUND INDENTED'] = 14
        self.stringToIndex['TALL BLACK SQUARE FILLED'] = 15
        self.stringToIndex['TALL BLACK SQUARE INDENTED'] = 16

        self.isDone = False
        self.isWon = False
        self.isDone = False

    def __str__(self):
        outputString = 'Piece Pool:\n'
        for piece in self.piecePool:
            outputString += f'  {piece}\n'

        outputString += '\nMatrices:\n'

        for type, matrice in self.boardMatrices.items():
            outputString += f'  type: {type}\n{matrice.getMatrix()}\n'

        outputString += '\nBoard:\n'

        occupancyIndeces = self.boardMatrices['OCCUPIED'].nonzero()

        occDict = {}

        for i in range(4):
            for j in range(4):
                print(f'{torch.tensor([0, 0, 0, 0])}')

        return outputString

    def printGame(self):
        print(f'#' * 64)
        print()

        for i in range(4):
            row = ""
            for j in range(4):
                boardValue = self.boardRep[(i, j)].item()
                if boardValue != 0:
                    encoding = self.indexToPieceTensor[boardValue].tolist()
                    row = row + str(encoding) + " " * 3
                else:
                    encoding = 0
                    row = row + str(encoding) + " " * 14

            print(row)

        print()
        print(f'#' * 64)

        print()
        print(f'PIECE ENCODING: [HEIGHT, COLOR, SHAPE, INDENTED]\n')

        print(f'\nPIECE POOL:')
        for i in range(16):
            row = ""
            pieceValue = int(self.piecePoolRep.view(-1)[i].item())
            if pieceValue > 0:
                print (f'index: {i+1}, encoding: {self.indexToPieceTensor[i+1]}, String Rep: {self.indexToString[i+1]}')
            else:
                print(f'/// index: {i+1}, encoding: {self.indexToPieceTensor[i+1]}, String Rep: {self.indexToString[i+1]} ///')

    def calculateSymmetries(self, simpleReps):
        boardRepSymmetry = simpleReps.clone()

        #print(f'BOOLS: {self.verticalSwap, self.horizontalSwap}')
        if self.horizontalSwap:
            boardRepSymmetry[0] = boardRepSymmetry[0].flip(0)
        if self.verticalSwap:
            boardRepSymmetry[0] = boardRepSymmetry[0].flip(1)

        return boardRepSymmetry

    def translateSymmetryToBoardState(self, position):
        row, col = position[0], position[1]
        if self.horizontalSwap:
            row = abs(3 - row)
        if self.verticalSwap:
            col = abs(3 - col)

        return ((row, col))

    # Creates a list of all valid combinations of free indeces and pieces. returns list of tuples: [(indx, piece), (indx, piece) ... ]
    def collectValidMoves(self):
        validIndeces = []
        if not self.boardRep.sum() == 0 or len((self.piecePoolRep == 0).nonzero()) > 0:  # Checks if the board is clean, if so don't place
            validIndeces = (self.boardRep == 0).nonzero().tolist()

        if len(validIndeces) < 1:
            validIndeces = [None]  # Start of the game has no indeces

        validPieceIds = [p.item() + 1 for p in self.piecePoolRep.view(-1).nonzero()]
        if len(validPieceIds) < 1:
            validPieceIds = [None]  # End of game has no valid pieces, only 1 index

        result = []

        for index in validIndeces:
            for piece in validPieceIds:
                result.append((index, piece))

        # print(f'### COLLECTVALID: LEN: {len(result)}')#, list: {result}')
        return result

    def getPiecePool(self):
        return self.piecePool

    def takePieceFromPool(self, pieceIdx):

        newPiecePool = self.piecePoolRep.clone()
        newPiecePool.view(-1)[pieceIdx-1] = 0

        newPickedPieceRep = torch.zeros((4,4))
        newPickedPieceRep.view(-1)[pieceIdx-1] = 1

        return newPiecePool, newPickedPieceRep

    def placePieceAt(self, pieceIdx, position):
        # Set symmetry direction of first move

        if not self.boardRep[position].item() == 0.0:
            raise Exception(f'POSITION {position} ALREADY TAKEN')

        cloneBoardRep = self.boardRep.clone()
        cloneBoardRep[position] = pieceIdx

        return cloneBoardRep

    def storeState(self, **args):
        if 'boardmat' in args.keys():
            self.boardRep = args['boardmat']
        if 'piecePool' in args.keys():
            self.piecePoolRep = args['piecePool']
        if 'pickedPieceRep' in args.keys():
            self.pickedPieceRep = args['pickedPieceRep']

    def setSyms(self, firstMovePos):
        self.firstMove = False
        position = firstMovePos
        #print(f'Sym Position?? {position}')
        row, col = position[0], position[1]
        if row >= 2:
            self.horizontalSwap = True
        if col >= 2:
            self.verticalSwap = True


    def isWinningMove(self, boardMatrice):

        checkingMatrice = boardMatrice.clone()

        checkingMatHEIGHT = torch.zeros(4,4)
        checkingMatCOLOR = torch.zeros(4,4)
        checkingMatSHAPE = torch.zeros(4,4)
        checkingMatINDENTATION = torch.zeros(4,4)
        checkingOCCUPIED = torch.zeros(4,4)

        #Collecting mats:
        for i in range(4):
            for j in range(4):
                pieceTens = self.indexToPieceTensor[checkingMatrice[i,j].item()]
                #print(f'pieceTens: {pieceTens}')
                if pieceTens is None:
                    checkingOCCUPIED[i,j]  = 0
                else:
                    checkingOCCUPIED[i, j] = 1
                    checkingMatHEIGHT[i, j] = pieceTens[0]
                    checkingMatCOLOR[i, j] = pieceTens[1]
                    checkingMatSHAPE[i, j] = pieceTens[2]
                    checkingMatINDENTATION[i, j] = pieceTens[3]

        #print(f'AFTERAFTER')


        #checking rows:
        for i in range(4):
            if len((checkingOCCUPIED[i] == 0).nonzero()) < 1:
                if checkingMatHEIGHT[i].sum() == 0 or checkingMatHEIGHT[i].sum() == 4:
                    return True
                if checkingMatCOLOR[i].sum() == 0 or checkingMatCOLOR[i].sum() == 4:
                    return True
                if checkingMatSHAPE[i].sum() == 0 or checkingMatSHAPE[i].sum() == 4:
                    return True
                if checkingMatINDENTATION[i].sum() == 0 or checkingMatINDENTATION[i].sum() == 4:
                    return True

        #checking cols:




        for i in range(4):
            if len((checkingOCCUPIED.transpose(0,1)[i] == 0).nonzero()) < 1:
                if checkingMatHEIGHT.transpose(0,1)[i].sum() == 0 or checkingMatHEIGHT.transpose(0,1)[i].sum() == 4:
                    return True
                if checkingMatCOLOR.transpose(0,1)[i].sum() == 0 or checkingMatCOLOR.transpose(0,1)[i].sum() == 4:
                    return True
                if checkingMatSHAPE.transpose(0,1)[i].sum() == 0 or checkingMatSHAPE.transpose(0,1)[i].sum() == 4:
                    return True
                if checkingMatINDENTATION.transpose(0,1)[i].sum() == 0 or checkingMatINDENTATION.transpose(0,1)[i].sum() == 4:
                    return True

        #checking diagonals:
        if len((checkingOCCUPIED.diag() == 0).nonzero()) < 1:
            if checkingMatHEIGHT.diag().sum() == 0 or checkingMatHEIGHT.diag().sum() == 4:
                return True
            if checkingMatCOLOR.diag().sum() == 0 or checkingMatCOLOR.diag().sum() == 4:
                return True
            if checkingMatSHAPE.diag().sum() == 0 or checkingMatSHAPE.diag().sum() == 4:
                return True
            if checkingMatINDENTATION.diag().sum() == 0 or checkingMatINDENTATION.diag().sum() == 4:
                return True

        #checking reverse diagonals
        if len((checkingOCCUPIED.rot90(1).diag() == 0).nonzero()) < 1:
            if checkingMatHEIGHT.rot90(1).diag().sum() == 0 or checkingMatHEIGHT.rot90(1).diag().sum() == 4:
                return True
            if checkingMatCOLOR.rot90(1).diag().sum() == 0 or checkingMatCOLOR.rot90(1).diag().sum() == 4:
                return True
            if checkingMatSHAPE.rot90(1).diag().sum() == 0 or checkingMatSHAPE.rot90(1).diag().sum() == 4:
                return True
            if checkingMatINDENTATION.rot90(1).diag().sum() == 0 or checkingMatINDENTATION.rot90(1).diag().sum() == 4:
                return True

        return False

    def isDraw(self):
        if not self.isDone and len((self.boardRep == 0).nonzero()) < 1:
            return True
        return False

    def randomInitOfGame(self):
        amount_of_moves = random.randint(0, 15)
        placementPiece = None

        for i in range(amount_of_moves):
            (placement, pieceIdx) = random.choice(self.collectValidMoves())

            if placement != None:
                placement = (placement[0], placement[1])
                newBoardRep = self.placePieceAt(placementPiece, placement)
                self.storeState(boardmat=newBoardRep)
                if self.firstMove:
                    self.setSyms(placement)

                if self.isWinningMove(newBoardRep):
                    self.isDone = True
                    return placementPiece

            if pieceIdx != None:
                newPiecePool, newPickedPieceRep = self.takePieceFromPool(pieceIdx)
                self.storeState(piecePool=newPiecePool, pickedPieceRep=newPickedPieceRep)
                placementPiece = pieceIdx

        return placementPiece