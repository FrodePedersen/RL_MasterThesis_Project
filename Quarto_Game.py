from bitarray import bitarray
import torch

class Piece():

    def __init__(self, attributes=torch.zeros(4, dtype=torch.int)):
        self.attributes = attributes

    def __str__(self):
        height = ''
        color = ''
        shape = ''
        indentation = ''
        if self.attributes[0] == 1:
            height = 'TALL'
        elif self.attributes[0] == 0:
            height = 'SHORT'
        else:
            raise Exception(f'Piece containing no height!')

        if self.attributes[1] == 1:
            color = 'BLACK'
        elif self.attributes[1] == 0:
            color = 'WHITE'
        else:
            raise Exception(f'Piece containing no color!')

        if self.attributes[2] == 1:
            shape = 'SQUARE'
        elif self.attributes[2] == 0:
            shape = 'ROUND'
        else:
            raise Exception(f'Piece containing no shape!')

        if self.attributes[3] == 1:
            indentation = 'INDENTED'
        elif self.attributes[3] == 0:
            indentation = 'FILLED'
        else:
            raise Exception(f'Piece containing no indentation')

        return f'{height} {color} {shape} {indentation}'


class GameBoard():

    def __init__(self):

        self.simpleRep = torch.zeros((4,4), dtype=torch.int32)

        self.boardMatrices = {'OCCUPIED': BoardMatrix(matType='OCCUPIED'),
                              'HEIGHT': BoardMatrix(matType='HEIGHT'),
                              'COLOR': BoardMatrix(matType='COLOR'),
                              'SHAPE': BoardMatrix(matType='SHAPE'),
                              'INDENTATION': BoardMatrix(matType='INDENTATION')}

        p1 = Piece(torch.tensor([0,0,0,0]))
        p2 = Piece(torch.tensor([0,0,0,1]))
        p3 = Piece(torch.tensor([0,0,1,0]))
        p4 = Piece(torch.tensor([0,0,1,1]))
        p5 = Piece(torch.tensor([0,1,0,0]))
        p6 = Piece(torch.tensor([0,1,0,1]))
        p7 = Piece(torch.tensor([0,1,1,0]))
        p8 = Piece(torch.tensor([0,1,1,1]))
        p9 = Piece(torch.tensor([1,0,0,0]))
        p10 = Piece(torch.tensor([1,0,0,1]))
        p11 = Piece(torch.tensor([1,0,1,0]))
        p12 = Piece(torch.tensor([1,0,1,1]))
        p13 = Piece(torch.tensor([1,1,0,0]))
        p14 = Piece(torch.tensor([1,1,0,1]))
        p15 = Piece(torch.tensor([1,1,1,0]))
        p16 = Piece(torch.tensor([1,1,1,1]))

        #piece pool
        self.piecePool = {}

        self.piecePool['SHORT WHITE ROUND FILLED'] =    p1
        self.piecePool['SHORT WHITE ROUND INDENTED'] =  p2
        self.piecePool['SHORT WHITE SQUARE FILLED'] =   p3
        self.piecePool['SHORT WHITE SQUARE INDENTED'] = p4
        self.piecePool['SHORT BLACK ROUND FILLED'] =    p5
        self.piecePool['SHORT BLACK ROUND INDENTED'] =  p6
        self.piecePool['SHORT BLACK SQUARE FILLED'] =   p7
        self.piecePool['SHORT BLACK SQUARE INDENTED'] = p8
        self.piecePool['TALL WHITE ROUND FILLED'] =     p9
        self.piecePool['TALL WHITE ROUND INDENTED'] =   p10
        self.piecePool['TALL WHITE SQUARE FILLED'] =   p11
        self.piecePool['TALL WHITE SQUARE INDENTED'] = p12
        self.piecePool['TALL BLACK ROUND FILLED'] =     p13
        self.piecePool['TALL BLACK ROUND INDENTED'] =   p14
        self.piecePool['TALL BLACK SQUARE FILLED'] =    p15
        self.piecePool['TALL BLACK SQUARE INDENTED'] =  p16

        #Index to String mapping
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

        #PieceToIndex Mapping
        self.pieceToIndex = {}

        self.pieceToIndex[p1] = 1
        self.pieceToIndex[p2] = 2
        self.pieceToIndex[p3] = 3
        self.pieceToIndex[p4] = 4
        self.pieceToIndex[p5] = 5
        self.pieceToIndex[p6] = 6
        self.pieceToIndex[p7] = 7
        self.pieceToIndex[p8] = 8
        self.pieceToIndex[p9] = 9
        self.pieceToIndex[p10] = 10
        self.pieceToIndex[p11] = 11
        self.pieceToIndex[p12] = 12
        self.pieceToIndex[p13] = 13
        self.pieceToIndex[p14] = 14
        self.pieceToIndex[p15] = 15
        self.pieceToIndex[p16] = 16

        #Index To Piece mapping
        self.indexToPiece = {}

        self.indexToPiece[0] = 0
        self.indexToPiece[1] = p1
        self.indexToPiece[2] = p2
        self.indexToPiece[3] = p3
        self.indexToPiece[4] = p4
        self.indexToPiece[5] = p5
        self.indexToPiece[6] = p6
        self.indexToPiece[7] = p7
        self.indexToPiece[8] = p8
        self.indexToPiece[9] = p9
        self.indexToPiece[10] = p10
        self.indexToPiece[11] = p11
        self.indexToPiece[12] = p12
        self.indexToPiece[13] = p13
        self.indexToPiece[14] = p14
        self.indexToPiece[15] = p15
        self.indexToPiece[16] = p16

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
                print(f'{torch.tensor([0,0,0,0])}')


        return outputString


    def printGame(self):
        print(f'#'*64)
        print()

        for i in range(4):
            row = ""
            for j in range(4):
                boardValue = self.simpleRep[(i,j)].item()
                if boardValue != 0:
                    piece = self.indexToPiece[boardValue]
                    encoding = piece.attributes.tolist()
                    row = row + str(encoding) + " " * 3
                else:
                    encoding = 0
                    row = row + str(encoding) + " " * 14

            print(row)

        print()
        print(f'#'*64)

        print()
        print(f'PIECE ENCODING: [HEIGHT, COLOR, SHAPE, INDENTED]\n')

        print(f'\nPIECE POOL:')
        index = 1
        for k,v in self.piecePool.items():
            if v != None:
                print(f'index: {self.pieceToIndex[v]}, encoding: {v.attributes}, String Rep: {k}')
            else:
                print(f'index: {index} TAKEN')
            index += 1

    def translateSimpleToComplex(self, simple):
        boardMatrices = {'OCCUPIED': BoardMatrix(matType='OCCUPIED'),
                         'HEIGHT': BoardMatrix(matType='HEIGHT'),
                         'COLOR': BoardMatrix(matType='COLOR'),
                         'SHAPE': BoardMatrix(matType='SHAPE'),
                         'INDENTATION': BoardMatrix(matType='INDENTATION')}

        for i in range(4):
            for j in range(4):
                pieceIndex = simple[(i,j)].item()
                if pieceIndex == 0:
                    continue
                else:
                    boardMatrices['OCCUPIED'].placePieceAt((i,j))
                    pieceString = self.indexToString[pieceIndex]
                    h = pieceString.split(" ")[0]
                    c = pieceString.split(" ")[1]
                    s = pieceString.split(" ")[2]
                    ind = pieceString.split(" ")[3]
                    if h == 'TALL':
                        boardMatrices['HEIGHT'].placePieceAt((i,j))
                    if c == 'BLACK':
                        boardMatrices['COLOR'].placePieceAt((i, j))
                    if s == 'SQUARE':
                        boardMatrices['SHAPE'].placePieceAt((i, j))
                    if ind == 'INDENTED':
                        boardMatrices['INDENTATION'].placePieceAt((i, j))

        return boardMatrices

    #Creates a list of all valid combinations of free indeces and pieces. returns list of tuples: [(indx, piece), (indx, piece) ... ]
    def collectValidMoves(self):
        validIndeces = [None] #Start of the game has no indeces
        if not self.simpleRep.sum() == 0 or None in self.piecePool.values():
            validIndeces = (self.simpleRep == 0).nonzero().tolist()

        validPieces = [p for p,v in self.piecePool.items() if v != None]
        if len(validPieces) < 1:
            validPieces = [None] #End of game has no valid pieces, only 1 index
        result = []
        for index in validIndeces:
            for piece in validPieces:
                result.append((index, piece))

        return result



    def getBoardMatrices(self):
        return self.boardMatrices

    def getPiecePool(self):
        return self.piecePool

    def takePieceFromPool(self, pieceString):
        piece = self.piecePool[pieceString]
        self.piecePool[pieceString] = None
        return piece

    def getBoardMatriceFor(self, boardString):
        return self.boardMatrices[boardString]

    def placePieceAt(self, piece, position):

        if piece in self.piecePool:
            raise Exception(f'PIECE {piece} IN PIECE POOL')

        if self.boardMatrices['OCCUPIED'][position] == 1:
            raise Exception(f'POSITION {position} ALREADY TAKEN')

        self.simpleRep[position] = self.pieceToIndex[piece]
        #print(self.simpleRep)

        HEIGHT = f'{piece}'.split(' ')[0]
        COLOR = f'{piece}'.split(' ')[1]
        SHAPE = f'{piece}'.split(' ')[2]
        INDENTATION = f'{piece}'.split(' ')[3]

        self.boardMatrices['OCCUPIED'].placePieceAt(position)

        if HEIGHT == 'TALL':
            self.boardMatrices['HEIGHT'].placePieceAt(position)
        if COLOR == 'BLACK':
            self.boardMatrices['COLOR'].placePieceAt(position)
        if SHAPE == 'SQUARE':
            self.boardMatrices['SHAPE'].placePieceAt(position)
        if INDENTATION == 'INDENTED':
            self.boardMatrices['INDENTATION'].placePieceAt(position)

        if self.isWinningMove(position):
            self.isWon = True
            self.isDone = True
        elif self.isDraw():
            self.isDone = True

    def isWinningMove(self, position):
        #Checking Rows
        for i in range(4):
            if self.boardMatrices['OCCUPIED'][i].sum() == 4:
                # check the rows in each type
                rowHeight = self.boardMatrices['HEIGHT'][i].sum()
                rowColor = self.boardMatrices['COLOR'][i].sum()
                rowShape = self.boardMatrices['SHAPE'][i].sum()
                rowIndentation = self.boardMatrices['INDENTATION'][i].sum()
                if rowHeight == 0 or rowHeight == 4:
                    return True
                if rowColor == 0 or rowColor == 4:
                    return True
                if rowShape == 0 or rowShape == 4:
                    return True
                if rowIndentation == 0 or rowIndentation == 4:
                    return True

        #Checking Columns
        transposedOccMats = self.boardMatrices['OCCUPIED'].t()
        for i in range(4):
            if transposedOccMats[i].sum() == 4:
                #check the columns in each type
                colHeight = self.boardMatrices['HEIGHT'].t()[i].sum()
                colColor = self.boardMatrices['COLOR'].t()[i].sum()
                colShape = self.boardMatrices['SHAPE'].t()[i].sum()
                colIndentation = self.boardMatrices['INDENTATION'].t()[i].sum()
                if colHeight == 0 or colHeight == 4:
                    return True
                if colColor == 0 or colColor == 4:
                    return True
                if colShape == 0 or colShape == 4:
                    return True
                if colIndentation == 0 or colIndentation == 4:
                    return True

        #checking diagonal
        if self.boardMatrices['OCCUPIED'].diag().sum() == 4:
            #checking diagonal in each type
            diagHeight = self.boardMatrices['HEIGHT'].diag().sum()
            diagColor = self.boardMatrices['COLOR'].diag().sum()
            diagShape = self.boardMatrices['SHAPE'].diag().sum()
            diagIndentation = self.boardMatrices['INDENTATION'].diag().sum()
            if diagHeight == 0 or diagHeight == 4:
                return True
            if diagColor == 0 or diagColor == 4:
                return True
            if diagShape == 0 or diagShape == 4:
                return True
            if diagIndentation == 0 or diagIndentation == 4:
                return True

        #checking reverse diagonal
        if self.boardMatrices['OCCUPIED'].rot90().diag().sum() == 4:
            # checking reverse diagonal in each type
            revdiagHeight = self.boardMatrices['HEIGHT'].rot90().diag().sum()
            revdiagColor = self.boardMatrices['COLOR'].rot90().diag().sum()
            revdiagShape = self.boardMatrices['SHAPE'].rot90().diag().sum()
            revdiagIndentation = self.boardMatrices['INDENTATION'].rot90().diag().sum()
            if revdiagHeight == 0 or revdiagHeight == 4:
                return True
            if revdiagColor == 0 or revdiagColor == 4:
                return True
            if revdiagShape == 0 or revdiagShape == 4:
                return True
            if revdiagIndentation == 0 or revdiagIndentation == 4:
                return True

        return False

    def isDraw(self):
        if not self.isWon and self.boardMatrices['OCCUPIED'].getMatrix().sum() == 16:
            return True

        return False


class BoardMatrix():

    #Create an empty 4x4 matrix
    def __init__(self, matrix=None, matType=''):

        self.matrix = matrix

        if self.matrix == None:
            self.matrix = torch.zeros((4, 4), dtype=torch.int32)


        self.matrixType = matType

    def __str__(self):
        return f'{self.matrixType}'

    def __getitem__(self, position):
        return self.matrix[position]

    def t(self):
        return self.matrix.t()

    def diag(self):
        return self.matrix.diag()

    def rot90(self):
        return torch.rot90(self.matrix)

    def getMatrix(self):
        return self.matrix

    def getType(self):
        return self.matrixType

    def placePieceAt(self,position):
        v = self.matrix[position].item()
        if v == 0:
            self.matrix[position] = 1
        else:
            raise Exception(f'A piece is already in {self.matrixType} at position {position}')

    def getValueAt(self, position):
        return self.matrix[position]

