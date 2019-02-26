from bitarray import bitarray
import numpy as np

class Piece():

    def __init__(self, attributes=bitarray(f'0'*8)):
        self.attributes = attributes

    def __str__(self):
        height = ''
        color = ''
        shape = ''
        indentation = ''
        if self.attributes[0] == 1:
            height = 'TALL'
        elif self.attributes[1] == 1:
            height = 'SHORT'
        else:
            raise Exception(f'Piece containing no height!')

        if self.attributes[2] == 1:
            color = 'BLACK'
        elif self.attributes[3] == 1:
            color = 'WHITE'
        else:
            raise Exception(f'Piece containing no color!')

        if self.attributes[4] == 1:
            shape = 'SQUARE'
        elif self.attributes[5] == 1:
            shape = 'ROUND'
        else:
            raise Exception(f'Piece containing no shape!')

        if self.attributes[6] == 1:
            indentation = 'INDENTED'
        elif self.attributes[7] == 1:
            indentation = 'FILLED'
        else:
            raise Exception(f'Piece containing no indentation')

        return f'{height} {color} {shape} {indentation}'

class GameBoard():

    def __init__(self, strategy):
        self.strategy = strategy
        self.boardMatrices = self.strategy.create8CNNMatrices()
        self.piecePool = self.strategy.create8DimensionalPieces()

    def __str__(self):
        outputString = 'Piece Pool:\n'
        for piece in self.piecePool:
            outputString += f'  {piece}\n'

        outputString += '\nMatrices:\n'

        for type, matrice in self.boardMatrices.items():
            outputString += f'  type: {type}\n{matrice.getMatrix()}\n'

        return outputString

    def getBoardMatrices(self):
        return self.boardMatrices

    def getPiecePool(self):
        return self.piecePool

    def takePieceFromPool(self, pieceString):
        piece = self.piecePool[pieceString]
        del self.piecePool[pieceString]
        return piece

    def getBoardMatriceFor(self, boardString):
        return self.boardMatrices[boardString]

    def placePieceAt(self, piece, position):
        self.strategy.placePieceAt(piece, position, self.boardMatrices)

    def isWinningMove(self, position):
        return self.strategy.isWinningMove(self.boardMatrices, position)

class BoardMatrix():

    #Create an empty 4x4 matrix
    def __init__(self, matrix=None, matType=''):

        self.matrix = np.zeros((4,4), dtype=int)

        if matrix != None:
            self.matrix = matrix

        self.matrixType = matType

    def __str__(self):
        return f'{self.matrixType}'

    def getMatrix(self):
        return self.matrix

    def getType(self):
        return self.matrixType

    def placePieceAt(self,position):
        if self.matrix[position] == 0:
            self.matrix[position] = 1
        else:
            raise Exception(f'A piece is already in {self.matrixType} at position {position}')

    def getValueAt(self, position):
        return self.matrix[position]

