from Quarto_Game import BoardMatrix
from Quarto_Game import Piece
#from bitarray import bitarray
import numpy as np

class Strategy8Matrices():

    def create8CNNMatrices(self):
        boardMatrices = {}
        boardMatrices['TALL'] = BoardMatrix(matType='TALL')
        boardMatrices['SHORT'] = BoardMatrix(matType='SHORT')
        boardMatrices['BLACK'] = BoardMatrix(matType='BLACK')
        boardMatrices['WHITE'] = BoardMatrix(matType='WHITE')
        boardMatrices['SQUARE'] = BoardMatrix(matType='SQUARE')
        boardMatrices['ROUND'] = BoardMatrix(matType='ROUND')
        boardMatrices['INDENTED'] = BoardMatrix(matType='INDENTED')
        boardMatrices['FILLED'] = BoardMatrix(matType='FILLED')
        return boardMatrices

    def create8DimensionalPieces(self):
        piecePool = {}
        # CODE FOR BITARRAY: HIGH LOW BLACK WHITE SQUARE ROUND INDENTED FILLED
        piecePool['SHORT WHITE ROUND FILLED'] = Piece(bitarray(f'01010101'))
        piecePool['SHORT WHITE ROUND INDENTED'] = Piece(bitarray(f'01010110'))
        piecePool['SHORT WHITE SQUARE FILLED'] = Piece(bitarray(f'01011001'))
        piecePool['SHORT WHITE SQUARE INDENTED'] = Piece(bitarray(f'01011010'))
        piecePool['SHORT BLACK ROUND FILLED'] = Piece(bitarray(f'01100101'))
        piecePool['SHORT BLACK ROUND INDENTED'] = Piece(bitarray(f'01100110'))
        piecePool['SHORT BLACK SQUARE FILLED'] = Piece(bitarray(f'01101001'))
        piecePool['SHORT BLACK SQUARE INDENTED'] = Piece(bitarray(f'01101010'))
        piecePool['TALL WHITE ROUND FILLED'] = Piece(bitarray(f'10010101'))
        piecePool['TALL WHITE ROUND INDENTED'] = Piece(bitarray(f'10010110'))
        piecePool['TALL WHITE SQUARED FILLED'] = Piece(bitarray(f'10011001'))
        piecePool['TALL WHITE SQUARED INDENTED'] = Piece(bitarray(f'10011010'))
        piecePool['TALL BLACK ROUND FILLED'] = Piece(bitarray(f'10100101'))
        piecePool['TALL BLACK ROUND INDENTED'] = Piece(bitarray(f'10100110'))
        piecePool['TALL BLACK SQUARE FILLED'] = Piece(bitarray(f'10101001'))
        piecePool['TALL BLACK SQUARE INDENTED'] = Piece(bitarray(f'10101010'))

        return piecePool

    def placePieceAt(self, piece, position, matrices):
        for k,v in matrices.items():
            if v.getValueAt(position) == 1:
                raise Exception(f'Invalid move on position {position}')

        HEIGHT = f'{piece}'.split(' ')[0]
        COLOR = f'{piece}'.split(' ')[1]
        SHAPE = f'{piece}'.split(' ')[2]
        INDENTATION = f'{piece}'.split(' ')[3]

        matrices[HEIGHT].placePieceAt(position)         # Insert into HEIGHT
        matrices[COLOR].placePieceAt(position)          # Insert into COLOR
        matrices[SHAPE].placePieceAt(position)          # Insert into SHAPE
        matrices[INDENTATION].placePieceAt(position)    # Insert into INDENTATION

    def isWinningMove(self, matrices, lastMovePosition):
        for k,v in matrices.items():
            mat = v.getMatrix()
            rowCheck = np.sum(mat[lastMovePosition[0]])
            columnCheck = np.sum(mat[lastMovePosition[1]])
            diagonalCheck1 = np.sum(mat.diagonal())
            diagonalCheck2 = np.sum(np.rot90(mat).diagonal())

            if  rowCheck == 4 or columnCheck == 4 or diagonalCheck1 == 4 or diagonalCheck2 == 4:
                return True

        return False