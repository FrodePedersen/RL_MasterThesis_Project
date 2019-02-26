import MatrixReps
import Quarto_Game as QG

def testWins4Moves():
    qG = QG.GameBoard(MatrixReps.Strategy8Matrices())
    print(qG)

    piece = qG.takePieceFromPool('SHORT BLACK ROUND FILLED')
    piece2 = qG.takePieceFromPool('SHORT WHITE SQUARE INDENTED')
    piece3 = qG.takePieceFromPool('SHORT WHITE ROUND INDENTED')
    piece4 = qG.takePieceFromPool('SHORT WHITE ROUND FILLED')

    qG.placePieceAt(piece, (3, 3))
    print(qG.isWinningMove((3, 3)))
    qG.placePieceAt(piece2, (3, 2))
    print(qG.isWinningMove((3, 2)))
    qG.placePieceAt(piece3, (3, 1))
    print(qG.isWinningMove((3, 1)))
    qG.placePieceAt(piece4, (3, 0))

    if qG.isWinningMove((3, 0)):
        print(f'Testing testWins4Moves SUCCESS!')
    else:
        raise Exception(f'Testing testWins4Moves FAILED!')