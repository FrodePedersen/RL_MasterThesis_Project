import MatrixReps
import Quarto_Game as QG
import torch

def testBaseLayout():
    qG = QG.GameBoard()

    piece = qG.takePieceFromPool('SHORT BLACK ROUND FILLED')
    piece2 = qG.takePieceFromPool('SHORT WHITE SQUARE INDENTED')
    piece3 = qG.takePieceFromPool('SHORT WHITE ROUND INDENTED')
    piece4 = qG.takePieceFromPool('SHORT WHITE ROUND FILLED')

    qG.placePieceAt(piece, (3, 3))
    #print(qG.isWinningMove((3, 3)))
    qG.placePieceAt(piece2, (3, 2))
    #print(qG.isWinningMove((3, 2)))
    qG.placePieceAt(piece3, (3, 1))
    #print(qG.isWinningMove((3, 1)))
    qG.placePieceAt(piece4, (3, 0))
    #print(qG.isWinningMove((3, 0)))
    #hMat = qG.getBoardMatriceFor('HEIGHT')
    #print(f'{hMat.getMatrix()}')

    if qG.isWinningMove((3, 0)):
        print(f'Testing testBaseLayout SUCCESS!')
    else:
        raise Exception(f'Testing testBaseLayout FAILED!')

def test_tookPiece():
    qG = QG.GameBoard()

    qG.takePieceFromPool('SHORT BLACK ROUND FILLED')

    if len(list(qG.getPiecePool())) == 15:
        print(f'Testing test_tookPiece SUCCESS!')
    else:
        raise Exception(f'Testing test_tookPiece FAILED!')

def test_took4Pieces():
    qG = QG.GameBoard()

    piece = qG.takePieceFromPool('SHORT BLACK ROUND FILLED')
    piece2 = qG.takePieceFromPool('SHORT WHITE SQUARE INDENTED')
    piece3 = qG.takePieceFromPool('SHORT WHITE ROUND INDENTED')
    piece4 = qG.takePieceFromPool('SHORT WHITE ROUND FILLED')

    if len(list(qG.getPiecePool())) == 12:
        print(f'Testing test_took4Pieces SUCCESS!')
    else:
        raise Exception(f'Testing test_took4Pieces FAILED!')

def test_placeSamePiece():
    #Needs to throw ERROR
    qG = QG.GameBoard()

    piece = qG.takePieceFromPool('SHORT BLACK ROUND FILLED')

    qG.placePieceAt(piece, (0,0))

    try:
        qG.placePieceAt(piece, (1, 2))
        print(f'Testing test_placeSamePiece FAILED!')
    except:
        print(f'Testing test_placeSamePiece SUCCESS!')


def test_colWin4():

    qG = QG.GameBoard()

    piece = qG.takePieceFromPool('SHORT BLACK ROUND FILLED')
    piece2 = qG.takePieceFromPool('SHORT WHITE SQUARE INDENTED')
    piece3 = qG.takePieceFromPool('SHORT WHITE ROUND INDENTED')
    piece4 = qG.takePieceFromPool('SHORT WHITE ROUND FILLED')

    qG.placePieceAt(piece, (0, 0))
    qG.placePieceAt(piece2, (1, 0))
    qG.placePieceAt(piece3, (2, 0))
    qG.placePieceAt(piece4, (3, 0))

    if qG.isWinningMove((3, 0)):
        print(f'Testing test_colWin4 SUCCESS!')
    else:
        raise Exception(f'Testing test_colWin4 FAILED!')

def test_isDone():
    qG = QG.GameBoard()

    piece = qG.takePieceFromPool('SHORT BLACK ROUND FILLED')
    piece2 = qG.takePieceFromPool('SHORT WHITE SQUARE INDENTED')
    piece3 = qG.takePieceFromPool('SHORT WHITE ROUND INDENTED')
    piece4 = qG.takePieceFromPool('SHORT WHITE ROUND FILLED')

    qG.placePieceAt(piece, (0, 0))
    qG.placePieceAt(piece2, (1, 0))
    qG.placePieceAt(piece3, (2, 0))
    qG.placePieceAt(piece4, (3, 0))

    if qG.isDone:
        print(f'Testing test_isDone SUCCESS!')
    else:
        raise Exception(f'Testing test_isDone FAILED!')

def test_draw():

    qG = QG.GameBoard()

    p1 = qG.takePieceFromPool('SHORT WHITE ROUND FILLED')
    p2 = qG.takePieceFromPool('SHORT WHITE ROUND INDENTED')
    p3 = qG.takePieceFromPool('SHORT WHITE SQUARE FILLED')
    p4 = qG.takePieceFromPool('SHORT WHITE SQUARE INDENTED')
    p5 = qG.takePieceFromPool('SHORT BLACK ROUND FILLED')
    p6 = qG.takePieceFromPool('SHORT BLACK ROUND INDENTED')
    p7 = qG.takePieceFromPool('SHORT BLACK SQUARE FILLED')
    p8 = qG.takePieceFromPool('SHORT BLACK SQUARE INDENTED')
    p9 = qG.takePieceFromPool('TALL WHITE ROUND FILLED')
    p10 = qG.takePieceFromPool('TALL WHITE ROUND INDENTED')
    p11 = qG.takePieceFromPool('TALL WHITE SQUARE FILLED')
    p12 = qG.takePieceFromPool('TALL WHITE SQUARE INDENTED')
    p13 = qG.takePieceFromPool('TALL BLACK ROUND FILLED')
    p14 = qG.takePieceFromPool('TALL BLACK ROUND INDENTED')
    p15 = qG.takePieceFromPool('TALL BLACK SQUARE FILLED')
    p16 = qG.takePieceFromPool('TALL BLACK SQUARE INDENTED')

    qG.placePieceAt(p16, (0, 0))
    qG.placePieceAt(p1, (0, 1))
    qG.placePieceAt(p15, (0, 2))
    qG.placePieceAt(p2, (0, 3))
    qG.placePieceAt(p9, (1, 0))
    qG.placePieceAt(p8, (1, 1))
    qG.placePieceAt(p10, (1, 2))
    qG.placePieceAt(p7, (1, 3))
    qG.placePieceAt(p14, (2, 0))
    qG.placePieceAt(p13, (2, 1))
    qG.placePieceAt(p5, (2, 2))
    qG.placePieceAt(p12, (2, 3))
    qG.placePieceAt(p4, (3, 0))
    qG.placePieceAt(p6, (3, 1))
    qG.placePieceAt(p11, (3, 2))
    qG.placePieceAt(p3, (3, 3))

    complex = qG.translateSimpleToComplex(qG.simpleRep)
    '''
    for key, mat in complex.items():
        print(f'complexMats: {key}, {mat.getMatrix()}')
    '''

    if qG.isDone == True:
        print(f'Testing test_draw SUCCESS!')
    else:
        raise Exception(f'Testing test_draw FAILED!')




def testAll():
    testBaseLayout()
    test_tookPiece()
    test_took4Pieces()
    test_placeSamePiece()
    test_colWin4()
    test_isDone()
    test_draw()

if __name__ == '__main__':
    testAll()

