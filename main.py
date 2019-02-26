import Quarto_Game as QG
import MatrixReps
import Tests.TestingStrategy8Matrices as s8M
import GameGUI as gGUI

def main():
    gui = gGUI.GameGUI()
    gui.startGUI()


def runTests():
    s8M.testWins4Moves()

if __name__ == '__main__':
    runTests()
    main()

