import Quarto_Game as QG
import MatrixReps
import Tests.TestingStrategy8Matrices as s8M
import Tests.TestingQuarto as tQ
import GameGUI as gGUI
import sys
import argparse
import rAgent

def main(args):
    #gui = gGUI.GameGUI()
    #gui.startGUI()
    players = ()
    qG = QG.GameBoard()
    seed = 23

    #print(f"c: {args.c}")
    if args.s:
        seed = args.s[0]
    if args.p:
        players = ('human', 'human')
    if args.c:
        if int(args.c[0]) == 0:
            players = ('human', rAgent.rAgent(qG, seed))
        else:
            players = (rAgent.rAgent(qG, seed), 'human')
    if args.v:
        players = (rAgent.rAgent(qG, seed), rAgent.rAgent(qG, seed))

    result = playGame(players, qG)
    if result.split(" ")[0] == "WON":
        print(f'Game is WON by player {result.split(" ")[1]}!')
    else:
        print(f'Game ended in a DRAW')

def playGame(players, qG):
    startOfGame = True
    placementPiece = None
    playerInTurn = False
    agent = None
    if players[0] != "human":
        agent = players[0]
    elif players[1] != "human":
        agent = players[1]

    print(playerInTurn)
    print(players[int(playerInTurn)])

    while qG.isDone != True:
        print(f'PLAYER TURN: {int(playerInTurn)}')
        #Human control
        if players[int(playerInTurn)] == 'human':
            if not startOfGame:
                qG.printGame()
                placement = input(f"\place piece {placementPiece}, encoding: {placementPiece.attributes}")
                placement = placement.split(",")
                placement = (int(placement[0]), int(placement[1]))
                qG.placePieceAt(placementPiece, placement)
                if qG.isDone:
                    playerInTurn = not playerInTurn
                    break

            qG.printGame()
            inp = int(input("\nSelect piece:"))
            placementPiece = qG.takePieceFromPool(qG.indexToString[inp])
        #Agent Control
        else:
            (placement, piece) = agent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                qG.placePieceAt(placementPiece, placement)

            if qG.isDone:
                #playerInTurn = not playerInTurn
                break

            if piece != None:
                placementPiece = qG.takePieceFromPool(piece)
            #Missing termination condition?!
            print(f"PLACEMENT {placement}, PIECE: {placementPiece}")

        playerInTurn = not playerInTurn
        startOfGame = False

    qG.printGame()
    if qG.isWon:
        return f'WON {int(playerInTurn)}'
    else:
        return f'DRAW'



def runTests():
    #s8M.testWins4Moves()
    tQ.testAll()
    qG = QG.GameBoard()
    print(qG.printBoard())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help="sets the game up for 2 human players", action='store_true')
    parser.add_argument('-c', nargs=1, help="sets the game up for 1 human player vs 1 computer player. Human goes <argument>")
    parser.add_argument('-v', nargs=2, help="sets the game up for 2 Agents <Agent1>, <Agent2>")
    parser.add_argument('-s', nargs=1, help="sets the seed for the randomizer")

    args = parser.parse_args()

    #runTests()
    main(args)

