import Quarto_Game as QG
import MatrixReps
import Tests.TestingStrategy8Matrices as s8M
import Tests.TestingQuarto as tQ
import GameGUI as gGUI
import sys
import argparse
import rAgent
import TDLambdaAgent as TDLA
import random

import torch

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
            players = ('human', rAgent.rAgent())
        else:
            players = (rAgent.rAgent(), 'human')
    if args.v:
        players = (rAgent.rAgent(), rAgent.rAgent())
    if args.t:
        agent = selectAgentIndex(int(args.t[0]))
        nEpisodes = 10
        trainAgent(agent, None, nEpisodes, seed)
    else:
        playManualGame(players, qG, seed)

def selectAgentIndex(index):
    iToA = {0: rAgent.rAgent(), 1: TDLA.TDLambdaAgent()}

    return iToA[index]


def playTrainingGame(agent, envAgent, qG):
    placementPiece = None
    playerInTurn = bool(random.getrandbits(1))
    agent = agent
    envAgent = envAgent

    z = 0

    while qG.isDone != True:
        #Training Agent
        if int(playerInTurn) == 0:
            (placement, piece) = agent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                qG.placePieceAt(placementPiece, placement)

            if qG.isDone:
                break

            if piece != None:
                placementPiece = qG.takePieceFromPool(piece)

        #EnvAgent
        else:
            (placement, piece) = envAgent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                qG.placePieceAt(placementPiece, placement)

            if qG.isDone:
                # playerInTurn = not playerInTurn
                break

            if piece != None:
                placementPiece = qG.takePieceFromPool(piece)

        playerInTurn = not playerInTurn

    qG.printGame()
    if qG.isWon:
        return int(playerInTurn)
    else:
        return -1

def playManualGame(players, qG, seed):
    random.seed(seed)
    startOfGame = True
    placementPiece = None
    playerInTurn = False
    agent = None
    if players[0] != "human":
        agent = players[0]
    elif players[1] != "human":
        agent = players[1]

    agent.setBoard(qG)

    while qG.isDone != True:
        #print(f'PLAYER TURN: {int(playerInTurn)}')
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

        playerInTurn = not playerInTurn
        startOfGame = False

    qG.printGame()
    if qG.isWon:
        print(f'WON by player {int(playerInTurn)}')
    else:
        return f'DRAW'

def trainAgent(agent, envAgent, nEpisodes, seed):
    random.seed(seed)

    for episode in range(nEpisodes):
        qG = QG.GameBoard()
        envAgent = selectAgentIndex(0)
        envAgent.setBoard(qG)
        agent.setBoard(qG)

        players = (agent, envAgent)

        result = playTrainingGame(agent, envAgent, qG)

        if result == -1:
            print(f'Game was a draw')
        else:
            print(f'{players[result]} WON')


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
    parser.add_argument('-t', nargs=1, help="sets the game up for training an Agent of type <agentIndex>")

    args = parser.parse_args()

    #runTests()
    main(args)

