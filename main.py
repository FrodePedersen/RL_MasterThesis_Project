import Quarto_Game as QG
import NewQuartoGame as NQG
import neuralNetwork as NN
import MatrixReps
import Tests.TestingStrategy8Matrices as s8M
import Tests.TestingQuarto as tQ
import GameGUI as gGUI
import sys
import argparse
import rAgent
import TDLambdaAgent as TDLA
import TDLambdaAgentNew as TDLAN
import DQNAgent as DQN
import random
import ReplayMemory as RP
from ReplayMemory import Transition
from ReplayMemory import TransitionDQN
import torch
import torch.nn as nn
import torch.optim as optim
from statistics import mean
import time

def main(args):
    #gui = gGUI.GameGUI()
    #gui.startGUI()
    players = ()
    qG = QG.GameBoard()
    functionAproxModel = NN
    seed = random.randint(1,100)
    #agent = rAgent.rAgent()

    oponent = rAgent.rAgent()
    episode = 1
    #print(f"c: {args.c}")
    train = False
    testing = False
    fIndex = 1

    if args.n:
        qG = NQG
        newBoard = True

    if args.f:
        fIndex = int(args.f[0])

    agent = TDLAN.TDLambdaAgent(functionIndex(fIndex))

    if args.s:
        seed = args.s[0]
    if args.p:
        players = ('human', 'human')
        playManualGame(players, qG, seed)

    if args.t:
        train = True
        agent = selectAgentIndex(int(args.t[0]), functionIndex(fIndex))

    if args.v:
        amount = int(args.v[0])
        testing = True

    if args.l:
        agent = selectAgentIndex(int(args.l[0]), functionIndex(fIndex))
        checkpoint = torch.load(f'./{args.l[1]}.tar')
        episode = checkpoint['episode']
        agent.targetNN.load_state_dict(checkpoint['target_state_dict'])
        agent.currentNN.load_state_dict(checkpoint['target_state_dict'])
        agent.lparams = checkpoint['agent_lparams']
        #agent.checkpointNumber = checkpoint['checkpoint_number']
        #agent.targetNN.load_state_dict(checkpoint['target_state_dict'])
        #agent.currentNN.load_state_dict(checkpoint['target_state_dict'])

    if args.o:
        oponent = selectAgentIndex(int(args.o[0]), functionIndex(fIndex))
        checkpoint = torch.load(f'./{args.o[1]}.tar')
        episode = checkpoint['episode']
        oponent.targetNN.load_state_dict(checkpoint['target_state_dict'])
        oponent.currentNN.load_state_dict(checkpoint['target_state_dict'])
        oponent.lparams = checkpoint['agent_lparams']

    if args.a:
        #"Assigns lparams <gamma> <lambda> <alpha> <alpha_decay> <epsilon> <epsilon_decay> of an Agent"
        agent.lparams['gamma']  = torch.tensor(float(args.a[0]))
        agent.lparams['lambda']  = torch.tensor(float(args.a[1]))
        agent.lparams['alpha'] = torch.tensor(float(args.a[2]))
        agent.lparams['alpha_decay'] = torch.tensor(float(args.a[3]))
        agent.lparams['epsilon'] = torch.tensor(float(args.a[4]))
        agent.lparams['epsilon_decay'] = torch.tensor(float(args.a[5]))

    if args.c:
        if int(args.c[0]) == 0:
            players = ('human', agent)# rAgent.rAgent())
        else:
            players = (agent, 'human') # rAgent.rAgent()
        playManuelGameNEWQuarto(players, qG, seed)

    if train:
        nEpisodes = episode + 100000
        if newBoard:
            trainAgentNEW(agent, nEpisodes, seed, qG, fIndex, initEpisode=episode)
        else:
            trainAgent(agent, nEpisodes, seed, initEpisode=episode)

    elif testing:
        players = (agent, oponent)
        players[0].setBoard(qG)
        players[1].setBoard(qG)

        conductTest(players, qG, seed, amount)



def selectAgentIndex(index, functionAproxModel):
    iToA = {0: rAgent.rAgent(),
            1: TDLA.TDLambdaAgent(functionAproxModel),
            2: TDLAN.TDLambdaAgent(functionAproxModel),
            3: DQN.DQNAgent(functionAproxModel)}

    return iToA[index]

def functionIndex(index):
    model = {0: NN.CNN_Model(), 1: NN.relu_Model_endSigmoid(), 2: NN.softmax_Model()}[index]
    if torch.cuda.is_available():
        model = model.cuda()
    return model

def conductTest(players, qg, seed, amount):
    winner = []
    win_ratio = 0
    startTime = time.time()
    print(f'players0 {players[0]}')
    players[0].lparams['epsilon'] = torch.tensor([0.0])
    if str(players[1]) != 'RANDOM AGENT':
        players[1].lparams['epsilon'] = torch.tensor([0.0])
    for i in range(amount):
        gameStart = time.time()
        qG = qg.GameBoard()
        players[0].setBoard(qG)
        players[1].setBoard(qG)
        winner.append(playTestGame(players, qG, seed).split(" ")[0])
        print(f'testGame took {time.time() - gameStart} seconds')



    endTime = time.time()

    for i in range(len(winner)):
        print(winner[i])
        print(f'winner: {winner[i]}, player: {players[0]}')
        print(f'{winner[i] == players[0]}')
        if str(winner[i]) == str(players[0]):
            win_ratio += 1

    win_ratio = win_ratio / amount
    e = players[0].lparams['epsilon']
    print(f'win_ratio of tested player: {win_ratio}, with epsilon: {e}')
    print(f'Run time: {endTime - startTime}')


def playTestGame(players, qG, seed):
    print()
    playerInTurn = bool(random.getrandbits(1))
    print(f'STARTING PLAYER: {int(playerInTurn)}')
    winner = 'draw'
    placementPiece = None
    amountOfMoves = -1

   #print(f'players: {players}')

    while qG.isDone != True:
        #Agent Control
        (placement, pieceIdx) = players[int(playerInTurn)].act()
        amountOfMoves += 1
        #print(f'placement: {placement}, piece: {piece}, playerTurn: {players[int(playerInTurn)]}')
        if placement != None:
            placement = (placement[0], placement[1])
            newBoardRep = qG.placePieceAt(placementPiece, placement)
            # print(f'###BEFORE PLACEMENT: {qG.boardMatrices}')
            qG.storeState(boardmat=newBoardRep)
            # print(f'###AFTER PLACEMENT!!: {qG.boardMatrices}')
            if qG.isWinningMove(newBoardRep):
                winner = str(players[int(playerInTurn)]) + " player: " + str(int(playerInTurn))
                qG.isDone = True
                break

        if pieceIdx != None:
            newPiecePool, newPickedPieceRep = qG.takePieceFromPool(pieceIdx)
            qG.storeState(piecePool=newPiecePool, pickedPieceRep=newPickedPieceRep)
            placementPiece = pieceIdx
        #Missing termination condition?!

        if qG.isDraw():
            break

        playerInTurn = not playerInTurn
        #startOfGame = False

    #qG.printGame()
    if winner != 'draw':
        print(f'Game is WON by: {winner}')
    elif winner == 'draw':
        print(f'Game is a DRAW!')
    else:
        raise Exception(f'Invalid winner result')

    print(f'Amount of Moves: {amountOfMoves}')
    return winner

def playTrainingGame(agent, envAgent, qG, placementPiece):
    placementPiece = placementPiece
    playerInTurn = bool(random.getrandbits(1))
    agent = agent
    envAgent = envAgent
    agentTookTurn = False
    envTookFollowupTurn = False

    winner = -1
    turnNumber = 0
    z_weights = []
    S = None

    while qG.isDone != True:
        reward = torch.tensor([0.0])
        turnNumber += 1
        #Training Agent
        if int(playerInTurn) == 0:
            S = qG.translateComplexToTensors(qG.translateSimpleToComplex(qG.calculateSymmetries(qG.getBoardStateSimple())))

            #print(f'S TRAINING: {S}')

            agentTookTurn = True
            g = agent.lparams['gamma']
            a = agent.lparams['alpha']
            l = agent.lparams['lambda']

            (placement, piece) = agent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                newSimpleBoardRep, newBoardMats = qG.placePieceAt(placementPiece, placement)
                qG.storeState(boardmats=newBoardMats, simpleBoardRep=newSimpleBoardRep)
                if qG.isWinningMove(newBoardMats):
                    winner = 1
                    reward = torch.tensor([1.0])
                    qG.isDone = True

            if piece != None:
                placementPiece, newPiecePool, newPieceRep, newPickedPieceRep = qG.takePieceFromPool(piece)
                qG.storeState(piecePool=newPiecePool, simplePieceRep=newPieceRep, simplePickedPieceRep=newPickedPieceRep)

            inputTens = S
            agent.currentNN.zero_grad()
            v_Sw = agent.currentNN(inputTens)
            #print(f'v_Sw: {v_Sw}')
            v_Sw.backward()

            if len(z_weights) < 1:
                #z_weights = []
                for par in agent.currentNN.parameters():
                    z_weights.append(par.grad.clone())
            else:
                for par, z_i in zip(agent.currentNN.parameters(), range(len(z_weights))):
                    z_weights[z_i] = g * l * z_weights[z_i] + par.grad.clone()

        #EnvAgent
        else:
            (placement, piece) = envAgent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                newSimpleBoardRep, newBoardMats = qG.placePieceAt(placementPiece, placement)
                qG.storeState(boardmats=newBoardMats, simpleBoardRep=newSimpleBoardRep)
                if qG.isWinningMove(newBoardMats):
                    winner = 0
                    qG.isDone = True


            if piece != None:
                placementPiece, newPiecePool, newPieceRep, newPickedPieceRep = qG.takePieceFromPool(piece)
                qG.storeState(piecePool=newPiecePool, simplePieceRep=newPieceRep,
                              simplePickedPieceRep=newPickedPieceRep)

            if agentTookTurn:
                envTookFollowupTurn = True

        if agentTookTurn and (envTookFollowupTurn or qG.isDone):
            S_PRIME = qG.translateComplexToTensors(qG.translateSimpleToComplex(qG.calculateSymmetries(qG.getBoardStateSimple())))
            agentTookTurn = False
            envTookFollowupTurn = False
            transition = Transition(S, reward, S_PRIME, z_weights, qG.isDone)
            print(f'Turn Number: {turnNumber}')
            trainNetwork(transition, agent)# 5, optimizer=None)
            S = S_PRIME

        if qG.isDone or qG.isDraw():
            break

        playerInTurn = not playerInTurn

    #qG.printGame()
    return winner

def trainNetworkDQN(agent, qG, rpm, optimizer, batch_size):
    if len(rpm) < batch_size:
        return

    agent.currentNN.zero_grad()
    agent.targetNN.zero_grad()

    transitions = rpm.sample(batch_size)
    batch = TransitionDQN(*zip(*transitions))

    reward_batch = torch.cat(batch.reward)
    state_batch = torch.stack(batch.state)

    action_batch = None
    for a in batch.action:
        coordIdx = a[0]
        if coordIdx is None:
            coordIdx = 16
        else:
            coordIdx = coordIdx[0] * 4 + coordIdx[1]
        pieceIdx = a[1]
        if pieceIdx is None:
            pieceIdx = 16
        else:
            pieceIdx = a[1] - 1

        #print(f'coord: {a[0]}, pieceIdx: {a[1]}')
        #print(f'coordIdx: {coordIdx}, pieceIdx {pieceIdx}')
        if action_batch is None:
            action_batch = torch.tensor([coordIdx * 16 + pieceIdx])
        else:
            action_batch = torch.cat([action_batch, torch.tensor([coordIdx * 16 + pieceIdx])])

    mask_batch = torch.stack(batch.mask)
    next_state_batch = torch.stack(batch.next_state)
    terminal_batch = torch.tensor(list(batch.terminal))
    agent.currentNN.train()
    if torch.cuda.is_available():
        state_batch = state_batch.cuda()
        mask_batch = mask_batch.cuda()
        action_batch = action_batch.cuda()
        reward_batch = reward_batch.cuda()
        terminal_batch = terminal_batch.detach()
        terminal_batch = terminal_batch.float().cuda()

    state_action_values = agent.currentNN(state_batch, mask_batch)#.gather(1, action_batch) #Get the action currentNN chose of current state. Q(s,a)

    Q_s_a_Current = None
    for i in range(state_action_values.size()[0]):
        if Q_s_a_Current is None:
            Q_s_a_Current = torch.tensor([state_action_values[i][action_batch[i]]])
            if torch.cuda.is_available():
                Q_s_a_Current = Q_s_a_Current.cuda()
        else:
            a = torch.stack([state_action_values[i][action_batch[i]]])
            if torch.cuda.is_available():
                a = a.cuda()
            Q_s_a_Current = torch.cat([Q_s_a_Current, a])

    y = torch.zeros(reward_batch.size())
    if torch.cuda.is_available():
        y = y.cuda()
        #print(f'Terminal_batch: {terminal_batch}')
        #print(f'Terminal_batch.nonzero: {terminal_batch.nonzero()}')
        #print(f'reward_batch: {reward_batch}')
        #print(f'y: {y}')
        y[terminal_batch.nonzero()] = reward_batch[terminal_batch.nonzero()]
    else:
        y[terminal_batch.nonzero()] = reward_batch[terminal_batch.nonzero()]


    next_state_masks = agent.find_mask_vector(next_state_batch)
    agent.targetNN.eval()
    if torch.cuda.is_available():
        next_state_batch = next_state_batch.cuda()
        next_state_masks = next_state_masks.cuda()
    Q_s_a_After = torch.abs(agent.lparams['gamma'] * torch.max(agent.targetNN(next_state_batch, next_state_masks), 1)[0])

    if torch.cuda.is_available():
        y[(terminal_batch == 0).nonzero()] = reward_batch[(terminal_batch == 0).nonzero()] + Q_s_a_After[(terminal_batch == 0).nonzero()]
    else:
        y[(terminal_batch == 0).nonzero()] = reward_batch[(terminal_batch == 0).nonzero()] +  Q_s_a_After[(terminal_batch == 0).nonzero()]

    #Compute MSELoss
    loss = nn.MSELoss()
    output = loss(torch.abs(Q_s_a_Current), y)
    #print(f'loss: {output}')
    if torch.cuda.is_available():
        output = output.cuda()
    optimizer.zero_grad()
    output.backward()
    optimizer.step()

    return output
    #raise Exception


    #state = transition.state


def trainNetwork(transition, agent, qG):
    agent.currentNN.train()
    agent.targetNN.eval()


    # print(f'training: {i}')
    agent.currentNN.zero_grad()
    agent.targetNN.zero_grad()
    state = transition.state#qG.calculateSymmetries(transition.state)
    reward = transition.reward
    next_state = qG.calculateSymmetries(transition.next_state)
    next_state = next_state #Normalize input
    z_i = transition.z
    is_terminal = transition.terminal
	
    g = agent.lparams['gamma']
    a = agent.lparams['alpha']
	
    if torch.cuda.is_available():
        state = state.cuda()
        next_state = next_state.cuda()
        g = g.cuda()
        a = a.cuda()
        reward  = reward.cuda()

	#WEIRD.... ALL be is the SAME.
    be = state #agent.currentNN(state)
    af = agent.targetNN(next_state)
    #print(f'TORCH.CUDA AVAILABLE? {torch.cuda.is_available()}')
    #print(f'g: {g}, be: {be}, af: {af}')

    #print(f'be: {be}')
    if is_terminal:
        '''
        if reward > 0:
            delta_error = reward
        else:
        '''
        delta_error = reward - be
    else:
        delta_error = reward + (agent.lparams['gamma'] * af ) - be

    #print(f'reward: {reward}')
    #print(f'delta_error: {delta_error.view(-1)}')
    #a = agent.lparams['alpha']
    #print(f'Gradients 2222: {a * z_i[2].data * delta_error.view(-1)}')
    #rint(f'Gradients 3333: {a * z_i[3].data * delta_error.view(-1)}')
    #print(f'Gradients 4444: {a * z_i[4].data * delta_error.view(-1)}')
    #print(f'Gradients 5555: {a * z_i[5].data * delta_error.view(-1)}')
    #i = 0

    for paramf, z in zip(agent.currentNN.parameters(), z_i):
        if torch.cuda.is_available():
            z.data = z.data.cuda()
        #print(f'z_data: {z.data}')
        #print(f'z_0s: {(z == 0).nonzero().size()}')
        #print(f'z size(): {len(z.view(-1))}')
        paramf.data += agent.lparams['alpha'] * z.data * delta_error.view(-1)
        #print(f'WEIGHTS: {paramf}')
        #i += 1

def playManuelGameNEWQuarto(players, qG, seed):
    qG = qG.GameBoard()
    random.seed(seed)
    startOfGame = True
    placementPiece = None
    playerInTurn = False
    agent = None
    if players[0] != "human":
        agent = players[0]
    elif players[1] != "human":
        agent = players[1]

    if str(agent) != 'RANDOM AGENT':
        agent.lparams['epsilon'] = torch.tensor([0.0])
    agent.setBoard(qG)

    winner = 'draw'

    while qG.isDone != True:
        # print(f'PLAYER TURN: {int(playerInTurn)}')
        # Human control
        if players[int(playerInTurn)] == 'human':
            if not startOfGame:
                qG.printGame()
                placement = input(f"\place piece {placementPiece}, encoding: {qG.indexToPieceTensor[placementPiece].data}")
                placement = placement.split(",")
                placement = (int(placement[0]), int(placement[1]))
                newBoardRep = qG.placePieceAt(placementPiece, placement)
                # print(f'###BEFORE PLACEMENT: {qG.boardMatrices}')
                qG.storeState(boardmat=newBoardRep)
                if qG.firstMove:
                    qG.setSyms(placement)

                if qG.isWinningMove(newBoardRep):
                    winner = 'human'
                    # reward = torch.tensor([1.0])
                    qG.isDone = True
                    break

            qG.printGame()
            inp = int(input("\nSelect piece:"))
            newPiecePool, newPickedPieceRep = qG.takePieceFromPool(inp)
            qG.storeState(piecePool=newPiecePool, pickedPieceRep=newPickedPieceRep)
            placementPiece = inp
        # Agent Control
        else:
            #print(f'AGENT L-PARAMS: {agent.lparams}')
            (placement, pieceIdx) = agent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                newBoardRep = qG.placePieceAt(placementPiece, placement)
                # print(f'###BEFORE PLACEMENT: {qG.boardMatrices}')
                qG.storeState(boardmat=newBoardRep)
                if qG.firstMove:
                    qG.setSyms(placement)
                # print(f'###AFTER PLACEMENT!!: {qG.boardMatrices}')
                if qG.isWinningMove(newBoardRep):
                    winner = str(agent)
                    qG.isDone = True


            if pieceIdx != None:
                newPiecePool, newPickedPieceRep = qG.takePieceFromPool(pieceIdx)
                qG.storeState(piecePool=newPiecePool, pickedPieceRep=newPickedPieceRep)
                placementPiece = pieceIdx
            # Missing termination condition?!

        playerInTurn = not playerInTurn
        startOfGame = False

    qG.printGame()
    if winner != 'draw':
        print(f'Game is WON by {winner}')
    elif winner == 'draw':
        print(f'Game is a DRAW!')
    else:
        raise Exception(f'Invalid winner result')

def trainAgentNEW(agent, nEpisodes, seed, qg, fIndex, initEpisode=1):
    random.seed(seed)
    arpm = RP.AgentMemory(100)
    rpm = RP.ReplayMemoryDQN(1000)

    print(agent)
    if str(agent) == 'TDLambdaAgentNEW':
        envAgent = TDLAN.TDLambdaAgent(functionIndex(fIndex))
        envAgent.lparams = agent.lparams.copy()
    elif str(agent) == 'DQNAgent':
        envAgent = DQN.DQNAgent(functionIndex(fIndex))
        envAgent.lparams = agent.lparams.copy()


    arpm.push(envAgent)
    # print(f'AGENT LPARAMS: {agent.lparams}')
    placementPiece = None
    print(f'TYPE OF AGENT: {agent}')
    eTimeBegin = time.time()
    eTimeEnd = 0
    optimizer = optim.SGD(agent.currentNN.parameters(), lr=agent.lparams['alpha'], momentum=0.9)
    avg_loss = []
    for episode in range(initEpisode, nEpisodes):
        qG = None #qg.GameBoard() # HHHHHHUUUUUUUUUUUUSSSSSSSKKKKKKKKK at få tilbage til randomized initial!!

        while True:
            qG = qg.GameBoard()
            placementPiece = qG.randomInitOfGame()
            if not qG.isDone:
                break

        #print(f'GAME INIT: {torch.stack([qG.boardRep, qG.piecePoolRep, qG.pickedPieceRep])}')

        envAgent = arpm.sample()
        envAgent.setBoard(qG)
        envAgent.lparams['epsilon'] = torch.tensor([0.0])
        agent.setBoard(qG)
        e = agent.lparams['epsilon']
        # print(f'agent EPSILON: {e}')
        # rpm = RP.ReplayMemory(1000)
        #print(f'TYPE OF ADVERSARY: {envAgent}')


        #result = playTrainingGameNEW(agent, envAgent, qG, placementPiece)
        result = playTrainingGameDQN(agent, envAgent, qG, placementPiece, rpm) # DQN TEST


        loss = trainNetworkDQN(agent, qG, rpm, optimizer, batch_size=50)
        if loss is not None:
            avg_loss.append(loss)
        '''
        if result == -1:
            print(f'Game was a draw')
            # break
        elif result == 0:
            print(f'ADVESARY WON!')
        elif result == 1:
            print(f'{agent} WON!')
        else:
            raise Exception('INVALID RESULT')
        '''

        if str(agent) == 'TDLambdaAgentNEW':
            newAgent = TDLAN.TDLambdaAgent(functionIndex(fIndex))
            newAgent.lparams = agent.lparams.copy()
            newAgent.currentNN.load_state_dict(agent.currentNN.state_dict().copy())
            arpm.push(newAgent)
        elif str(agent) == 'DQNAgent':
            newAgent = DQN.DQNAgent(functionIndex(fIndex))
            newAgent.lparams = agent.lparams.copy()
            newAgent.currentNN.load_state_dict(agent.currentNN.state_dict().copy())
            arpm.push(newAgent)



        #print(f'Lparams: {agent.lparams}')
        if episode % 50 == 0:
            av_loss = torch.mean(torch.stack(avg_loss))
            print(f'avg_loss {av_loss}')
            agent.targetNN.load_state_dict(agent.currentNN.state_dict().copy())
            agent.targetNN.eval()
            print(f'ROUND {episode}')
            eTimeEnd = time.time() - eTimeBegin
            print(f'eTimeEnd: {eTimeEnd}')
            torch.save({
                'eTime': eTimeEnd,
                'target_state_dict': agent.targetNN.state_dict(),
                'current_state_dict': agent.currentNN.state_dict(),
                'episode': episode,
                'avg_loss': av_loss,
                'agent_lparams': agent.lparams.copy(),
                'checkpoint_number': episode / 50}, f"./modelTargets/DQN/un_restricted_1_h/agent{int(episode / 50)}.tar")
            eTimeBegin = time.time()

            avg_loss = []


        if agent.lparams['alpha'] > 0.1:
            agent.lparams['alpha'] *= agent.lparams['alpha_decay']

        if agent.lparams['epsilon'] > 0.1:
            agent.lparams['epsilon'] *= agent.lparams['epsilon_decay']

def playTrainingGameDQN(agent, envAgent, qG, placementPiece, rpm):
    placementPiece = placementPiece
    playerInTurn = bool(random.getrandbits(1))
    agent = agent
    envAgent = envAgent
    agentTookTurn = False
    envTookFollowupTurn = False

    agent.trainingAgent = True
    envAgent.trainingAgent = False

    winner = -1
    turnNumber = 0
    z_weights = []
    S = None
    S_PRIME = None
    action = None
    reward = torch.tensor([0.0])

    while qG.isDone != True:
        turnNumber += 1
        # print(f'TurnNumber Beging: {turnNumber}, playerInTurn: {int(playerInTurn)}')
        # Training Agent
        if int(playerInTurn) == 0:
            S = torch.stack([qG.boardRep.clone(), qG.piecePoolRep.clone(), qG.pickedPieceRep.clone()])

            move_scores, mask_vector = agent.act()
            if random.random() > agent.lparams['epsilon'].item():
                moveIdx = move_scores.argmax()
            else:
                mask = move_scores > move_scores.min()
                moveIdx = random.choice(mask.nonzero())
            (placement, pieceIdx) = agent.translateScoresToMove(moveIdx)
            action = (placement, pieceIdx)
            if placement is not None:
                placement = (placement[0], placement[1])
                newBoardRep = qG.placePieceAt(placementPiece, placement)
                qG.storeState(boardmat=newBoardRep)
                if qG.firstMove:
                    qG.setSyms(placement)
                if qG.isWinningMove(newBoardRep):
                    winner = 1
                    reward = torch.tensor([1.0])
                    qG.isDone = True

            if pieceIdx is not None:
                newPiecePool, newPickedPieceRep = qG.takePieceFromPool(pieceIdx)
                qG.storeState(piecePool=newPiecePool, pickedPieceRep=newPickedPieceRep)
                placementPiece = pieceIdx

            S_PRIME = torch.stack([qG.boardRep.clone(), qG.piecePoolRep.clone(), qG.pickedPieceRep.clone()])

            agentTookTurn = True
            envTookFollowupTurn = False

        # EnvAgent
        else:
            move_scores, mask_vector = envAgent.act()
            (placement, pieceIdx) = agent.translateScoresToMove(move_scores.argmax())
            if placement is not None:
                placement = (placement[0], placement[1])
                newBoardRep = qG.placePieceAt(placementPiece, placement)
                qG.storeState(boardmat=newBoardRep)

                if qG.firstMove:
                    qG.setSyms(placement)

                if qG.isWinningMove(newBoardRep):
                    winner = 0
                    # reward = torch.tensor([-1.0])
                    qG.isDone = True

            if pieceIdx is not None:
                newPiecePool, newPickedPieceRep = qG.takePieceFromPool(pieceIdx)
                qG.storeState(piecePool=newPiecePool, pickedPieceRep=newPickedPieceRep)
                placementPiece = pieceIdx

            if agentTookTurn:
                envTookFollowupTurn = True

        if (agentTookTurn and qG.isDone) or (envTookFollowupTurn):
            rpm.push(S, reward, action, mask_vector, S_PRIME, qG.isDone)  # push to rpm for DQN
            agentTookTurn = False

        if qG.isDone or qG.isDraw():
            break

        playerInTurn = not playerInTurn

    return winner

def playTrainingGameNEW(agent, envAgent, qG, placementPiece):
    placementPiece = placementPiece
    playerInTurn = bool(random.getrandbits(1))
    agent = agent
    envAgent = envAgent
    agentTookTurn = False
    envTookFollowupTurn = False

    agent.trainingAgent = True
    envAgent.trainingAgent = False


    winner = -1
    turnNumber = 0
    z_weights = []
    S = None
    S_PRIME = None
    action = None
    reward = torch.tensor([0.0])

    while qG.isDone != True:
        turnNumber += 1
        #print(f'TurnNumber Beging: {turnNumber}, playerInTurn: {int(playerInTurn)}')
        #Training Agent
        if int(playerInTurn) == 0:
            S = torch.stack([qG.boardRep.clone(), qG.piecePoolRep.clone(), qG.pickedPieceRep.clone()])
            #print(f'BOARD REPS NEW TRAINING: {S}')
            #print(f'S TRAINING: {S}')
            #agentTookTurn = True
            g = agent.lparams['gamma']
            a = agent.lparams['alpha']
            l = agent.lparams['lambda']

            (placement, pieceIdx) = agent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                newBoardRep = qG.placePieceAt(placementPiece, placement)
                qG.storeState(boardmat=newBoardRep)
                if qG.firstMove:
                    qG.setSyms(placement)
                if qG.isWinningMove(newBoardRep):
                    winner = 1
                    reward = torch.tensor([1.0])
                    qG.isDone = True

            if pieceIdx != None:
                newPiecePool, newPickedPieceRep = qG.takePieceFromPool(pieceIdx)
                qG.storeState(piecePool=newPiecePool, pickedPieceRep=newPickedPieceRep)
                placementPiece = pieceIdx

            #print(f'INPUT TENS: {S}')
            #print(f'SYMM: horizon, vertical {qG.horizontalSwap, qG.verticalSwap}')
            inputTens = qG.calculateSymmetries(S)
            inputTens = inputTens #normalize input
            #print(f'CALC SYMMETRIES?! {inputTens}')
            agent.currentNN.zero_grad()
            if torch.cuda.is_available():
                inputTens = inputTens.cuda()
            agent.currentNN.train()
            v_Sw = agent.currentNN(inputTens)
            #print(v_Sw)
            #print(f'v_Sw: {v_Sw}')
            v_Sw.backward()

            if len(z_weights) < 1:
                #z_weights = []
                for par in agent.currentNN.parameters():
                    z_weights.append(par.grad.clone())
            else:
                for par, z_i in zip(agent.currentNN.parameters(), range(len(z_weights))):
                    z_weights[z_i] = g * l * z_weights[z_i] + par.grad.clone()

            S_PRIME = torch.stack([qG.boardRep.clone(), qG.piecePoolRep.clone(), qG.pickedPieceRep.clone()])

            agentTookTurn = True
            envTookFollowupTurn = False

        #EnvAgent
        else:
            (placement, pieceIdx) = envAgent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                newBoardRep = qG.placePieceAt(placementPiece, placement)
                qG.storeState(boardmat=newBoardRep)

                if qG.firstMove:
                    qG.setSyms(placement)

                if qG.isWinningMove(newBoardRep):
                    winner = 0
                    #reward = torch.tensor([-1.0])
                    qG.isDone = True


            if pieceIdx != None:
                newPiecePool, newPickedPieceRep = qG.takePieceFromPool(pieceIdx)
                qG.storeState(piecePool=newPiecePool, pickedPieceRep=newPickedPieceRep)
                placementPiece = pieceIdx

            if agentTookTurn:
                envTookFollowupTurn = True

        if (agentTookTurn and qG.isDone) or (envTookFollowupTurn):
            transition = Transition(v_Sw, reward, S_PRIME, z_weights, qG.isDone)
            #print(f'turnNumber: {turnNumber}')
            trainNetwork(transition, agent, qG)
            agentTookTurn = False

        if qG.isDone or qG.isDraw():
            break

        playerInTurn = not playerInTurn

    #qG.printGame()
    return winner


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

    if str(agent) != 'RANDOM AGENT':
        agent.lparams['epsilon'] = torch.tensor([0.0])
    agent.setBoard(qG)

    winner = 'draw'

    while qG.isDone != True:
        #print(f'PLAYER TURN: {int(playerInTurn)}')
        #Human control
        if players[int(playerInTurn)] == 'human':
            if not startOfGame:
                qG.printGame()
                placement = input(f"\place piece {placementPiece}, encoding: {placementPiece.attributes}")
                placement = placement.split(",")
                placement = (int(placement[0]), int(placement[1]))
                newSimpleBoardRep, newBoardMats = qG.placePieceAt(placementPiece, placement)
                # print(f'###BEFORE PLACEMENT: {qG.boardMatrices}')
                qG.storeState(boardmats=newBoardMats, simpleBoardRep=newSimpleBoardRep)

                if qG.isWinningMove(newBoardMats):
                    winner = 'human'
                    # reward = torch.tensor([1.0])
                    qG.isDone = True
                    break

            qG.printGame()
            inp = int(input("\nSelect piece:"))
            placementPiece, newPiecePool, newPieceRep, newPickedPieceRep = qG.takePieceFromPool(qG.indexToString[inp])
            qG.storeState(piecePool=newPiecePool, simplePieceRep=newPieceRep,
                          simplePickedPieceRep=newPickedPieceRep)
        #Agent Control
        else:
            print(f'AGENT L-PARAMS: {agent.lparams}')
            (placement, piece) = agent.act()
            if placement != None:
                placement = (placement[0], placement[1])
                newSimpleBoardRep, newBoardMats = qG.placePieceAt(placementPiece, placement)
                # print(f'###BEFORE PLACEMENT: {qG.boardMatrices}')
                qG.storeState(boardmats=newBoardMats, simpleBoardRep=newSimpleBoardRep)
                # print(f'###AFTER PLACEMENT!!: {qG.boardMatrices}')
                if qG.isWinningMove(newBoardMats):
                    winner = str(agent)
                    qG.isDone = True

            if piece != None:
                placementPiece, newPiecePool, newPieceRep, newPickedPieceRep = qG.takePieceFromPool(piece)
                qG.storeState(piecePool=newPiecePool, simplePieceRep=newPieceRep,
                              simplePickedPieceRep=newPickedPieceRep)
            #Missing termination condition?!

        playerInTurn = not playerInTurn
        startOfGame = False

    qG.printGame()
    if winner != 'draw':
        print(f'Game is WON by {winner}')
    elif winner == 'draw':
        print(f'Game is a DRAW!')
    else:
        raise Exception(f'Invalid winner result')

def trainAgent(agent, nEpisodes, seed, initEpisode=1):
    random.seed(seed)
    arpm = RP.AgentMemory(100)
    arpm.push(agent)
    #print(f'AGENT LPARAMS: {agent.lparams}')
    placementPiece = None

    for episode in range(initEpisode, nEpisodes):
        qG = None
        while True:
            qG = QG.GameBoard()
            #print(f'INSIDE WHILE LOOP')
            placementPiece = qG.randomInitOfGame()
            if not qG.isDone:
                break

        envAgent = arpm.sample()
        envAgent.setBoard(qG)
        agent.setBoard(qG)
        #rpm = RP.ReplayMemory(1000)

        result = playTrainingGame(agent, envAgent, qG, placementPiece)

        print(f'ROUND {episode}')

        if result == -1:
            print(f'Game was a draw')
            #break
        elif result == 0:
            print(f'ADVESARY WON!')
        elif result == 1:
            print(f'{agent} WON!')
        else:
            raise Exception('INVALID RESULT')

        agent.targetNN.load_state_dict(agent.currentNN.state_dict())
        agent.targetNN.eval()

        newAgent = TDLA.TDLambdaAgent()
        newAgent.lparams = agent.lparams
        newAgent.currentNN.load_state_dict(agent.currentNN.state_dict())
        arpm.push(newAgent)

        if episode % 50 == 0:
            torch.save({
                'target_state_dict': agent.targetNN.state_dict(),
                'episode': episode,
                'agent_lparams': agent.lparams,
                'checkpoint_number': episode/50}, f"./modelTargets/TDLambda/Initial/NEWNN{int(episode/50)}.tar")

def runTests():
    #s8M.testWins4Moves()
    tQ.testAll()
    qG = QG.GameBoard()
    print(qG.printBoard())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', help="sets the game up for 2 human players", action='store_true')
    parser.add_argument('-c', nargs=1, help="sets the game up for 1 human player vs 1 computer player. Human goes <argument>")
    parser.add_argument('-v', nargs=1, help="sets the game up for 2 Agents <Agent1>, <Agent2>, <amount>")
    parser.add_argument('-s', nargs=1, help="sets the seed for the randomizer")
    parser.add_argument('-t', nargs=1, help="sets the game up for training an Agent of type <agentIndex>")
    parser.add_argument('-a', nargs=6, help="Assigns lparams <gamma> <lambda> <alpha> <alpha_decay> <epsilon> <epsilon_decay> of an Agent")
    parser.add_argument('-l', nargs=2, help="sets the game up using <agent Type> <agent Path>")
    parser.add_argument('-o', nargs=2, help="loads a model for the oponent")
    parser.add_argument('-n', help="Starts a version of the NEW QUARTO GAME")
    parser.add_argument('-f', nargs=1, help='Select Function Approximation Index')

    args = parser.parse_args()

    #runTests()
    main(args)

#Manual testing:
# -c 0 -l modelTargets/TDLambda/Initial/saveTargetNN199
# -c 0 -l modelTargets/TDLambda/Initial/saveTargetNN884
# -c 0 -l modelTargets/TDLambda/Initial/saveCurrentNN20

#MANUAL testing NEW
# -c 0 -n 1 -f 1

#Training model:
# -t 1 -l modelTargets/TDLambda/Initial/saveTargetNN884
# -t 1 -l modelTargets/rAgent/saveTargetNN6
# -t 1 -l modelTargets/TDLambda/Initial/saveDECAYNN2
# -t 1 -l modelTargets/TDLambda/Initial/NEWNN1009

#Training new:
#-t 2 -l 2 modelTargets/TDLambda/Corrected_eigth/agent1347 -n 1 -f 1

#TRAINING NEW:
#-t 2 -a 1 0 1 0.999 0.3 0.999 -n 1 -f 1 #TDLAN
#-t 3 -a 0.7 0.4 1 0.9999 0.3 0.9999 -n 1 -f 1 #DQN

#Training from scratch:
#-t 1 -a 1 0 1 0.999 0.3 0.999

#TRAINING DQN
#        g l a
#-t 3 -a 1 0 1 0.9999 0.3 0.9999 -n 1 -f 2

#Use new QuartoGame
#-n

#TESTING:
#-v 100 -l modelTargets/TDLambda/Initial/saveCurrentNN20

#-v 100 -l 1 modelTargets/TDLambda/Initial/NEWNN1479 -o 1 modelTargets/TDLambda/Initial/NEWNN1478