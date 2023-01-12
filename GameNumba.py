import numpy as np
import random
import time
import numba
from numba import jit  # jit convertit une fonction python => fonction C


###################################################################

# PLayer 0 => Vertical    Player
# PLayer 1 => Horizontal  Player

# IdMove : code servant à identifier un coup particulier sur le jeu
# P   : id player 0/1
# x,y : coordonnées de la tuile, Player0 joue sur (x,y)+(x,y+1) et Player1 sur (x,y)+(x+1,y)

# convert: player,x,y <=> IDmove

# IDmove=123 <=> player 1 plays at position x = 2 and y = 3
# ce codage tient sur 8 bits !

StartingBoard  = np.zeros(144,dtype=np.uint8)


@jit(nopython=True)
def GetIDmove(player,x,y):
    return player * 100 + x * 10 + y

@jit(nopython=True)
def DecodeIDmove(IDmove):
    y = IDmove % 10
    x = int(IDmove/10) % 10
    player = int(IDmove / 100)
    return player,x,y

###################################################################

# Numba requiert des numpy array pour fonctionner

# toutes les données du jeu sont donc stockées dans 1 seul array numpy

# Data Structure  - numpy array de taille 144 uint8 :
# B[ 0- 63] List of possibles moves
# B[64-127] Gameboard (x,y) => 64 + x + 8*y
# B[-1] : number of possible moves
# B[-2] : reserved
# B[-3] : current player


@jit(nopython=True)   # pour x,y donné => retourne indice dans le tableau B
def iPxy(x,y):
    return 64 + 8 * y + x

@jit(nopython=True)
def _PossibleMoves(idPlayer,B):   # analyse B => liste des coups possibles par ordre croissant
    nb = 0

    #player V
    if idPlayer == 0 :
        for x in range(8):
            for y in range(7):
                p = iPxy(x,y)
                if B[p] == 0 and B[p+8] == 0 :
                    B[nb] = GetIDmove(0,x,y)
                    nb+=1
    # player H
    if idPlayer == 1 :
        for x in range(7):
            for y in range(8):
                p = iPxy(x,y)
                if B[p] == 0 and B[p+1] == 0 :
                    B[nb] = GetIDmove(1,x,y)
                    nb+=1

    B[-1] = nb


###################################################################

# Numba ne gère pas les classes...

# fonctions de gestion d'une partie
# les fonctions sans @jit ne sont pas accélérées

# Player 0 win => Score :  1
# Player 1 win => Score : -1


# def CreateNewGame()   => StartingBoard.copy()
# def CopyGame(B)       => return B.copy()

@jit(nopython=True)
def Terminated(B):
    return B[-1] == 0

@jit(nopython=True)
def GetScore(B):
    if B[-2] == 10 : return  1
    if B[-2] == 20 : return -1
    return 0


@jit(nopython=True)
def Play(B,idMove):
    player,x,y = DecodeIDmove(idMove)
    p = iPxy(x,y)

    B[p]   = 1
    if player == 0 : B[p+8] = 1
    else :           B[p+1] = 1

    nextPlayer = 1 - player

    _PossibleMoves(nextPlayer,B)
    B[-3] = nextPlayer

    if B[-1] == 0  :             # gameover
        B[-2] = (player+1)*10    # player 0 win => 10  / player 1 win => 20



################################################################

IA = 0
IAN0 = 0
IAN1 = 1

@jit(nopython=True)
def Playout(B):
    while B[-1] != 0: 
        if B[-3] == IA:
            id = random.randint(0,B[-1]-1)  # IA select random move        
        else:
            id = random.randint(0,B[-1]-1) 
        idMove = B[id]
        Play(B,idMove)


@numba.jit(nopython=True)
def ParrallelPlayoutSimu(nbSimus,Board):
    total_score = 0
    for simu in range(nbSimus):
        Bsimu = Board.copy()
        Playout(Bsimu)
        total_score += GetScore(Bsimu)
    return total_score


@numba.jit(nopython=True)
def ParrallelPlayoutSimuMCTS(nbSimus,Board,c):
    """
    nbSimus : nombre total de simulation = somme des simulations effectuées sur tous les coups possibles
    c : coef exploration / exploitation
    """
    nbPossiblemove = Board[-1]
    nbSimus *= nbPossiblemove                                             # We launch the same number of simus of traditional methods  

    UCB_scores = np.zeros(nbPossiblemove,dtype=np.float32) 
    scores = np.zeros(nbPossiblemove, dtype=np.int32)
    n_try = np.ones(nbPossiblemove,dtype=np.uint16)
    means = np.zeros(nbPossiblemove,dtype=np.float32)

    for simu in range(1,nbSimus+1):   
        Bmove = Board.copy()
        best_move_idx = np.argmax(UCB_scores)                             # Selection of best move given the highest value of UCB_scores
        idxBestMoveUCB = Bmove[best_move_idx] 
        Play(Bmove,idxBestMoveUCB)                                        # Play the best move
        Playout(Bmove)                                                    # Simulate a game following the best move
        scores[best_move_idx] += GetScore(Bmove)                          # Add the score of the simulation
        n_try[best_move_idx] += 1                                         # Add one for one simulation play with the current move idx
        means[best_move_idx] = scores[best_move_idx]/(n_try[best_move_idx] - 1)          # Retrieve the mean

        for i in range(nbPossiblemove):
            UCB_scores[i] = means[i] + c * np.sqrt(np.log(simu)/n_try[i])
        
    id = np.argmax(UCB_scores)
    return id

################################################################

@jit(nopython=True)
def PlayoutIANP(B,N):
    while B[-1] != 0: 
        if B[-3] == IA:   
            nbPossiblemove = B[-1]
            scores = np.zeros(nbPossiblemove,dtype=np.int32)            
            for move in range(nbPossiblemove - 1):   
                Bmove = B.copy()
                idMovetest = B[move]
                Play(Bmove,idMovetest)
                scores[move] = ParrallelPlayoutSimu(nbSimus=N,Board = Bmove) # Run N simulation of a game with a specific move and average those scores 
            id = np.argmax(scores)
        else:
            id = random.randint(0,B[-1]-1) 
        idMove = B[id]
        Play(B,idMove)



@jit(nopython=True)
def PlayoutIANPvsNpP(B,N,Np):
    while B[-1] != 0: 
        nbPossiblemove = B[-1]
        scores = np.zeros(nbPossiblemove,dtype=np.int32)            
        
        if B[-3] == IAN0:   
            for move in range(nbPossiblemove - 1):   
                Bmove = B.copy()
                idMovetest = B[move]
                Play(Bmove,idMovetest)
                scores[move] = ParrallelPlayoutSimu(nbSimus = N,Board = Bmove) 
                id = np.argmax(scores)
        if B[-3] == IAN1:   
            for move in range(nbPossiblemove - 1):   
                Bmove = B.copy()
                idMovetest = B[move]
                Play(Bmove,idMovetest)
                scores[move] = ParrallelPlayoutSimu(nbSimus = Np,Board = Bmove) 
                id = np.argmin(scores)

        idMove = B[id]
        Play(B,idMove)


@jit(nopython=True)
def PlayoutMCTS(B,nbSimus,c):
    while B[-1] != 0: 
        if B[-3] == IA:
            id = ParrallelPlayoutSimuMCTS(nbSimus = nbSimus,Board = B,c = c)  
        else:
            id = random.randint(0,B[-1]-1) 
        idMove = B[id]
        Play(B,idMove)


_PossibleMoves(0,StartingBoard) 

#########################################
#
#   Fonctions de Simulation des Parties

@numba.jit(nopython=True, parallel=True)
def ParralelPlayout(nbGames):
    gain_IA,gain_Player = 0,0
    for _ in numba.prange(nbGames):
        B = StartingBoard.copy()
        Playout(B)
        if GetScore(B) > 0: gain_IA += 1 
        else : gain_Player += 1

    print("gainIA : ", 100*gain_IA/nbGames ,"%","gainPlayer : ", 100*gain_Player/nbGames,"%")

@numba.jit(nopython=True, parallel=True)
def ParralelPlayoutIANP(nbGames,N):
    gain_IA,gain_Player = 0,0
    for _ in numba.prange(nbGames):
        B = StartingBoard.copy()
        PlayoutIANP(B,N)
        if GetScore(B) > 0: gain_IA += 1 
        else : gain_Player += 1
    print("gainIA : ", 100*gain_IA/nbGames ,"%","gainPlayer : ", 100*gain_Player/nbGames,"%")

@numba.jit(nopython=True, parallel=True)
def ParralelPlayoutIANPvsNpP(nbGames,N,Np):
    gain_IAN,gain_IANp = 0,0
    for _ in numba.prange(nbGames):
        B = StartingBoard.copy()
        PlayoutIANPvsNpP(B,N,Np)
        if GetScore(B) > 0: gain_IAN += 1 
        else : gain_IANp += 1

    print("gainIA",N,"P :", 100*gain_IAN/nbGames ,"%","gainIA",Np,"P : ", 100*gain_IANp/nbGames,"%")

@numba.jit(nopython=True, parallel=False)
def ParralelPlayoutIAMCTS(nbGames,nbSimus,c):
    gain_IAMCTS,gain_Player = 0,0
    for i in numba.prange(nbGames):
        B = StartingBoard.copy()
        PlayoutMCTS(B,nbSimus,c)
        if GetScore(B) > 0: gain_IAMCTS += 1 
        else : gain_Player += 1

    print("gain_IA_MCTS : ", 100*gain_IAMCTS/nbGames ,"%","gainPlayer : ", 100*gain_Player/nbGames,"%")


##################################################################
#
#   for demo only - do not use for computation

def Print(B):
    for yy in range(8):
        y = 7 - yy
        s = str(y)
        for x in range(8):
            if     B[iPxy(x,y)] == 1 : s += '::'
            else:                      s += '[]'
        print(s)
    s = ' '
    for x in range(8): s += str(x)+str(x)
    print(s)


    nbMoves = B[-1]
    print("Possible moves :", nbMoves);
    s = ''
    for i in range(nbMoves):
        s += str(B[i]) + ' '
    print(s)

def PlayoutDebug(B,verbose=False,display = True):
    if display: Print(B)
    while not Terminated(B):
        id = random.randint(0,B[-1]-1)
        idMove = B[id]
        player,x,y = DecodeIDmove(idMove)
        Play(B,idMove)
        if display:
            print("Playing : ",idMove, " -  Player: ",player, "  X:",x," Y:",y)
            Print(B)
            print("---------------------------------------")


# ################################################################
# #
# #  Version Debug Demo pour affichage et test

# B = StartingBoard.copy()
# PlayoutDebug(B,True)
# print("Score : ",GetScore(B))
# print("")

# for c in [0.2,0.4,0.6,0.8,1.0,1.2,1.6,2.0]:
#     ParralelPlayoutIAMCTS(1000,2000,c)

