import GameNumba as GN
from GameNumba import *

SIZE_DATASET = 100000
C_VALUE = 0.3

label = np.zeros((SIZE_DATASET,64),dtype = np.int8)
train_set = np.zeros((SIZE_DATASET,3,64),dtype = np.int8)

def add_to_database(B,idx_start,nbSimus,c):
    turn = 0
    while B[-1] != 0:
        idx = idx_start + turn 
        if idx >= SIZE_DATASET:
            return SIZE_DATASET

        if B[-3] == 0:
            id = GN.ParrallelPlayoutSimuMCTS(nbSimus = nbSimus,Board = B,c = c,player = 0)  
            train_set[idx,2,:] = np.zeros(64)       # append 0 list for player 0
        else:
            id = GN.ParrallelPlayoutSimuMCTS(nbSimus = nbSimus,Board = B,c = c,player = 1)  
            train_set[idx,2,:] = np.ones(64)      # append 1 list for player 1  
        
        idMove = B[id]           
        _,x,y = GN.DecodeIDmove(idMove)   
        idx_move = 8 * y + x
        label[idx,idx_move] = 1

        GN.Play(B,idMove)
        train_set[idx,0,:] = B[64:128]          # append the game world
        train_set[idx,1,:] = -(B[64:128]  - 1)  # "" ""  the game world negative
        turn += 1
    return turn


idx_start = 0
count = 0
while(True):
    B = StartingBoard.copy()
    idx_start += add_to_database(B,idx_start,nbSimus=100,c=C_VALUE)
    
    if idx_start >= SIZE_DATASET:
        break

    if count % 100 == 0:
        print(idx_start, " entry") 
    count += 1


np.save('data/X.npy', train_set)
np.save('data/Y.npy', label)

