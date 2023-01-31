import GameNumba as GN
from GameNumba import *

Ngames = 50
print("Test perf IA100P vs normal player on ", Ngames, " games")
GN.ParralelPlayoutIANPvsRand(Ngames,100)

print("Test perf IA10P vs IA100P on ", Ngames, " games")
GN.ParralelPlayoutIANPvsNpP(Ngames,10,100)

print("Test perf IA100P vs IA1KP on ", Ngames, " games")
GN.ParralelPlayoutIANPvsNpP(Ngames,100,1000)

print("Test perf IA1KP vs IA10KP on ", Ngames, " games")
GN.ParralelPlayoutIANPvsNpP(Ngames,1000,10000)

Nsimus = 100
Ngames = 50
print("Test perf IAMCTS vs normal player on ", Ngames, " games")
for c in [0.1,0.2,0.4,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:
    GN.ParralelPlayoutMCTSvsRand(Ngames,nbSimus=Nsimus,c=c)

print("Test perf IAMCTS vs IANp on ", Ngames, " games")
for c in [0.1,0.2,0.4,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:
    GN.ParralelPlayoutMCTSvsIANp(Ngames,nbSimus=Nsimus,c=c)
    
print("Test perf IAMCTS vs IADeep on ", Ngames, " games")
for c in [0.1,0.2,0.4,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:
    GN.ParralelPlayoutMCTSvsIADeep(Ngames,nbSimus=Nsimus,c=c)