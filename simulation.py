import GameNumba as GN
from GameNumba import *

Ngames = 1
print("Test perf IARand vs normal player on ", Ngames, " games")
GN.ParralelPlayout(Ngames)

Ngames = 200
print("\n"*2)
print("Test perf IA100P vs normal player on ", Ngames, " games")
GN.ParralelPlayoutIANP(Ngames,100)

Ngames = 50
print("\n"*2)
print("Test perf IA10P vs IA100P on ", Ngames, " games")
GN.ParralelPlayoutIANPvsNpP(Ngames,10,100)

Ngames = 10
print("\n"*2)
print("Test perf IA100P vs IA1KP on ", Ngames, " games")
GN.ParralelPlayoutIANPvsNpP(Ngames,100,1000)

print("\n"*2)
print("Test perf IA1KP vs IA10KP on ", Ngames, " games")
GN.ParralelPlayoutIANPvsNpP(Ngames,1000,10000)

Ngames = 100
print("Test perf IARand vs normal player on ", Ngames, " games")
ParralelPlayoutIAMCTS(Ngames,nbSimus=100,c=0.5)