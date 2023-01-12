import GameNumba as GN
from GameNumba import *

# Ngames = 1
# print("Test perf IARand vs normal player on ", Ngames, " games")
# GN.ParralelPlayout(Ngames)

Ngames = 1000
Nsimus = 10
print("\n"*2)
print("Test perf IA100P vs normal player on ", Ngames, " games")
GN.ParralelPlayoutIANP(Ngames,Nsimus)

# Ngames = 50
# print("\n"*2)
# print("Test perf IA10P vs IA100P on ", Ngames, " games")
# GN.ParralelPlayoutIANPvsNpP(Ngames,10,100)

# Ngames = 10
# print("\n"*2)
# print("Test perf IA100P vs IA1KP on ", Ngames, " games")
# GN.ParralelPlayoutIANPvsNpP(Ngames,100,1000)

# print("\n"*2)
# print("Test perf IA1KP vs IA10KP on ", Ngames, " games")
# GN.ParralelPlayoutIANPvsNpP(Ngames,1000,10000)

print("Test perf IAMCTS vs normal player on ", Ngames, " games")
for c in [0.1,0.2,0.4,0.8,1.0,1.2,1.4,1.6,1.8,2.0]:
    ParralelPlayoutIAMCTS(Ngames,nbSimus=Nsimus,c=c)