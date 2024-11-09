*
* Generic triode model: 6N6
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Sun Nov 22 23:27:49 2009
*           Plate
*           | Grid
*           | | Cathode
*           | | |
.SUBCKT 6N6 A G K
.PARAM X1=0.28728514 X2=0.023818988 X3=-0.69616265
.PARAM X4=0.68300952 X5=13.30831 X6=2.1961627
.PARAM X7=0.0027779074 X8=19.484809 X9=0.0016279111
.PARAM Y1=0.0013889537 Y2=0.0016055263
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 3.5p
CGK G K 4.4p
CAK A K 1.7p
.ENDS
