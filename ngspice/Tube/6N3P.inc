*
* Generic triode model: 6N3P
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Thu Jun 11 18:27:43 2009
*            Plate
*            | Grid
*            | | Cathode
*            | | |
.SUBCKT 6N3P A G K
.PARAM X1=-0.069189567 X2=0.019068965 X3=-1.2621392
.PARAM X4=0.54305736 X5=23.962635 X6=2.7621392
.PARAM X7=0.0034531579 X8=44.125422 X9=0.0027770019
.PARAM Y1=0.001726579 Y2=0.0018446282
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 1.6p
CGK G K 2.8p
CAK A K 1.4p
.ENDS
