*
* Generic triode model: C3GT
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved
* Version 3.10, Generated on Tue Jun 23 00:43:12 2009
*            Plate
*            | Grid
*            | | Cathode
*            | | |
.SUBCKT C3GT A G K
.PARAM X1=-0.16053394 X2=0.0043084008 X3=-0.28973288
.PARAM X4=0.8381139 X5=37.574522 X6=1.7897329
.PARAM X7=0.0098836949 X8=44.832238 X9=0.0061379447
.PARAM Y1=0.0049418474 Y2=0.005274374
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 3.3p
CGK G K 4.9p
CAK A K 3.5p
.ENDS
