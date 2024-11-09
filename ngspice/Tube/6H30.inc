*
* Generic triode model: 6H30
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Tue Aug 10 23:01:40 2010
*            Plate
*            | Grid
*            | | Cathode
*            | | |
.SUBCKT 6H30 A G K
.PARAM X1=1 X2=0.0016665882 X3=-0.038461562
.PARAM X4=0.97499998 X5=15.000715 X6=1.5384616
.PARAM X7=0.0038313265 X8=15.385349 X9=0.003327778
.PARAM Y1=0.0019156633 Y2=0.0022952053
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 5p
CGK G K 6.3p
CAK A K 2.4p
.ENDS
