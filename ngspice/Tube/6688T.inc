*
* Generic triode model: 6688T
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Wed Jun 24 22:55:42 2009
*             Plate
*             | Grid
*             | | Cathode
*             | | |
.SUBCKT 6688T A G K
.PARAM X1=-0.59576533 X2=0.0094827837 X3=-0.8531559
.PARAM X4=0.63744183 X5=38.233306 X6=2.3531559
.PARAM X7=0.01902963 X8=59.979286 X9=0.011758543
.PARAM Y1=0.0095148149 Y2=0.009992698
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 3.43p
CGK G K 5.1p
CAK A K 3.5p
.ENDS
