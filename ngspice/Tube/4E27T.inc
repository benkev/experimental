*
* Generic triode model: 4E27T
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Thu Jun 04 17:35:18 2009
*               Plate
*               | Grid
*               | | Cathode
*               | | |
.SUBCKT 4E27T A G K
.PARAM X1=0.99999999 X2=0.038865244 X3=-0.2993371
.PARAM X4=0.83364034 X5=4.2804222 X6=1.7993371
.PARAM X7=0.00048970507 X8=5.134615 X9=0.00030228693
.PARAM Y1=0.00024485253 Y2=0.00039466689
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 5p
CGK G K 8p
CAK A K 6.5p
.ENDS
