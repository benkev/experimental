*
* Generic triode model: D3A
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Tue Dec 24 08:47:57 2013
*                 Plate
*                 | Grid
*                 | | Cathode
*                 | | |
.SUBCKT D3A A G K
.PARAM X1=0.38087015 X2=0.0021582473 X3=-0.28773641
.PARAM X4=0.83904987 X5=74.574462 X6=1.7877364
.PARAM X7=0.03039942 X8=88.879654 X9=0.018902992
.PARAM Y1=0.01519971 Y2=0.015714194
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 2.7p
CGK G K 7.3p
CAK A K 3.1p
.ENDS
