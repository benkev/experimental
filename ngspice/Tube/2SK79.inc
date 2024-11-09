*
** Generic triode model: 2SK79
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Sun Feb 28 23:48:12 2010
*               Plate
*               | Grid
*               | | Cathode
*               | | |
.SUBCKT 2SK79 A G K
.PARAM X1=0.10157886 X2=0.005502006 X3=-0.28096081
.PARAM X4=0.84224201 X5=28.672814 X6=1.7809608
.PARAM X7=0.019995035 X8=34.043439 X9=0.012489311
.PARAM Y1=0.0099975173 Y2=0.010884964
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 15p
CGK G K 16p
CAK A K 4p
.ENDS
