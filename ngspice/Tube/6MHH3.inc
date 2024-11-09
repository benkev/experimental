*
* Generic triode model: 6MHH3
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Thu Aug 13 09:45:57 2020
*             Plate
*             | Grid
*             | | Cathode
*             | | |
.SUBCKT 6MHH3 A G K
.PARAM X1=-0.08153105 X2=0.019587925 X3=-1.1261845
.PARAM X4=0.57117084 X5=21.892526 X6=2.6261845
.PARAM X7=0.0043762117 X8=38.329208 X9=0.003168903
.PARAM Y1=0.0021881059 Y2=0.0023604796
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 1.3p
CGK G K 2.4p
CAK A K 0.4p
.ENDS
