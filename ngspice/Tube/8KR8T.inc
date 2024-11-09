*
* Generic triode model: 8KR8(T)
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Sat Aug 15 16:15:03 2020
*               Plate
*               | Grid
*               | | Cathode
*               | | |
.SUBCKT 8KR8T A G K
.PARAM X1=0.59067387 X2=0.032825021 X3=-2.69289
.PARAM X4=0.35774847 X5=19.565914 X6=4.19289
.PARAM X7=0.0065857775 X8=54.691817 X9=0.031837877
.PARAM Y1=0.0032928888 Y2=0.0034743361
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 0.075p
CGK G K 13p
CAK A K 4.4p
.ENDS
