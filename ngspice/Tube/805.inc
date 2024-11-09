*
* Generic triode model: 805
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Sat Aug 01 11:40:19 2009
*           Plate
*           | Grid
*           | | Cathode
*           | | |
.SUBCKT 805 A G K
.PARAM X1=-1 X2=0.19254103 X3=-16.929908
.PARAM X4=0.081389445 X5=4.7709859 X6=18.429908
.PARAM X7=0.00078806053 X8=58.619221 X9=5.2042766e+14
.PARAM Y1=0.00039403026 Y2=0.00041428161
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 8p
CGK G K 6.5p
CAK A K 5p
.ENDS
