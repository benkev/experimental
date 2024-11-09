*
* Generic triode model: 5GH8(T)
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Tue Sep 22 16:15:32 2020
*               Plate
*               | Grid
*               | | Cathode
*               | | |
.SUBCKT 5GH8T A G K
.PARAM X1=0.99998933 X2=0.049419301 X3=-2.921024
.PARAM X4=0.33928791 X5=13.369515 X6=4.421024
.PARAM X7=0.0023961101 X8=39.404631 X9=0.016787239
.PARAM Y1=0.0011980551 Y2=0.0012898431
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 1.7p
CGK G K 3.4p
CAK A K 0.3p
.ENDS
