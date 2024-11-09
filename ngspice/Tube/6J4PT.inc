*
* Generic triode model: 6J4PT
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Mon Oct 05 16:34:25 2009
*             Plate
*             | Grid
*             | | Cathode
*             | | |
.SUBCKT 6J4PT A G K
.PARAM X1=0.35903154 X2=0.0051230024 X3=-0.3031279
.PARAM X4=0.83188775 X5=32.815182 X6=1.8031279
.PARAM X7=0.0040192654 X8=39.446647 X9=0.00247534
.PARAM Y1=0.0020096327 Y2=0.002163434
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 0.015p
CGK G K 11p
CAK A K 6p
.ENDS
