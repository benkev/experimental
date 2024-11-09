*
* Generic triode model: 4RHH2
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Tue Sep 22 14:43:38 2020
*              Plate
*              | Grid
*              | | Cathode
*              | | |
.SUBCKT 4RHH2 A G K
.PARAM X1=0.011065975 X2=0.075583733 X3=-4.3672066
.PARAM X4=0.25565829 X5=9.8479088 X6=5.8672066
.PARAM X7=0.0061327353 X8=38.51981 X9=0.65242302
.PARAM Y1=0.0030663676 Y2=0.0033067258
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 1.2p
CGK G K 5.6p
CAK A K 2.4p
.ENDS
