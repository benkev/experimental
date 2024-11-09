*
* Generic triode model: 100TH
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Fri Nov 26 13:06:10 2010
*             Plate
*             | Grid
*             | | Cathode
*             | | |
.SUBCKT 100TH A G K
.PARAM X1=-0.99906964 X2=0.0014094818 X3=-0.083738904
.PARAM X4=0.94712581 X5=37.513209 X6=1.5837389
.PARAM X7=0.00034469629 X8=39.607419 X9=0.00027070609
.PARAM Y1=0.00017234814 Y2=0.00018548443
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^6
BA A K I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 2p
CGK G K 2.9p
CAK A K 0.3p
.ENDS
