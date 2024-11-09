*
* Generic triode model: 814T
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Sun Nov 21 21:29:55 2010
*            Plate
*            | Grid
*            | | Cathode
*            | | |
.SUBCKT 814T A G K
.PARAM X1=0.99997793 X2=0.0050824018 X3=-0.057991999
.PARAM X4=0.96277773 X5=7.3237559 X6=1.557992
.PARAM X7=0.00052636102 X8=7.6069021 X9=0.0004358701
.PARAM Y1=0.00026318051 Y2=0.00037031315
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(A,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(A,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(A,K))/X5))^X6
BA A K  I=URAMP((Y2*URAMP(V(A,K))^1.5)-URAMP((Y2*URAMP(V(A,K))^1.5)-V(IK)+Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)))+1E-10*V(A,K)
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 4p
CGK G K 10p
CAK A K 13.5p
.ENDS
