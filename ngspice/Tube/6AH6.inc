*
* Generic pentode model: 6AH6
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Sat Aug 15 11:54:42 2020
*            Plate
*            | Screen Grid
*            | | Control Grid
*            | | | Cathode
*            | | | |
.SUBCKT 6AH6 A S G K
.PARAM X1=-0.10666057 X2=0.020062112 X3=-1.6597045
.PARAM X4=0.47472794 X5=26.182292 X6=3.1597045
.PARAM X7=0.00859215 X8=55.152204 X9=0.010163153
.PARAM Y1=0.004296075 Y2=0.84172336 Y3=0.15827664
.PARAM Y4=0.0045308158 EA=6375
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(S,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(S,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(S,K))/X5))^X6
BL IL 0 V=(V(IK)-Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4))*(1-.4*(EXP(-URAMP(V(A,K))/URAMP(V(S,K))*15)-EXP(-15)))
BM IM 0 V=V(IL)*(URAMP(V(A,K))+EA)/(URAMP(V(S,K))+EA)-URAMP(V(IL)*(URAMP(V(A,K))+EA)/(URAMP(V(S,K))+EA)-(Y4*(URAMP(V(A,K))+URAMP(URAMP(V(S,K))-URAMP(V(A,K))))^1.5))
BA A K I=URAMP(Y4*URAMP(V(A,K))^1.5-URAMP(Y4*URAMP(V(A,K))^1.5-V(IM)+V(IL)*(Y2*(1-URAMP(V(A,K))/(URAMP(V(A,K))+10))^1.5+Y3)))+1E-10*V(A,K)
BS S K I=URAMP(V(IM)-URAMP(Y4*URAMP(V(A,K))^1.5-URAMP(Y4*URAMP(V(A,K))^1.5-V(IM)+V(IL)*(Y2*(1-URAMP(V(A,K))/(URAMP(V(A,K))+10))^1.5+Y3))))
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 0.02p
CGK G K 6p
CGS G S 4p
CAK A K 3.6p
.ENDS