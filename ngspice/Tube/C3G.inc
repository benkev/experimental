*
* Generic pentode model: C3G
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved
* Version 3.10, Generated on Tue Jun 23 00:59:12 2009
*           Plate
*           | Screen Grid
*           | | Control Grid
*           | | | Cathode
*           | | | |
.SUBCKT C3G A S G K
.PARAM X1=-0.16053394 X2=0.0043084008 X3=-0.28973288
.PARAM X4=0.8381139 X5=37.574522 X6=1.7897329
.PARAM X7=0.0098836949 X8=44.832238 X9=0.0061379447
.PARAM Y1=0.0049418474 Y2=0.79240682 Y3=0.20759318
.PARAM Y4=0.005274374 EA=7975
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(S,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(S,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(S,K))/X5))^X6
BL IL 0 V=(V(IK)-Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4))*(1-.4*(EXP(-URAMP(V(A,K))/URAMP(V(S,K))*15)-EXP(-15)))
BM IM 0 V=V(IL)*(URAMP(V(A,K))+EA)/(URAMP(V(S,K))+EA)-URAMP(V(IL)*(URAMP(V(A,K))+EA)/(URAMP(V(S,K))+EA)-(Y4*(URAMP(V(A,K))+URAMP(URAMP(V(S,K))-URAMP(V(A,K))))^1.5))
BA A K I=URAMP(Y4*URAMP(V(A,K))^1.5-URAMP(Y4*URAMP(V(A,K))^1.5-V(IM)+V(IL)*(Y2*(1-URAMP(V(A,K))/(URAMP(V(A,K))+10))^1.5+Y3)))+1E-10*V(A,K)
BS S K I=URAMP(V(IM)-URAMP(Y4*URAMP(V(A,K))^1.5-URAMP(Y4*URAMP(V(A,K))^1.5-V(IM)+V(IL)*(Y2*(1-URAMP(V(A,K))/(URAMP(V(A,K))+10))^1.5+Y3))))
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 0.012p
CGK G K 4.9p
CGS G S 3.3p
CAK A K 3.5p
.ENDS
