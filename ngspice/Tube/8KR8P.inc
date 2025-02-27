*
* Generic pentode model: 8KR8(P)
* Copyright 2003--2008 by Ayumi Nakabayashi, All rights reserved.
* Version 3.10, Generated on Sat Aug 15 17:44:30 2020
*               Plate
*               | Screen Grid
*               | | Control Grid
*               | | | Cathode
*               | | | |
.SUBCKT 8KR8P A S G K
.PARAM X1=0.58060879 X2=0.0021175926 X3=-0.12870649
.PARAM X4=0.92097625 X5=37.317729 X6=1.6287065
.PARAM X7=0.017809062 X8=40.519752 X9=0.012983051
.PARAM Y1=0.0089045311 Y2=0.87576705 Y3=0.12423295
.PARAM Y4=0.0095678555 EA=1655
BK IK 0 V=U(V(G,K)+X1)*X7*URAMP(V(G,K)+X1+URAMP(V(S,K))/X8)^1.5+(1-U(V(G,K)+X1))*X9*(X2*URAMP(V(S,K)))^X3*(X4*URAMP(V(G,K)+X1+URAMP(V(S,K))/X5))^X6
BL IL 0 V=(V(IK)-Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4))*(1-.4*(EXP(-URAMP(V(A,K))/URAMP(V(S,K))*15)-EXP(-15)))
BM IM 0 V=V(IL)*(URAMP(V(A,K))+EA)/(URAMP(V(S,K))+EA)-URAMP(V(IL)*(URAMP(V(A,K))+EA)/(URAMP(V(S,K))+EA)-(Y4*(URAMP(V(A,K))+URAMP(URAMP(V(S,K))-URAMP(V(A,K))))^1.5))
BA A K I=URAMP(Y4*URAMP(V(A,K))^1.5-URAMP(Y4*URAMP(V(A,K))^1.5-V(IM)+V(IL)*(Y2*(1-URAMP(V(A,K))/(URAMP(V(A,K))+10))^1.5+Y3)))+1E-10*V(A,K)
BS S K I=URAMP(V(IM)-URAMP(Y4*URAMP(V(A,K))^1.5-URAMP(Y4*URAMP(V(A,K))^1.5-V(IM)+V(IL)*(Y2*(1-URAMP(V(A,K))/(URAMP(V(A,K))+10))^1.5+Y3))))
BG G K I=Y1*URAMP(V(G,K))^1.5*(URAMP(V(G,K))/(URAMP(V(A,K))+URAMP(V(G,K)))*1.2+.4)
* CAPS
CGA G A 2.6p
CGK G K 0.9p
CGS G S 0.6p
CAK A K 3.2p
.ENDS
