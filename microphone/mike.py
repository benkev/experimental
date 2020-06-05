from pylab import *

R = array([1., 2., 3., 5., 6.7, 10., 100.])
Vr = array([0.656, 1.268, 1.813, 2.766, 3.435, 4.25, 11.82])

#Vm = array([11.37, 10.76, 10.22, 9.22, 8.6, 7.77, 0.206])
Vm = 12.02 - Vr

#I = array([0.656, 0.634, 0.6043, 0.553, 0.5127, 0.425, 0.118])
I = Vr/R

figure()
plot(Vm, I, 'b', lw=2); grid(1)
plot(Vm, I, 'ro')

ylim(-0.01, 1.45)
xlim(-0.05, 12.05)
title('Electret Microphone I vs V')
xlabel('Vm  (V)')
ylabel('I  (mA)')

text(2., 0.8, r'Dynamic resistance r $\approx$ 21k')

show()





