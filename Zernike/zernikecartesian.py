def zernikecartesian(coefficient,x,y):
    """
    ------------------------------------------------
    zernikecartesian(coefficient,x,y):
 
    Return combined aberration
 
    Zernike Polynomials Caculation in Cartesian coordinates
 
    coefficient: Zernike Polynomials Coefficient from input
    x: x in Cartesian coordinates
    y: y in Cartesian coordinates
    ------------------------------------------------
    """
   
    Z = np.zeros(37)

    if len(coefficient) < len(Z):
        c = Z.copy()
        c[:len(coefficient)] += coefficient
    else:
        c = Z.copy()
        c[:len(b)] += Z

    r = np.sqrt(x**2 + y**2)
    Z1  =  c[0]  * 1
    Z2  =  c[1]  * 2*x
    Z3  =  c[2]  * 2*y
    Z4  =  c[3]  * np.sqrt(3)*(2*r**2-1)
    Z5  =  c[4]  * 2*np.sqrt(6)*x*y
    Z6  =  c[5]  * np.sqrt(6)*(x**2-y**2)
    Z7  =  c[6]  * np.sqrt(8)*y*(3*r**2-2)
    Z8  =  c[7]  * np.sqrt(8)*x*(3*r**2-2)
    Z9  =  c[8]  * np.sqrt(8)*y*(3*x**2-y**2)
    Z10 =  c[9] * np.sqrt(8)*x*(x**2-3*y**2)
    Z11 =  c[10] * np.sqrt(5)*(6*r**4-6*r**2+1)
    Z12 =  c[11] * np.sqrt(10)*(x**2-y**2)*(4*r**2-3)
    Z13 =  c[12] * 2*np.sqrt(10)*x*y*(4*r**2-3)
    Z14 =  c[13] * np.sqrt(10)*(r**4-8*x**2*y**2)
    Z15 =  c[14] * 4*np.sqrt(10)*x*y*(x**2-y**2)
    Z16 =  c[15] * np.sqrt(12)*x*(10*r**4-12*r**2+3)
    Z17 =  c[16] * np.sqrt(12)*y*(10*r**4-12*r**2+3)
    Z18 =  c[17] * np.sqrt(12)*x*(x**2-3*y**2)*(5*r**2-4)
    Z19 =  c[18] * np.sqrt(12)*y*(3*x**2-y**2)*(5*r**2-4)
    Z20 =  c[19] * np.sqrt(12)*x*(16*x**4-20*x**2*r**2+5*r**4)
    Z21 =  c[20] * np.sqrt(12)*y*(16*y**4-20*y**2*r**2+5*r**4)
    Z22 =  c[21] * np.sqrt(7)*(20*r**6-30*r**4+12*r**2-1)
    Z23 =  c[22] * 2*np.sqrt(14)*x*y*(15*r**4-20*r**2+6)
    Z24 =  c[23] * np.sqrt(14)*(x**2-y**2)*(15*r**4-20*r**2+6)
    Z25 =  c[24] * 4*np.sqrt(14)*x*y*(x**2-y**2)*(6*r**2-5)
    Z26 =  c[25] * np.sqrt(14)*(8*x**4-8*x**2*r**2+r**4)*(6*r**2-5)
    Z27 =  c[26] * np.sqrt(14)*x*y*(32*x**4-32*x**2*r**2+6*r**4)
    Z28 =  c[27] * np.sqrt(14)*(32*x**6-48*x**4*r**2+18*x**2*r**4-r**6)
    Z29 =  c[28] * 4*y*(35*r**6-60*r**4+30*r**2-4)
    Z30 =  c[29] * 4*x*(35*r**6-60*r**4+30*r**2-4)
    Z31 =  c[30] * 4*y*(3*x**2-y**2)*(21*r**4-30*r**2+10)
    Z32 =  c[31] * 4*x*(x**2-3*y**2)*(21*r**4-30*r**2+10)
    Z33 =  c[32] * 4*(7*r**2-6)*(4*x**2*y*(x**2-y**2)+y*(r**4-8*x**2*y**2))
    Z34 =  c[33] * (4*(7*r**2-6)*(x*(r**4-8*x**2*y**2)-4*x*y**2*(x**2-y**2)))
    Z35 =  c[34] * (8*x**2*y*(3*r**4-16*x**2*y**2)+4*y*(x**2-y**2)*(r**4-16*x**2*y**2))
    Z36 =  c[35] * (4*x*(x**2-y**2)*(r**4-16*x**2*y**2)-8*x*y**2*(3*r**4-16*x**2*y**2))
    Z37 =  c[36] * 3*(70*r**8-140*r**6+90*r**4-20*r**2+1)
    ZW =    Z1 + Z2 +  Z3+  Z4+  Z5+  Z6+  Z7+  Z8+  Z9+ \
            Z10+ Z11+ Z12+ Z13+ Z14+ Z15+ Z16+ Z17+ Z18+ Z19+ \
            Z20+ Z21+ Z22+ Z23+ Z24+ Z25+ Z26+ Z27+ Z28+ Z29+ \
            Z30+ Z31+ Z32+ Z33+ Z34+ Z35+ Z36+ Z37
    return ZW 
