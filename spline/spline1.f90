program tsp
  !
  ! Test the spline functions from Numerical Recipes in FORTRAN
  !
  implicit none 
  integer, parameter :: N = 4
  integer, parameter :: M = 20
  integer :: i, j
  real,    parameter :: pi = 4.*atan(1.)
  real :: x(N), y(N), p2z(N,N), zi(M,M), d, xi, yi
  real :: z(N,N) = reshape([ 0.,   0.,   0.,   0.,     &
                             0.2,  0.5,  0.5,  0.2,    &
                            -0.2,  0.,  -0.1,  0.5,    &
                             0.,   0.1, -0.5,  0.1], [4,4])

  do i = 1, N
     x(i) = i - 2   ! x(1..4) = -1, 0, 1, 2
     y(i) = i - 2   ! y(1..4) = -1, 0, 1, 2
  end do

  call splie2(x, y, z, N, N, p2z) ! --> p2z(N,N) - 2hn derivatives

  d = 3./(M - 1)
  do i = 1, M
     do j = 1, M
        xi = -1. + (i-1)*d
        yi = -1. + (j-1)*d
        !    splin2(x1a, x2a, ya, y2a, m, n, x1, x2, y)  
        call splin2(x, y, z, p2z, N, N, xi, yi, zi(i,j))  
     end do
  end do
  
  do i = 1, N
     do j = 1, N
        write(*,*) z(i,j)
     end do
  end do

  write(*,*) '-----------------------------------------'

  do i = 1, M
     do j = 1, M
        write(*,*) zi(i,j)
     end do
  end do


end program tsp

!!$program tsp
!!$  !
!!$  ! Test the spline functions from Numerical Recipes in FORTRAN
!!$  !
!!$  implicit none 
!!$  integer, parameter :: N = 11
!!$  integer, parameter :: M = 101
!!$  integer :: i, j
!!$  real,    parameter :: pi = 4.*atan(1.)
!!$  real :: x(N), y(N), p2y(N), xi(M), yi(M)
!!$  real :: dx, dxi
!!$
!!$  dx = 2.*pi/(N - 1)
!!$  do i = 1, N
!!$     x(i) = i*dx
!!$     y(i) = 1./x(i)   !sin(x(i))
!!$  end do
!!$
!!$  call spline(x, y, N, 1e30, 1e30, p2y)
!!$
!!$  dxi = 2.*pi/(M - 1)
!!$  do i = 1, M
!!$     xi(i) = i*dxi
!!$     call splint(x, y, p2y, N, xi(i), yi(i))
!!$  end do
!!$  
!!$  do i = 1, N
!!$     write(*,*) x(i), y(i)
!!$  end do
!!$  write(*,*) '-----------------------------------------'
!!$  do i = 1, M
!!$     write(*,*) xi(i), yi(i)
!!$  end do
!!$
!!$end program tsp


!
! Given arrays x and y of length N containing a tabulated function, i.e., 
! yi = f(xi), with x1 < x2 < ... < xN, and given values yp1 and ypn for 
! the first derivative of the interpolating function at points 1 and N, 
! respectively, this routine returns an array y2 of length N that contains 
! the second derivatives of the interpolating function at the tabulated 
! points xi. If yp1 and/or ypn are equal to 1.0e30 or larger, the routine 
! is signaled to set the corresponding boundary condition for a natural 
! spline, with zero second derivative on that boundary.
!
subroutine spline(x, y, n, yp1, ypn, y2)  
  integer n
  real yp1,ypn,x(n),y(n),y2(n)  
  integer, parameter :: NMAX=500  
  integer i,k  
  real p,qn,sig,un,u(NMAX)  
  if (yp1 > 0.99e30) then  
     y2(1)=0.  
     u(1)=0.  
  else  
     y2(1)=-0.5  
     u(1)=(3./(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)  
  endif
  do i=2,n-1  
     sig=(x(i)-x(i-1))/(x(i+1)-x(i-1))  
     p=sig*y2(i-1)+2.  
     y2(i)=(sig-1.)/p  
     u(i)=(6.*((y(i+1)-y(i))/(x(i+1)-x(i))- &
          (y(i)-y(i-1))/(x(i)-x(i-1)))/(x(i+1)-x(i-1))-sig*u(i-1))/p  
  end do
  if (ypn > 0.99e30) then  
     qn=0.  
     un=0.  
  else  
     qn=0.5  
     un=(3./(x(n)-x(n-1)))*(ypn-(y(n)-y(n-1))/(x(n)-x(n-1)))  
  endif
  y2(n)=(un-qn*u(n-1))/(qn*y2(n-1)+1.)  
  do k=n-1,1,-1  
     y2(k)=y2(k)*y2(k+1)+u(k)  
  end do
  return  
end subroutine spline



!
! Given the arrays xa and ya, which tabulate a function (with the xai’s 
! in increasing or decreasing order), and given the array y2a, which is 
! the output from spline above, and given a value of x, this routine 
! returns a cubic-spline interpolated value. The arrays xa, ya and y2a 
! are all of the same size.
!
subroutine splint(xa, ya, y2a, n, x, y)  
  integer n  
  real x,y,xa(n),y2a(n),ya(n)  
  integer k,khi,klo  
  real a,b,h  
  klo=1  
  khi=n  
  do while(khi-klo > 1)
     k=(khi+klo)/2  
     if(xa(k) > x)then  
        khi=k  
     else  
        klo=k  
     endif
  end do
  h=xa(khi)-xa(klo)  
  if (h == 0.) then
     write(*,*) 'SPLINT: bad xa input in splint'
     stop
  end if
  a=(xa(khi)-x)/h  
  b=(x-xa(klo))/h  
  y=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**2)/6.  
end subroutine splint



!
! Given an M × N tabulated function ya, and N tabulated independent 
! variables x2a, this routine constructs one-dimensional natural cubic 
! splines of the rows of ya and returns the second derivatives in 
! the M × N array y2a. (The array x1a is included in the argument
! list merely for consistency with routine splin2.)
!
subroutine splie2(x1a, x2a, ya, m, n, y2a)  
  integer m,n
  real x1a(m),x2a(n),y2a(m,n),ya(m,n)  
  integer, parameter :: NN=100;
  integer j,k  
  real y2tmp(NN),ytmp(NN)  
  do j=1,m  
     do k=1,n  
        ytmp(k)=ya(j,k)  
     end do
     call spline(x2a,ytmp,n,1.e30,1.e30,y2tmp)  
     do k=1,n  
        y2a(j,k)=y2tmp(k)  
     end do
  end do
end subroutine splie2



!
! Given x1a, x2a, ya as described in splie2 and y2a as produced by that 
! routine; and given a desired interpolating point x1,x2; this routine 
! returns an interpolated function value by bicubic spline interpolation.
!
subroutine splin2(x1a, x2a, ya, y2a, m, n, x1, x2, y)  
  integer m,n
  real x1,x2,y,x1a(m),x2a(n),y2a(m,n),ya(m,n)  
  integer, parameter :: NN=100;
  integer j,k  
  real y2tmp(NN),ytmp(NN),yytmp(NN)  
  do j=1,m  
     do k=1,n  
        ytmp(k)=ya(j,k)  
        y2tmp(k)=y2a(j,k)  
     end do
     call splint(x2a,ytmp,y2tmp,n,x2,yytmp(j))  
  end do
  call spline(x1a,yytmp,m,1.e30,1.e30,y2tmp)  
  call splint(x1a,yytmp,y2tmp,m,x1,y)  
end subroutine splin2





