!
! Given arrays x and y of length N containing a tabulated function, i.e., 
! yi = f(xi), with x1 < x2 < ... < xN, and given values yp1 and ypn for 
! the first derivative of the interpolating function at points 1 and N, 
! respectively, this routine returns an array y2 of length N that contains 
! the second derivatives of the interpolating function at the tabulated 
! points xi. If yp1 and/or ypn are equal to 1 × 1030 or larger, the routine 
! is signaled to set the corresponding boundary condition for a natural 
! spline, with zero second derivative on that boundary.
!
SUBROUTINE spline(x,y,yp1,ypn,y2)
  USE nrtype; USE nrutil, ONLY : assert_eq
  USE nr, ONLY : tridag
  IMPLICIT NONE
  REAL(SP), DIMENSION(:), INTENT(IN) :: x,y
  REAL(SP), INTENT(IN) :: yp1,ypn
  REAL(SP), DIMENSION(:), INTENT(OUT) :: y2
  INTEGER(I4B) :: n
  REAL(SP), DIMENSION(size(x)) :: a,b,c,r
  n=assert_eq(size(x),size(y),size(y2),’spline’)

  ! Set up the tridiagonal equations.
  c(1:n-1)=x(2:n)-x(1:n-1)
  r(1:n-1)=6.0_sp*((y(2:n)-y(1:n-1))/c(1:n-1))
  r(2:n-1)=r(2:n-1)-r(1:n-2)
  a(2:n-1)=c(1:n-2)
  b(2:n-1)=2.0_sp*(c(2:n-1)+a(2:n-1))
  b(1)=1.0
  b(n)=1.0
  !The lower boundary condition is set either to be “natr(1)=0.0 ural”
  if (yp1 > 0.99e30_sp) then 
     c(1)=0.0
  else or else to have a specified first derivative.
     r(1)=(3.0_sp/(x(2)-x(1)))*((y(2)-y(1))/(x(2)-x(1))-yp1)
     c(1)=0.5
  end if !The upper boundary condition is set either to be
  if (ypn > 0.99e30_sp) then “natural”
     r(n)=0.0
     a(n)=0.0
  else or else to have a specified first derivative.
     r(n)=(-3.0_sp/(x(n)-x(n-1)))*((y(n)-y(n-1))/(x(n)-x(n-1))-ypn)
     a(n)=0.5
  end if
  call tridag(a(2:n),b(1:n),c(1:n-1),r(1:n),y2(1:n))
END SUBROUTINE spline












FUNCTION splint(xa,ya,y2a,x)
  USE nrtype; USE nrutil, ONLY : assert_eq,nrerror
  USE nr, ONLY: locate
  IMPLICIT NONE
  REAL(SP), DIMENSION(:), INTENT(IN) :: xa,ya,y2a
  REAL(SP), INTENT(IN) :: x
  REAL(SP) :: splint
  ! Given the arrays xa and ya, which tabulate a function (with the xai’s 
  ! in increasing or
  decreasing order), and given the array y2a, which is the output from spline above, and
  given a value of x, this routine returns a cubic-spline interpolated value. The arrays xa, ya
  and y2a are all of the same size.
  INTEGER(I4B) :: khi,klo,n
  REAL(SP) :: a,b,h
  n=assert_eq(size(xa),size(ya),size(y2a),’splint’)
  klo=max(min(locate(xa,x),n-1),1)
  We will find the right place in the table by means of locate’s bisection algorithm. This is
  optimal if sequential calls to this routine are at random values of x. If sequential calls are in
  order, and closely spaced, one would do better to store previous values of klo and khi and
  test if they remain appropriate on the next call.
  khi=klo+1 klo and khi now bracket the input value of x.
  h=xa(khi)-xa(klo)
  if (h == 0.0) call nrerror(’bad xa input in splint’) The xa’s must be distinct.
  a=(xa(khi)-x)/h Cubic spline polynomial is now evaluated.
  b=(x-xa(klo))/h
  splint=a*ya(klo)+b*ya(khi)+((a**3-a)*y2a(klo)+(b**3-b)*y2a(khi))*(h**2)/6.0_sp
END FUNCTION splint





SUBROUTINE splie2(x1a,x2a,ya,y2a)
  USE nrtype; USE nrutil, ONLY : assert_eq
  USE nr, ONLY : spline
  IMPLICIT NONE
  REAL(SP), DIMENSION(:), INTENT(IN) :: x1a,x2a
  REAL(SP), DIMENSION(:,:), INTENT(IN) :: ya
  REAL(SP), DIMENSION(:,:), INTENT(OUT) :: y2a
  ! Given an M × N tabulated function ya, and N tabulated independent 
  ! variables x2a, this routine constructs one-dimensional natural cubic 
  ! splines of the rows of ya and returns the second derivatives in 
  ! the M × N array y2a. (The array x1a is included in the argument
  ! list merely for consistency with routine splin2.)
    INTEGER(I4B) :: j,m,ndum
    m=assert_eq(size(x1a),size(ya,1),size(y2a,1),’splie2: m’)
    ndum=assert_eq(size(x2a),size(ya,2),size(y2a,2),’splie2: ndum’)
    do j=1,m
       call spline(x2a,ya(j,:),1.0e30_sp,1.0e30_sp,y2a(j,:))
    end do
  END SUBROUTINE splie2





  FUNCTION splin2(x1a,x2a,ya,y2a,x1,x2)
    USE nrtype; USE nrutil, ONLY : assert_eq
    USE nr, ONLY : spline,splint
    IMPLICIT NONE
    REAL(SP), DIMENSION(:), INTENT(IN) :: x1a,x2a
    REAL(SP), DIMENSION(:,:), INTENT(IN) :: ya,y2a
    REAL(SP), INTENT(IN) :: x1,x2
    REAL(SP) :: splin2
    Given x1a, x2a, ya as described in splie2 and y2a as produced by that routine; and given
    a desired interpolating point x1,x2; this routine returns an interpolated function value by
      bicubic spline interpolation.
      INTEGER(I4B) :: j,m,ndum
      REAL(SP), DIMENSION(size(x1a)) :: yytmp,y2tmp2
      m=assert_eq(size(x1a),size(ya,1),size(y2a,1),’splin2: m’)
      ndum=assert_eq(size(x2a),size(ya,2),size(y2a,2),’splin2: ndum’)
      do j=1,m
         yytmp(j)=splint(x2a,ya(j,:),y2a(j,:),x2)
         Performm evaluations of the row splines constructed by splie2, using the one-dimensional
         spline evaluator splint.
      end do
      call spline(x1a,yytmp,1.0e30_sp,1.0e30_sp,y2tmp2)
      Construct the one-dimensional column spline and evaluate it.
      splin2=splint(x1a,yytmp,y2tmp2,x1)
    END FUNCTION value

















