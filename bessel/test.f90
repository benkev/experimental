program test
implicit none

integer :: i
real(8) :: x

do i = 1, 21
   x = i - 1
   write(*,*) bessel_j0(x), bessel_j1(x), bessel_jn(2,x), bessel_jn(3,x)
enddo ! i

end program test 
