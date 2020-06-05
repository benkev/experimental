program test
implicit none

integer(1) byar(16)
integer :: i, j, k
integer :: p(10) =   (/0,0,0,0,0,0,0,0,0,0/), iv(5) = (/2,3,6,8,9/)
integer :: inv(10) = (/0,1,2,0,0,3,0,4,5,0/), q(5) =  (/0,0,0,0,0/)

!do i = 1, 16; byar(i) = i; end do
byar = (/ (4*i, i = 1, 16) /)
write(*,*) byar
write(*,*) 

p(iv) = 1
write(*,*) p
!q(inv) = q ! Error: Different shape for array assignment at (1) 
            ! on dimension 1 (10 and 5)
q(inv) = 99

write(*,*) q

do i = 1, 10;    write(*,*) i, i**2, i**3, i**10; enddo ! i

end program test 
