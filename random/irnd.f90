program irnd
  
  use mtmod        ! random number generator based on MT method

  implicit none

  interface
     function gauss(mu, sigma)
       real(8)             :: gauss
       real(8), intent(in) :: mu
       real(8), intent(in) :: sigma
     end function gauss
     function int_uniform(n)
       integer             :: int_uniform
       integer, intent(in) :: n
     end function int_uniform
  end interface


  integer :: i
  real(8) :: r
  do i = 1, 1000
     !write(*,*) int_uniform(66)
     r = gauss(5d0, 3d0)
     write(*,*) r, nint(r), int(r)
  end do

end program irnd
!contains

!================================================================

! Function for creating random gaussian with Box-Muller transform
! input  mu: mean of Gaussian, sigma : deviation of Gaussian
! output, x: random number which follows gaussian probability dist.
function gauss(mu, sigma)

  use mtmod        ! random number generator based on MT method

  implicit none
  real(8), parameter :: pi = atan(1.0d0)*4.0d0
  real(8)             :: gauss
  real(8), intent(in) :: mu
  real(8), intent(in) :: sigma
  !real(8) function  grnd()
  !real(8) function  gauss()

  real(8) :: x0, y0

!!$  interface
!!$     function grnd()
!!$       real(8) :: grnd
!!$     end function grnd
!!$  end interface

  x0 = grnd()
  y0 = grnd()
  gauss  = sqrt(-2.0d0*log(x0))*sin(2.0d0*pi*y0)
  gauss  = gauss*sigma + mu
  
end function gauss


!=======================================================================

! Generate an integer random number distributed uniformly in [1..n]
function int_uniform(range)

  use mtmod        ! random number generator based on MT method

  implicit none
  integer             :: int_uniform
  integer, intent(in) :: range
  integer :: k

  do     ! Loop until k is less than n
     k = int(range*grnd()) + 1
     if (k <= n) then
        exit
     end if
  end do
  int_uniform = k
end function int_uniform
