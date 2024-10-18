module mod_bcstw
  use mod_types
  implicit none
  private
#if defined (_STW)
  public bc_rhs_stw, corr_abc_stw
#endif
  contains
  !
#if defined (_STW)
  subroutine bc_rhs_stw(rhs,dlc,dlf,dl,lo)
    use mod_param       , only: time,amp,lambda,omega
    implicit none
    real(rp), intent(inout), dimension(:,:,0:) :: rhs
    real(rp), intent(in   ), dimension(0:1)    :: dlc,dlf
    real(rp), intent(in   ), dimension(3)      :: dl
    integer , intent(in   ), dimension(3)      :: lo
    !
    real(rp)   :: v_stw
    real(rp)   :: xgrid
    integer    :: nx,ny,i
    !
    ny = size(rhs,2)
    nx = size(rhs,1)
    do i=1,nx
      xgrid      = (i+lo(1)-1-.5)*dl(1)
      v_stw      = amp*sin(lambda*xgrid - omega*time)
      rhs(i,:,0) = -2.*v_stw/dlc(0)/dlf(0)
      rhs(i,:,1) = -2.*v_stw/dlc(1)/dlf(1)
    end do
  end subroutine bc_rhs_stw
#endif 
  !
  subroutine corr_abc_stw(a,b,c)
    implicit none
    real(rp), intent(in   ), dimension(:) :: a
    real(rp), intent(inout), dimension(:) :: b
    real(rp), intent(in   ), dimension(:) :: c
    integer :: n
    !
    n = size(b)
    !
    b(1) = b(1) - c(1)
    b(n) = b(n) - a(n)
  end subroutine corr_abc_stw    
end module mod_bcstw
