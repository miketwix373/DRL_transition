module mod_drl
  use mod_types
  use mod_common_mpi, only: myid,ierr,intracomm
  use mpi
  implicit none
  public
  real(rp) :: omega, n_act, offset
  integer  :: bnx, bny
  integer , dimension(2) :: bn_arr,n_arr
  real(rp) :: drl_dt, drl_dtn, drl_dpdx
  logical  :: drl_flag
  integer  :: drl_ind_mean, drl_inc
  real(rp) :: e_ks, e_ks_1, e_kw_1, e_kw_2, e_kw, e_ks_rew, e_drl
  character(len=5) :: req
  logical :: send_drl, start_drl
  !real(rp), dimension(48) :: u_obs
  !real(rp), dimension(4,12) :: u_obs
  !real(rp), dimension(0:1535) :: u_obs_all
  real(rp), allocatable   :: u_obs(:,:)
  real(rp), allocatable   :: u_obs_us(:,:)
  real(rp), allocatable   :: u_obs_send(:,:)
  !MPI stuff
  integer :: custom_type, custom_type_old, typesize
  integer, allocatable :: recvcounts(:), displs(:)
  integer(kind=mpi_address_kind) :: lb, extent
  contains
  !
  subroutine get_obs_drl(p,ind,inc,obs_num)
    implicit none
    real(rp), intent(in   ), dimension(0:,0:,0:) :: p
    integer , intent(in   )  :: ind,inc
    real(rp), intent(in   )  :: obs_num
    !
    integer                  :: i,j,nx,ny
    character(len=10)        :: obs_id
    character(len=10)        :: obs_id_num
    !
    nx = size(p,1)-2
    ny = size(p,2)-2
    !
    write(obs_id    ,'(i0)') myid
    write(obs_id_num,'(i0)') nint(obs_num)
    !
    open(1,file='obs/obs-'//trim(obs_id)//'_'//trim(obs_id_num)//'.dat')
    do i=1,nx,inc
      do j=1,ny,inc
       write(1,'(f7.5)') p(i,j,ind)
      end do
    end do
    close(1)
  end subroutine get_obs_drl

  subroutine get_obs_mpi1d(p,ind,inc,dims,p_obs)
    implicit none
    real(rp), intent(in   ), dimension(0:,0:,0:) :: p
    integer , intent(in   )  :: ind,inc,dims
    real(rp), intent(  out), dimension(0:)       :: p_obs
    !
    integer                  :: i,j,nx,ny,k
    !
    nx = size(p,1)-2
    ny = size(p,2)-2
    !
    k=0
    do i=1,nx,inc
      do j=1,ny,inc
        p_obs(k) = p(i,j,ind)
        k = k+1
      end do
    end do
    close(1)
  end subroutine get_obs_mpi1d

  subroutine get_obs_mpi(p,ind,inc,p_obs)
    implicit none
    real(rp), intent(in   ), dimension(0:,0:,0:) :: p
    integer , intent(in   )  :: ind,inc
    real(rp), intent(  out), dimension(0:,0:)    :: p_obs
    !
    !call undersample(p,ind,inc,p_obs)
 
  end subroutine get_obs_mpi


  subroutine get_obs_drl_vort(pv,pw,dl,dzf,ind,inc,obs_num)
    use mod_utils, only: vort_x
    implicit none
    real(rp), intent(in   ), dimension(0:,0:,0:) :: pv,pw
    real(rp), intent(in   ), dimension(3 )       :: dl
    real(rp), intent(in   ), dimension(0:)       :: dzf
    integer , intent(in   )  :: ind,inc
    real(rp), intent(in   )  :: obs_num
    !
    integer                  :: i,j,nx,ny
    real(rp)                 :: vort
    character(len=10)        :: obs_id
    character(len=10)        :: obs_id_num
    !
    nx = size(pv,1)-2
    ny = size(pv,2)-2
    !
    write(obs_id    ,'(i0)') myid
    write(obs_id_num,'(i0)') nint(obs_num)
    !
    open(1,file='obs/obs-'//trim(obs_id)//'_'//trim(obs_id_num)//'.dat')
    do i=1,nx,inc
      do j=1,ny,inc
       call vort_x(pv,pw,i,j,ind,dl,dzf,vort)
       write(1,*) vort
      end do
    end do
    close(1)
  end subroutine get_obs_drl_vort


  subroutine read_input_drl(myid)
    use mpi
    implicit none
    integer , intent(in   ) :: myid
    integer :: ierr, id_in
    namelist /stw_drl/ &
                      offset,  &
                      n_act,   &
                      bnx,     &
                      bny,     &
                      drl_inc, &
                      omega
    id_in = 101
    open(newunit=id_in,file='drl.nml',status='old',action='read',iostat=ierr)
      if (ierr==0) then
        read(id_in,nml=stw_drl,iostat=ierr)
      else
        if (myid==0) print*, 'Error reading DRl file'
        if (myid==0) print*, 'Aborting...'
        call MPI_FINALIZE(ierr)
        error stop
      end if
    close(id_in)
  end subroutine read_input_drl
  
  subroutine write_drl_io(myid,t_max,dpdx)
    implicit none
    integer , intent(in   ) :: myid
    real(rp), intent(in   ) :: t_max, dpdx
    integer :: id_out
    
    id_out = 102
    open(newunit=id_out,file='drl.nml',status='replace')
      write(id_out,*) '&stw_drl'
      write(id_out,*) 'n_act=' , n_act+1.
      write(id_out,*) 'offset=', offset + t_max*omega
      !write(id_out,*) ''
    close(id_out)
  
    id_out = 103
    open(newunit=id_out,file='dpdx.dat')
      write(id_out,*) dpdx
    close(id_out)
  end subroutine write_drl_io

  subroutine init_drl_var(id)
    implicit none
    integer , intent(in   ) :: id
    drl_flag  = .false.
    drl_dt    = 9.9
    drl_dtn   = 1.
    drl_dpdx  = 0.
    drl_ind_mean  = 69
    e_ks          = 0.
    e_ks_rew      = 0.
    e_ks_1        = 0.
    e_kw          = 0.
    e_kw_1        = 0.
    e_kw_2        = 0.
    u_obs         = 0.
    !u_obs_all     = 0.
    send_drl      = .false.
    req           = 'NULLL'
    n_act     = 1
    start_drl = .true.
    send_drl  = .false.
    
    if (id == 0) print*, 'DRL vars initialised'
  end subroutine init_drl_var

  subroutine drl_read_request(re,t_max,dpdx,off,obs,done,kill)
    implicit none
    character(len=5), intent(in) :: re
    real(rp), intent(in   ) :: t_max
    real(rp), intent(inout) :: dpdx,off
    real(rp), intent(inout), dimension(:,:) :: obs
    logical , intent(inout) :: done, kill

    if (re == 'START') then
      dpdx    = 0.
      obs     = 0.
      off     = off + omega*t_max
      call MPI_BCAST(omega,1,MPI_DOUBLE,0,intracomm,ierr)
      if (myid==0) print*, 'omega = ', omega 
    else if (re == 'CONTN') then
      dpdx    = 0.
      obs     = 0.
      off     = off + omega*t_max
      call MPI_BCAST(omega,1,MPI_DOUBLE,0,intracomm,ierr)
      if (myid==0) print*, 'omega = ', omega 
    else
      print*, 'Wrong message sent from Python: ',req,', aborting...'
      done = .true.
      kill    = .true.
    end if
  end subroutine drl_read_request

  subroutine drl_end_episode(re,done,kill)
    implicit none
    character(len=5), intent(in) :: re
    logical , intent(inout) :: done, kill

    if (re == 'CONTN') then
      if (myid == 0) print*, re, ' executed at the end'
      done = .false.
    else if (re == 'CLOSE') then
      if (myid == 0) print*, re, ' executed at the end'
      done = .true.
    else if (re == 'ENDED') then
      if (myid==0) print*, 'TRAINING DONE, check saved data'
      done = .true.
      kill = .true.
    else   
      if (myid==0) print*, 'Something went wrong with the communication, aborting'
      if (myid==0) print*, 'I received ', re
      done = .true.
      kill = .true.
    end if
  end subroutine drl_end_episode

  subroutine undersample(p,ind,inc,p_us,idd)
  implicit none
  real(rp), intent(in   ), dimension(0:,0:,0:) :: p
  integer , intent(in   ) :: ind,inc
  real(rp), intent(  out), dimension(:,:)    :: p_us
  integer , intent(in   ) :: idd
  integer :: nx,ny,i,j
  
  nx = size(p,1)-2
  ny = size(p,2)-2

  if ((mod(nx,inc).ne.0).or.(mod(ny,inc).ne.0)) then
    error stop "Something went wrong - nx or ny are not multiples of inc"
  end if

  do i = 1,nx,inc
    do j = 1,ny,inc
      if (inc.ne.1) then
        p_us((i+inc)/inc,(j+inc)/inc) = p(i,j,ind)
        !p_us((i+1)/inc,(j+1)/inc2) = real(idd)
      else
        p_us(i,j) = p(i,j,ind)
      end if
    enddo
  enddo
  
  end subroutine undersample

  subroutine interp_obs(mat_old,mat_new)
  implicit none
  real(rp), intent(in   ), dimension(:,:) :: mat_old
  real(rp), intent(  out), dimension(:,:) :: mat_new
  integer  :: new_x, new_y, old_x, old_y
  integer  :: i, j
  real(rp) :: ratio, x_mapped, y_mapped
  integer  :: x_low, y_low
  real(rp) :: x_diff, y_diff
  real(rp) :: tl, tr, bl, br


  old_x = size(mat_old,1)
  old_y = size(mat_old,2)
  new_x = size(mat_new,1)
  new_y = size(mat_new,2)

  ratio = old_x/new_x

  do i = 1, new_x
    do j = 1, new_y
      x_mapped = (i - 1) * ratio
      y_mapped = (j - 1) * ratio
      x_low = FLOOR(x_mapped)
      y_low = FLOOR(y_mapped)
      x_diff = x_mapped - x_low
      y_diff = y_mapped - y_low

      if (x_low >= old_x - 1) x_low = old_x - 2
      if (y_low >= old_y - 1) y_low = old_y - 2

      tl = mat_old(x_low + 1, y_low + 1)
      tr = mat_old(x_low + 2, y_low + 1)
      bl = mat_old(x_low + 1, y_low + 2)
      br = mat_old(x_low + 2, y_low + 2)

      mat_new(i, j) = (1.0 - x_diff) * (1.0 - y_diff) * tl + &
                      x_diff * (1.0 - y_diff) * tr + &
                      (1.0 - x_diff) * y_diff * bl + &
                      x_diff * y_diff * br
    end do
  end do
  end subroutine interp_obs

end module mod_drl
