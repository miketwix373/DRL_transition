! -
!
! SPDX-FileCopyrightText: Copyright (c) 2017-2022 Pedro Costa and the CaNS contributors. All rights reserved.
! SPDX-License-Identifier: MIT
!
! -
module mod_utils
      use mod_types
      use mod_common_mpi
      use mpi


  implicit none
  private
  public bulk_mean,f_sizeof,swap, linear_interp, read_eigenval, read_waveparam, linear_interp2
#if defined (_DRL)
  public bulk_mean_mod, bulk_mean_mod_sq, vort_x
#endif
  !@acc public device_memory_footprint
contains
subroutine read_eigenval(zf, zc, upert, vpert, reDelta0,n,datadir)
    ! Reads eigenvalue data and interpolates to zf and zc grids

    real(rp), dimension(0:), intent(in) :: zf, zc
    complex(rp), dimension(0:), intent(inout) :: vpert, upert
    real(rp), intent(in) :: reDelta0
    integer,intent(in) :: n(3) 
    character(len=*),intent(in),optional :: datadir
    character(len=120):: filename

    ! Local variables
    integer :: i, io_error, n_lines, unit_id
    real(rp), allocatable, dimension(:) :: z_data
    real(rp), allocatable, dimension(:) :: v_re_data, v_im_data
    real(rp), allocatable, dimension(:) :: u_re_data, u_im_data
    real(rp), allocatable, dimension(:) :: v_real_interp, v_imag_interp
    real(rp), allocatable, dimension(:) :: u_real_interp, u_imag_interp
    character(len=256) :: z_filename, v_re_filename, v_im_filename
    character(len=256) :: u_re_filename, u_im_filename
    logical :: file_exists
    real(rp) :: z_val
    
    ! Fixed filenames
    z_filename = 'z_eigen.txt'
    v_re_filename = 'real_v.txt'
    v_im_filename = 'imag_v.txt'
    u_re_filename = 'real_u.txt'
    u_im_filename = 'imag_u.txt'
    
    upert = 0.0
    vpert = 0.0

    ! Only root process reads the files
    if (myid == 0) then
      ! Check if z file exists
      inquire(file=trim(z_filename), exist=file_exists)
      if (.not. file_exists) then
        write(*,*) 'ERROR: File ', trim(z_filename), ' does not exist!'
        call MPI_Abort(canscomm, 1, io_error)
      end if
      
      ! Count number of lines in z file to determine array sizes
      open(newunit=unit_id, file=trim(z_filename), status='old', action='read')
      n_lines = 0
      do
        read(unit_id, *, iostat=io_error)
        if (io_error /= 0) exit
        n_lines = n_lines + 1
      end do
      close(unit_id)
      
      ! Allocate arrays
      allocate(z_data(n_lines))
      allocate(v_re_data(n_lines), v_im_data(n_lines))
      allocate(u_re_data(n_lines), u_im_data(n_lines))
      
      ! Read z data - handle scientific notation format
      open(newunit=unit_id, file=trim(z_filename), status='old', action='read')
      do i = 1, n_lines
        read(unit_id, *, iostat=io_error) z_val
        if (io_error /= 0) then
          write(*,*) 'ERROR: Problem reading z data at line', i
          call MPI_Abort(canscomm, 1, io_error)
        end if
        z_data(i) = z_val
      end do
      close(unit_id)
      
      ! Read v real part
      open(newunit=unit_id, file=trim(v_re_filename), status='old', action='read')
      do i = 1, n_lines
        read(unit_id, *, iostat=io_error) v_re_data(i)
        if (io_error /= 0) then
          write(*,*) 'ERROR: Problem reading v real data at line', i
          call MPI_Abort(canscomm, 1, io_error)
        end if
      end do
      close(unit_id)
      
      ! Read v imaginary part
      open(newunit=unit_id, file=trim(v_im_filename), status='old', action='read')
      do i = 1, n_lines
        read(unit_id, *, iostat=io_error) v_im_data(i)
        if (io_error /= 0) then
          write(*,*) 'ERROR: Problem reading v imaginary data at line', i
          call MPI_Abort(canscomm, 1, io_error)
        end if
      end do
      close(unit_id)
      
      ! Read u real part
      open(newunit=unit_id, file=trim(u_re_filename), status='old', action='read')
      do i = 1, n_lines
        read(unit_id, *, iostat=io_error) u_re_data(i)
        if (io_error /= 0) then
          write(*,*) 'ERROR: Problem reading u real data at line', i
          call MPI_Abort(canscomm, 1, io_error)
        end if
      end do
      close(unit_id)
      
      ! Read u imaginary part
      open(newunit=unit_id, file=trim(u_im_filename), status='old', action='read')
      do i = 1, n_lines
        read(unit_id, *, iostat=io_error) u_im_data(i)
        if (io_error /= 0) then
          write(*,*) 'ERROR: Problem reading u imaginary data at line', i
          call MPI_Abort(canscomm, 1, io_error)
        end if
      end do
      close(unit_id)
      
      ! Allocate arrays for interpolation
      allocate(v_real_interp(size(zf)))
      allocate(v_imag_interp(size(zf)))
      allocate(u_real_interp(size(zc)))
      allocate(u_imag_interp(size(zc)))
      
      ! Interpolate real and imaginary parts separately
      call linear_interp2(z_data, v_re_data, zf/reDelta0, v_real_interp)
      call linear_interp2(z_data, v_im_data, zf/reDelta0, v_imag_interp)
      
      call linear_interp2(z_data, u_re_data, zc/reDelta0, u_real_interp)
      call linear_interp2(z_data, u_im_data, zc/reDelta0, u_imag_interp)
      
      ! Combine real and imaginary parts
      do i = 1, size(zf)
        vpert(i) = cmplx(v_real_interp(i), v_imag_interp(i), kind=rp)
      end do

      do i = 1, size(zc)
        upert(i) = cmplx(u_real_interp(i), u_imag_interp(i), kind=rp)
      end do
      
      ! Free memory
      !deallocate(z_data)
      !deallocate(v_re_data, v_im_data)
      !deallocate(u_re_data, u_im_data)
      !deallocate(v_real_interp, v_imag_interp)
      !deallocate(u_real_interp, u_imag_interp)
    end if
    
    ! Broadcast interpolated values to all processes
    call MPI_Bcast(vpert, size(vpert), MPI_COMPLEX_RP, 0, canscomm, ierr)
    call MPI_Bcast(upert, size(upert), MPI_COMPLEX_RP, 0, canscomm, ierr)

    if (present(datadir)) then
      if (myid.eq.0) then
          filename = trim(datadir)//'non_interp.out'
          open(99, file=filename)
            do i=1,n_lines
              write(99,*) z_data(i), v_re_data(i), v_im_data(i), u_re_data(i), u_im_data(i)
            end do
          close(99)
      end if
    end if
    

    if (present(datadir)) then
      if (myid.eq.0) then
          filename = trim(datadir)//'interp.out'
          open(99, file=filename)
            do i=0,n(3)+1
              write(99,*) zf(i), zf(i)/reDelta0, real(vpert(i)), real(upert(i)) 
            end do
          close(99)
      end if
    end if
  end subroutine read_eigenval

  !-----------------------------------------------------------------------------

  subroutine read_waveparam(alpha, beta_r, beta_i)

    ! Reads wave parameters from waveparam.txt file

    real(rp), intent(inout) :: alpha, beta_r, beta_i
    ! Local variables
    integer :: io_error, unit_id
    character(len=256) :: wave_filename
    logical :: file_exists
    real(rp) :: alpha_temp, beta_r_temp, beta_i_temp

    ! Fixed filename
    wave_filename = 'waveparam.txt'
    
    ! Initialize values
    alpha = 0.0_rp
    beta_r = 0.0_rp
    beta_i = 0.0_rp

    ! Initialize values
    alpha_temp = 0.0_rp
    beta_r_temp = 0.0_rp
    beta_i_temp = 0.0_rp

    ! Only root process reads the file
    if (myid == 0) then
      ! Check if waveparam file exists and read it
      inquire(file=trim(wave_filename), exist=file_exists)
      if (.not. file_exists) then
        write(*,*) 'WARNING: File ', trim(wave_filename), ' does not exist!'
        write(*,*) 'Setting alpha=0, beta_r=0, beta_i=0'
      else
        ! Read wave parameters (alpha, beta_r, beta_i in single column)
        open(newunit=unit_id, file=trim(wave_filename), status='old', action='read')
        
        read(unit_id, *, iostat=io_error) alpha_temp
        if (io_error /= 0) then
          write(*,*) 'ERROR: Problem reading alpha from ', trim(wave_filename)
          call MPI_Abort(canscomm, 1, io_error)
        end if
        
        read(unit_id, *, iostat=io_error) beta_r_temp
        if (io_error /= 0) then
          write(*,*) 'ERROR: Problem reading beta_r from ', trim(wave_filename)
          call MPI_Abort(canscomm, 1, io_error)
        end if
        
        read(unit_id, *, iostat=io_error) beta_i_temp
        if (io_error /= 0) then
          write(*,*) 'ERROR: Problem reading beta_i from ', trim(wave_filename)
          call MPI_Abort(canscomm, 1, io_error)
        end if
        
        close(unit_id)
        
        write(*,*) 'Read wave parameters: alpha=', alpha_temp, ' beta_r=', beta_r_temp, ' beta_i=', beta_i_temp
      end if
    end if
    
    ! Broadcast wave parameters to all processes
    call MPI_ALLREDUCE(alpha_temp, alpha, 1, MPI_REAL_RP, MPI_SUM, canscomm, ierr)
    call MPI_ALLREDUCE(beta_r_temp, beta_r, 1, MPI_REAL_RP, MPI_SUM, canscomm, ierr)
    call MPI_ALLREDUCE(beta_i_temp, beta_i, 1, MPI_REAL_RP, MPI_SUM, canscomm, ierr)
    
  end subroutine read_waveparam


  subroutine linear_interp(x, y, xnew, ynew)
    use mod_types
    ! x - Input x coordinates (must be monotonically increasing)
    ! y - Input y values at x points
    ! xnew - Points where interpolated values are desired
    ! ynew - Output interpolated values (must be pre-allocated)
    real(rp), intent(in), dimension(:) :: x, y, xnew
    real(rp), intent(out), dimension(:) :: ynew
    integer :: i, j, n, nnew, idx_low
    real(rp) :: t, x_val
    ! Get array sizes
    n = size(x)
    nnew = size(xnew)
    ! Binary search for each xnew point
    idx_low = 1
    do i = 1, nnew
    x_val = xnew(i)
    ! Handle lower extrapolation case
    if (x_val <= x(1)) then
    ! Linear extrapolation using first two points
    t = (x_val - x(1)) / (x(2) - x(1))
    ynew(i) = y(1) + t * (y(2) - y(1))
    cycle
    endif
    ! Handle upper extrapolation case
    if (x_val >= x(n)) then
    ! Linear extrapolation using last two points
    t = (x_val - x(n-1)) / (x(n) - x(n-1))
    ynew(i) = y(n-1) + t * (y(n) - y(n-1))
    cycle
    endif
    if (x_val < x(idx_low)) then
    idx_low = 1
    endif
    ! Start searching from the previous found position
    j = idx_low
    do while (j < n .and. x(j+1) <= x_val)
    j = j + 1
    end do
    ! Store this position for the next search
    idx_low = j
    ! Linear interpolation
    t = (x_val - x(j)) / (x(j+1) - x(j))
    ynew(i) = y(j) + t * (y(j+1) - y(j))
    end do
  end subroutine linear_interp

  subroutine linear_interp2(x, y, xnew, ynew)
    use mod_types
    ! x - Input x coordinates (must be monotonically increasing)
    ! y - Input y values at x points
    ! xnew - Points where interpolated values are desired
    ! ynew - Output interpolated values (must be pre-allocated)
    real(rp), intent(in), dimension(:) :: x, y, xnew
    real(rp), intent(out), dimension(:) :: ynew
    integer :: i, j, n, nnew, idx_low
    real(rp) :: t, x_val
    ! Get array sizes
    n = size(x)
    nnew = size(xnew)
    ! Binary search for each xnew point
    idx_low = 1
    do i = 1, nnew
    x_val = xnew(i)
    ! Handle lower boundary case - fix to first value
    if (x_val <= x(1)) then
    ynew(i) = y(1)
    cycle
    endif
    ! Handle upper boundary case - fix to last value
    if (x_val >= x(n)) then
    ynew(i) = y(n)
    cycle
    endif
    if (x_val < x(idx_low)) then
    idx_low = 1
    endif
    ! Start searching from the previous found position
    j = idx_low
    do while (j < n .and. x(j+1) <= x_val)
    j = j + 1
    end do
    ! Store this position for the next search
    idx_low = j
    ! Linear interpolation
    t = (x_val - x(j)) / (x(j+1) - x(j))
    ynew(i) = y(j) + t * (y(j+1) - y(j))
    end do
  end subroutine linear_interp2

  subroutine bulk_mean(n,grid_vol_ratio,p,mean)
    !
    ! compute the mean value of an observable over the entire domain
    !
    use mpi
    use mod_types
    implicit none
    integer , intent(in), dimension(3) :: n
    real(rp), intent(in), dimension(0:) :: grid_vol_ratio
    real(rp), intent(in), dimension(0:,0:,0:) :: p
    real(rp), intent(out) :: mean
    integer :: i,j,k
    integer :: ierr
    mean = 0.
    !$acc data copy(mean) async(1)
    !$acc parallel loop collapse(3) default(present) reduction(+:mean) async(1)
    !$OMP PARALLEL DO   COLLAPSE(3) DEFAULT(shared)  REDUCTION(+:mean)
    do k=1,n(3)
      do j=1,n(2)
        do i=1,n(1)
          mean = mean + p(i,j,k)*grid_vol_ratio(k)
        end do
      end do
    end do
    !$acc end data
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,mean,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
  end subroutine bulk_mean

  subroutine bulk_mean_mod(n,grid_vol_ratio,p,ind,mean)
    !
    ! compute the mean value of an observable over a box
    ! of dimensions (Lx, Ly, z(ind))
    !
    use mpi
    use mod_types
    implicit none
    integer , intent(in), dimension(3) :: n
    real(rp), intent(in), dimension(0:) :: grid_vol_ratio
    real(rp), intent(in), dimension(0:,0:,0:) :: p
    integer , intent(in)  :: ind
    real(rp), intent(out) :: mean
    integer :: i,j,k
    integer :: ierr
    mean = 0.
    !$acc data copy(mean) async(1)
    !$acc parallel loop collapse(3) default(present) reduction(+:mean) async(1)
    !$OMP PARALLEL DO   COLLAPSE(3) DEFAULT(shared)  REDUCTION(+:mean)
    do k=1,ind
      do j=1,n(2)
        do i=1,n(1)
          mean = mean + p(i,j,k)*grid_vol_ratio(k)
        end do
      end do
    end do
    !$acc end data
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,mean,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
  end subroutine bulk_mean_mod

  subroutine bulk_mean_mod_sq(n,grid_vol_ratio,p,ind,mean)
    !
    ! compute the mean value of an observable over a box
    ! of dimensions (Lx, Ly, z(ind))
    !
    use mpi
    use mod_types
    implicit none
    integer , intent(in), dimension(3) :: n
    real(rp), intent(in), dimension(0:) :: grid_vol_ratio
    real(rp), intent(in), dimension(0:,0:,0:) :: p
    integer , intent(in)  :: ind
    real(rp), intent(out) :: mean
    integer :: i,j,k
    integer :: ierr
    mean = 0.
    !$acc data copy(mean) async(1)
    !$acc parallel loop collapse(3) default(present) reduction(+:mean) async(1)
    !$OMP PARALLEL DO   COLLAPSE(3) DEFAULT(shared)  REDUCTION(+:mean)
    do k=1,ind
      do j=1,n(2)
        do i=1,n(1)
          mean = mean + (p(i,j,k)**2)*grid_vol_ratio(k)
        end do
      end do
    end do
    !$acc end data
    !$acc wait(1)
    call MPI_ALLREDUCE(MPI_IN_PLACE,mean,1,MPI_REAL_RP,MPI_SUM,canscomm,ierr)
  end subroutine bulk_mean_mod_sq

  subroutine vort_x(pv,pw,indi,indj,indk,dl,dzf,vort)
    use mod_types
    implicit none
    !
    ! compute streamwise vorticity at x(indi),y(indj),z(indk)
    !
    real(rp), intent(in   ), dimension(0:,0:,0:) :: pv,pw
    integer , intent(in   )                      :: indi,indj,indk
    real(rp), intent(in   ), dimension(3 )       :: dl
    real(rp), intent(in   ), dimension(0:)       :: dzf
    real(rp), intent(  out)                      :: vort
    real(rp)  :: dwdy,dvdz
    
    dwdy = pw(indi,indj+1,indk+1) + pw(indi,indj+1,indk  ) + pw(indi,indj-1,indk  ) + pw(indi,indj-1,indk+1)
    dvdz = pv(indi,indj+1,indk+1) + pv(indi,indj  ,indk  ) + pv(indi,indj  ,indk-1) + pv(indi,indj+1,indk-1)
    vort = 0.25*(dwdy/dl(2) - dvdz/dzf(indk))
  end subroutine vort_x

  pure integer function f_sizeof(val) result(isize)
    !
    ! returns storage size of the scalar argument val in bytes
    !
    implicit none
    class(*), intent(in) :: val
    isize = storage_size(val)/8
  end function f_sizeof
  subroutine swap(arr1,arr2)
    use mod_types, only: rp
    implicit none
    real(rp), intent(inout), pointer, contiguous, dimension(:,:,:) :: arr1,arr2
    real(rp),                pointer, contiguous, dimension(:,:,:) :: tmp
    tmp  => arr1
    arr1 => arr2
    arr2 => tmp
  end subroutine swap
#if defined(_OPENACC)
  function device_memory_footprint(n,n_z) result(itotal)
    !
    ! estimate GPU memory footprint, assuming one MPI task <-> one GPU
    !
    use mod_types, only: i8,rp
    integer, intent(in), dimension(3) :: n,n_z
    integer :: nh(3)
    integer(i8) :: itotal,itemp,rp_size
    rp_size = f_sizeof(1._rp)
    itotal = 0
    !
    ! 1. 'main' arrays: u,v,w,p,pp
    !
    nh(:) = 1
    itotal = itotal + product(n(:)+2*nh(:))*rp_size*5
    !
    ! 2. grids arrays: zc,zf,dzc,dzf,dzci,dzfi,grid_vol_ratio_c,grid_vol_ratio_f (tiny footprint)
    !
    nh(:) = 1
    itotal = itotal + (n(3)+2*nh(3))*rp_size*8
    !
    ! 3. solver eigenvalues and Gauss elimination coefficient arrays (small footprint)
    !    rhs?%[x,y,z] arrays, lambdaxy? arrays, and a?,b?,c? arrays
    !
    block
      integer(i8) :: itemp1,itemp1_(3),itemp2,itemp3
      itemp1_(:) = [n_z(2)*n_z(3)*2,n_z(1)*n_z(3)*2,n_z(1)*n_z(2)*2]
      itemp1 = sum(itemp1_(:))   ! rhs
      itemp2 = product(n_z(1:2)) ! lambdaxy
      itemp3 = n_z(3)*3          ! a,b,c
#if   !defined(_IMPDIFF)
      !
      ! rhsbp, lambdaxyp, ap,bp,cp
      !
      itotal = itotal + itemp1*rp_size                + itemp2*rp_size   + itemp3*rp_size
#elif  defined(_IMPDIFF_1D)
      !
      ! rhsbp,rhsb[u,v,w,buf]%z, lambdaxyp, a?,b?,c? [p,u,v,w,buf]
      !
      itotal = itotal + (itemp1+itemp1_(3)*4)*rp_size + itemp2*rp_size   + itemp3*rp_size*5
#else
      !
      ! rhsbp,rhsb[u,v,w,buf]%[x,y,z], lambdaxy[p,u,v,w], (a?,b?,c?)[p,u,v,w,buf]
      !
      itotal = itotal + itemp1*rp_size*(1+4)          + itemp2*rp_size*5 + itemp3*rp_size*5
#endif
    end block
    !
    ! 4. prediction velocity arrays arrays d[u,v,w]dtrk_t, d[u,v,w]dtrko_t
    !
    itemp  = product(n(:))*rp_size
    itotal = itotal + itemp*6
#if defined(_IMPDIFF)
    itotal = itotal + itemp*3
#endif
    !
    ! 5. transpose & FFT buffer arrays, halo buffer arrays, and solver arrays
    !    taken directly from `mod_common_cudecomp`
    !
    block
      use mod_common_cudecomp, only: work,work_halo,solver_buf_0,solver_buf_1,pz_aux_1,pz_aux_2
      itemp = storage_size(work        ,i8)*size(work        ) + storage_size(work_halo   ,i8)*size(work_halo   ) + &
              storage_size(solver_buf_0,i8)*size(solver_buf_0) + storage_size(solver_buf_1,i8)*size(solver_buf_1) + &
              storage_size(pz_aux_1    ,i8)*size(pz_aux_1    ) + storage_size(pz_aux_2    ,i8)*size(pz_aux_2    )
      itotal = itotal + itemp/8
    end block
  end function device_memory_footprint
#endif
end module mod_utils
