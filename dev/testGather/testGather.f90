program gatherv_example
    use mpi
    implicit none

    integer, parameter :: nx = 10, ny = 6
    integer, parameter :: pnx = 2, pny = 3
    integer :: rank, size, ierr, i
    integer :: custom_type, custom_type_old, typesize
    integer(kind=mpi_address_kind) :: lb, extent
    integer, allocatable :: recvcounts(:), displs(:)
    real, allocatable :: local_data(:,:), global_data(:,:)

    call MPI_Init(ierr)
    call MPI_Comm_rank(MPI_COMM_WORLD, rank, ierr)
    call MPI_Comm_size(MPI_COMM_WORLD, size, ierr)

    ! Check if we have the correct number of processes
    if (size /= nx*ny/pnx/pny) then
        if (rank == 0) then
            print *, 'This program requires exactly ',nx*ny/pnx/pny, ' processes.'
        endif
        call MPI_Finalize(ierr)
        stop
    endif

    ! Allocate local array for each process
    allocate(local_data(pnx, pny))

    ! Fill local_data with some values for demonstration
    local_data = real(rank)

    ! Define MPI datatype for the sub-domain
    call MPI_Type_create_subarray(2,[nx,ny],[pnx,pny],[0,0],MPI_ORDER_FORTRAN,MPI_INT,custom_type_old,ierr)
    call MPI_Type_size(MPI_INT, typesize, ierr)
    lb = 1
    extent = pnx*typesize
    call MPI_Type_create_resized(custom_type_old,lb,extent,custom_type,ierr)
    call MPI_Type_commit(custom_type, ierr)

    ! Allocate the global array and arrays for counts and displacements on the root process
    if (rank == 0) then
        allocate(global_data(nx, ny))
        allocate(recvcounts(size), displs(size))
    endif

    ! Calculate receive counts and displacements for all processes
    if (rank == 0) then
        do i = 0, size - 1
            recvcounts(i+1) = pnx*pny
            displs(i+1) = (mod(i,nx/pnx) + i/(nx/pnx)*pny*nx/pnx)
        end do
        global_data = 0.
        !displs = [0,1,2,6,7,8,12,13,14]
    endif
    
    if (rank==0) print*, 'displs = ', displs
    ! Perform the gatherv operation
    call MPI_Gatherv(local_data, pnx*pny, MPI_REAL, &
                     global_data, recvcounts, displs, custom_type, 0, MPI_COMM_WORLD, ierr)

    ! On the root process, the global_data now contains the entire domain
    if (rank == 0) then
        print*, 'Values = '
        do i = 1,nx
          print*, global_data(i,:)
        enddo
    endif

    call MPI_Finalize(ierr)
end program gatherv_example

