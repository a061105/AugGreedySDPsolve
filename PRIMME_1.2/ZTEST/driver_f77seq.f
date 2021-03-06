************************************************************************
*   PRIMME PReconditioned Iterative MultiMethod Eigensolver
*   Copyright (C) 2005  James R. McCombs,  Andreas Stathopoulos
*
*   This file is part of PRIMME.
*
*   PRIMME is free software; you can redistribute it and/or
*   modify it under the terms of the GNU Lesser General Public
*   License as published by the Free Software Foundation; either
*   version 2.1 of the License, or (at your option) any later version.
*
*   PRIMME is distributed in the hope that it will be useful,
*   but WITHOUT ANY WARRANTY; without even the implied warranty of
*   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
*   Lesser General Public License for more details.
*
*   You should have received a copy of the GNU Lesser General Public
*   License along with this library; if not, write to the Free Software
*   Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA
*   02110-1301  USA
*
* ----------------------------------------------------------------------
*
*  Sequential driver using zprimme's Fortran interface. Calling format:
*
*          seqf77_zprimme 
*
*               Filename and parameters are hard coded in this program.
*               Reads only full (not triangular) coordinate matrices.
*
*               Preconditioning is set by PrecChoice int parameter:
*               0    no preconditioner
*               1    K = Diagonal of (A - shift I)
*               2    K = (Diagonal_of_A - primme.shift_i I)
*
*               For PRIMME parameters look in the readme.txt file.
* ----------------------------------------------------------------------

        Program primmeF77Driver
!-----------------------------------------------------------------------
        implicit none
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!       Pointer to the PRIMME data structure used internally by PRIMME
!
!       Note that for 64 bit systems, pointers are 8 bytes so use:
        ! integer*8 primme
        integer primme
        include 'primme_f77.h'
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!       Problem setup, parameters, filenames etc.
!       Alternatively, read these (except for max dimensions) from a file
! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        ! Matrix information
        character*256 MatrixFileName
        data MatrixFileName /'HA.mtx'/
        integer Nmax,NZmax
        parameter(Nmax = 245, NZmax = 60025)

        ! Preconditioning parameters
        integer PrecChoice
        real*8 shift
        parameter (
     :            PrecChoice      = 0,
     :            shift           = 0.1
     :  )

        ! Solver Parameters
        integer NUMEmax,BASISmax,BLOCKmax,maxMatvecs,
     :          printLevel, method, whichEvals, numTargetShifts
        real*8 ETOL

        parameter (
     :            BASISmax        = 25,
     :            NUMEmax         = 4,
     :            BLOCKmax        = 1,
     :            maxMatvecs      = 300000,
     :            ETOL            = 1.0D-14,
     :            printLevel      = 3,
     :            whichEvals      = PRIMMEF77_closest_geq,
     :            numTargetShifts = 4,
     :            method          = PRIMMEF77_JDQMR_ETol
     :  )
        real*8 TargetShifts(numTargetShifts)
        data TargetShifts /0.0, 0.0, 0.17, 0.17 /

! - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
!
!       Matrix 
!
        integer n, ja(NZmax),ia(Nmax+1),iau(Nmax+1)
        complex*16   a(NZmax)
        common /Matrix/ a,ja,ia,iau,n
!
!       Preconditioner
!
        integer lenFactmax
        parameter(lenFactmax = 2*NZmax)
        integer F_n,Fact_JA(lenFactmax),Fact_IA(Nmax+1),Fact_IAU(Nmax+1)
        complex*16   Fact_A(lenFactmax)
        common /Factors/ Fact_A,Fact_JA,Fact_IA,Fact_IAU, F_n

!       Eigenvalues, eigenvectors, and their residual norms
!
        real*8     evals(NUMEmax), rnorms(NUMEmax)
        complex*16 evecs(Nmax*NUMEmax)

!       Other vars
!
        integer infile
        integer nze,i,j,ierr,ncol,length
        real*8  fnorm, epsil, aNorm, temp, tempi
        complex*16 ztemp
        character*256 title

!       External subroutines
        external MV, Apply_Inv_Diag_Prec, Apply_Diag_Shifted_Prec


!-----------------------------------------------------------------------
!       Start executable 
!-----------------------------------------------------------------------
!
!       Read the matrix in coordinate format and convert to CSR
!       ----------------------------------------------------------------
        infile = 11
        open(infile, file=MatrixFileName)

        ! Read any initial comments from an MTX format
 10     read(infile, '(A)', END=996) title
        if (title(1:1) .eq. '%') goto 10
        i = len_trim(title)
        print*, i
        read (title(1:i), *) n
        print*, title(1:i), '    ',n

        ia(1) = 1
        ncol = 1
        j = 1

 20     read(infile,*,END=30) ja(j), i, temp, tempi
           a(j) = DCMPLX(temp,tempi)
           if (i .ne. ncol)  then
              ncol = i
              IA(i) = j
           endif
           j=j+1
        goto 20
 30     continue
        IA(n+1) = j
        nze = j-1 
!
!     find the index of the diagonal elements (for diag preconditioning)
!
        do i = 1,n
           do j=ia(i),ia(i+1)-1
              if (i.eq.ja(j)) then
                iau(i) = j
                ! Force diagonal to be real, but avoid real(),realpart()
                a(j) = (DCONJG(a(j))+a(j))/2.0  
              endif
           enddo
        enddo
!
!       Compute Frobenious norm of A
!
        fnorm = 0.d0
        do i =1, nze
           fnorm = fnorm + DBLE(DCONJG(a(i))*a(i))
        enddo
        fnorm = sqrt(fnorm)
!       ----------------------------------------------------------------
!       Build the preconditioner
!       ----------------------------------------------------------------
        F_n = n
        if (PrecChoice .eq. 1) then
           ! preinverted diagonal preconditioner (diag(A)-shiftI)^(-1)
           ! User provided shift 
           temp = 1d-15*fnorm;
           do i=1,n
              Fact_IA(i) = i
              Fact_JA(i) = i
              ztemp =  A(iau(i)) - shift
              if (zabs(ztemp) < temp) then
                 Fact_A(i) = 1.0/temp
              else
                 Fact_A(i) = 1.0/ztemp
              endif
           enddo
           Fact_IA(i+1) = n+1 
        elseif (PrecChoice .eq. 2) then
           ! Preconditioner (diag(A)-(shift_i) ) with PRIMME provided shift_i
           ! Will be inverted on application
           do i=1,n
              Fact_IA(i) = i
              Fact_JA(i) = i
              Fact_A(i) = A(iau(i))
           enddo
           Fact_IA(i+1) = n+1 
        elseif (PrecChoice .eq. 3) then
           print*, "No complex ILUT. Choose a different preconditioner"
           stop
        endif
!       ----------------------------------------------------------------
!       Initialize PRIMME
!       ----------------------------------------------------------------
!
        call primme_initialize_f77(primme)

!       Set a few basic solver parameters  (only n is really required)
        call primme_set_member_f77(primme, PRIMMEF77_n, n)
        call primme_set_member_f77(primme, PRIMMEF77_numEvals, NUMEmax)
        call primme_set_member_f77(primme, PRIMMEF77_maxBasisSize, 
     :                                                        BASISmax)
        call primme_set_member_f77(primme, PRIMMEF77_maxBlockSize,
     :                                                        BLOCKmax)

!       Set matvec 
        call primme_set_member_f77(primme, PRIMMEF77_matrixMatvec, MV)
        
!       Set preconditioner 
        call primme_set_member_f77(primme, 
     :       PRIMMEF77_correctionParams_precondition, 1)
        if (PrecChoice.eq.0 .OR. PrecChoice.gt.2) then
          call primme_set_member_f77(primme, 
     :        PRIMMEF77_correctionParams_precondition, 0)
        elseif (PrecChoice .eq. 1) then
          call primme_set_member_f77(primme, 
     :        PRIMMEF77_applyPreconditioner, Apply_Inv_Diag_Prec )
        elseif (PrecChoice .eq. 2) then
          call primme_set_member_f77(primme, 
     :        PRIMMEF77_applyPreconditioner, Apply_Diag_Shifted_Prec )
        endif 
!
!       Set the method to be used (after n, numEvals, and precondition have
!       been set. Also after basisSize is set if desired.)

        call primme_set_method_f77(primme, method, ierr)

        if (ierr .lt. 0) 
     :     write(*,*) 'No preset method. Using custom settings'
!
!       Set a few other solver parameters  
!
        call primme_set_member_f77(primme, PRIMMEF77_aNorm, fnorm)
        call primme_set_member_f77(primme, PRIMMEF77_eps, ETOL)
        call primme_set_member_f77(primme, PRIMMEF77_target, whichEvals)
        call primme_set_member_f77(primme, PRIMMEF77_numTargetShifts, 
     :                                                 numTargetShifts)
        call primme_set_member_f77(primme, PRIMMEF77_targetShifts, 
     :                                                    TargetShifts)
        call primme_set_member_f77(primme, PRIMMEF77_printLevel, 
     :                                                      printLevel)
        call primme_set_member_f77(primme, PRIMMEF77_maxMatvecs,
     :                                                      maxMatvecs)
        call primme_set_member_f77(primme, 
     :              PRIMMEF77_restartingParams_scheme, PRIMMEF77_thick)

!       ----------------------------------------------------------------
!       Initial estimates
!       ----------------------------------------------------------------
!
        i = 1
        call primme_set_member_f77(primme, PRIMMEF77_initSize, i)
        do i=1,n
           evecs(i) = DCMPLX(1.0D0/sqrt(dble(n)),0.d0)
        enddo
!       ----------------------------------------------------------------
!       Display what parameters are used
!       ----------------------------------------------------------------
        call primme_display_params_f77(primme)
!       ----------------------------------------------------------------
!       Calling the PRIMME solver
!       ----------------------------------------------------------------

        call zprimme_f77(evals, evecs, rnorms, primme, ierr)

!       ----------------------------------------------------------------
!       Reporting results

        if (ierr.eq.0) then
           print *, 'ZPRIMME has returned successfully'
        else 
           print *, 'ZPRIMME returned with error: ', ierr
        endif

        call primme_display_stats_f77(primme)
!       
!       Example of obtaining primme members from the driver:
!
        call primmetop_get_member_f77(primme, PRIMMEF77_eps, epsil)
        call primmetop_get_member_f77(primme, PRIMMEF77_aNorm, aNorm)
        print*, 'Tolerance used: ',epsil, ' Esimated norm(A):', aNorm
!
!       Reporting of evals and residuals
!
        do i = 1, numemax
           write (*, 9000) i, evals(i),rnorms(i)
        enddo
 9000   FORMAT (1x,'E(',i1,') = ',G24.16,4x,
     &         'residual norm =', E12.4)

        stop
 996    write(0,*) 'ERROR! No data in the file'
        stop
        end
!-----------------------------------------------------------------------
! Supporting subroutines
!-----------------------------------------------------------------------
!       ----------------------------------------------------------------
        subroutine MV(x,y,k,primme)
!       ----------------------------------------------------------------
        complex*16 x(*), y(*)
        integer k, primme
        parameter(Nmax = 245, NZmax = 60025)
        integer n, ja(NZmax),ia(Nmax+1),iau(Nmax+1)
        complex*16   a(NZmax)
        common /Matrix/ a,ja,ia,iau,n
        external zamux

        do i=1,k
           call zamux(n, x(n*(i-1)+1), y(n*(i-1)+1), A, JA, IA)
        enddo
        end

!       ----------------------------------------------------------------
        subroutine Apply_Inv_Diag_Prec(x,y,k,primme)
!       ----------------------------------------------------------------
        complex*16 x(*), y(*) 
        integer k, primme
        integer Nmax,NZmax
        parameter(Nmax = 245, NZmax = 60025)
        integer lenFactmax
        parameter(lenFactmax = 2*NZmax)
        integer n, Fact_JA(lenFactmax),Fact_IA(Nmax+1),Fact_IAU(Nmax+1)
        complex*16   Fact_A(lenFactmax)
        common /Factors/ Fact_A,Fact_JA,Fact_IA,Fact_IAU, n
        integer i
        external zamux

        do i=1,k
          call zamux(n,x(n*(i-1)+1),y(n*(i-1)+1),Fact_A,Fact_JA,Fact_IA)
        enddo
        end

!       ----------------------------------------------------------------
        subroutine Apply_Diag_Shifted_Prec(x,y,k,primme)
!       ----------------------------------------------------------------
        complex*16 x(*), y(*) 
        integer k, primme
        integer Nmax,NZmax
        parameter(Nmax = 245, NZmax = 60025)
        integer lenFactmax
        parameter(lenFactmax = 2*NZmax)
        integer n, Fact_JA(lenFactmax),Fact_IA(Nmax+1),Fact_IAU(Nmax+1)
        complex*16   Fact_A(lenFactmax)
        common /Factors/ Fact_A,Fact_JA,Fact_IA,Fact_IAU, n
        integer i,j
        real*8 shift
        complex*16 denominator

        do i=1,k
           call primme_get_prec_shift_f77(primme, i, shift)
           do j=1,n
              denominator = Fact_A(j) - shift
              if (zabs(denominator) < 1D-14) then
                 y(n*(i-1)+j) = x(n*(i-1)+j)/denominator
              else
                 y(n*(i-1)+j) = x(n*(i-1)+j)*1.0D14
              endif
           enddo
        enddo
        end
