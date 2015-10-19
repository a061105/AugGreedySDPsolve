CC = g++
PRIMME_PATH = ../PRIMME_1.2
LINKER = ${CC}
CFLAGS = -O3 -g -Wall -fPIC -I${PRIMME_PATH}/PRIMMESRC/COMMONSRC

PROB_OBJ = maxcut.o util.o problem.o lp.o 

#---------------------------------------------------------------
#  Basic Link flags for LAPACK/BLAS and math
#---------------------------------------------------------------
# Mac OSX 10.4 using veclib for BLAS/LAPACK
#
LDFLAGS = -Wl,-framework -Wl,vecLib -m64
# Use -Wl,-framework -Wl,Accelerate -m64 for Mac OS X 10.3
LIBS = -L${PRIMME_PATH} -lprimme
#---------------------------------------------------------------
# Linux 2.6.13-15.12-smp (64 bit image)
#LIBS = -lprimme -lm -L/usr/lib64/ -llapack -lblas -lgfortran -lg2c
#ifeq ($(CC),c++)
#LIBS += -lstdc++
#endif

#---------------------------------------------------------------
# Preprocessor definition for using F77 under various OS/compilers
#---------------------------------------------------------------

DEFINES = -DF77UNDERSCORE
# If a C++ compiler is used instead of C add this
ifeq ($(CC),c++)
DEFINES = -DF77UNDERSCORE -DCplusplus
endif     

all:  clean sdp_omp 

sdp_omp: $(PROB_OBJ) omp.o
	${LINKER} -o sdp_omp ${PROB_OBJ} omp.o ${LDFLAGS} ${LIBS}

omp.o: $(PROB_OBJ) omp.cpp
	$(CC) $(CFLAGS) $(PROB_OBJ) -c omp.cpp -o omp.o   

maxcut.o: util.o
	$(CC) $(CFLAGS) ${DEFINES} -c maxcut.cpp -o maxcut.o

problem.o: problem.cpp util.o
	$(CC) $(CFLAGS) ${DEFINES} -c problem.cpp -o problem.o

lp.o: util.o
	$(CC) $(CFLAGS) ${DEFINES} -c lp.cpp -o lp.o

util.o:
	$(CC) $(CFLAGS) -c util.cpp -o util.o

clean:
	rm -f *.o *~ sdp_omp 
