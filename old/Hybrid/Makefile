CC = g++
PRIMME_PATH = ../../PRIMME_1.2
LINKER = ${CC}
CFLAGS = -O3 -fPIC -std=c++11 -I${PRIMME_PATH}/PRIMMESRC/COMMONSRC

PROB_OBJ = maxcut.o util.o problem.o lp.o 

#---------------------------------------------------------------
#  Basic Link flags for LAPACK/BLAS and math
#---------------------------------------------------------------
# Mac OSX 10.4 using veclib for BLAS/LAPACK
#
# LDFLAGS = -Wl,-framework -Wl,vecLib -m64
# Use -Wl,-framework -Wl,Accelerate -m64 for Mac OS X 10.3
# LIBS = -L${PRIMME_PATH} -lprimme
#---------------------------------------------------------------
# Linux 2.6.13-15.12-smp (64 bit image)
LIBS = -L${PRIMME_PATH} -lprimme -lm -L/usr/lib64/ -llapack -lblas -lgfortran -I../../lib/include/ -L../../lib/lib/ -llbfgs -I../../lib/share/
# LIBS = -L${PRIMME_PATH} -lprimme -lm -L/usr/lib64/ -llapack -lblas -lgfortran -lg2c
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

sdp_omp: $(PROB_OBJ) hybrid.o
	${LINKER} -o sdp_omp ${PROB_OBJ} hybrid.o  $(CFLAGS) ${LDFLAGS} ${LIBS}

hybrid.o: $(PROB_OBJ) hybrid.cpp
	$(CC) $(CFLAGS) -c -o hybrid.o hybrid.cpp 

maxcut.o: util.o
	$(CC) $(CFLAGS) ${DEFINES} -c maxcut.cpp -o maxcut.o

problem.o: problem.cpp util.o
	$(CC) $(CFLAGS) ${DEFINES} ${LIBS} -c problem.cpp -o problem.o

lp.o: util.o
	$(CC) $(CFLAGS) ${DEFINES} -c -o lp.o lp.cpp 
util.o:
	$(CC) ${CFLAGS} -c util.cpp -o util.o

clean:
	rm -f *.o *~ sdp_omp 

test:
	./sdp_omp -p 1 -e 1 -t 1 -i 10 -o 1 -y 10000 ../data/G1
G60:
	./sdp_omp -p 1 -e 1 -t 1 -i 10 -o 1 -y 10000 ../data/G60
G81:
	./sdp_omp -p 1 -e 1 -t 1 -i 10 -o 1 -y 10000 ../data/G81
