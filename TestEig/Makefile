CC = gcc
PRIMME_PATH = /Users/teddy/Documents/Work/GITHUB/AugGreedySDPsolve/PRIMME_1.2
LINKER = ${CC}
CFLAGS = -O2 -Wall -fPIC -I${PRIMME_PATH}/PRIMMESRC/COMMONSRC

OBJS = testEig.o 

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

all: testEig

testEig: ${OBJS}
	${LINKER} -o testEig ${OBJS} ${LDFLAGS} ${LIBS}

%.o: %.c
	${CC} -c $< ${CFLAGS} ${DEFINES}

clean:
	rm -f *.o

veryclean:
	rm -f testEig *.o core
