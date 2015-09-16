# include the environment, compilation, and linking flags from PRIMME
 include ../../Make_flags
 include ../../Link_flags

# Makefile for ParaSails
# Please modify for your local system.
# If you are using ESSL instead of LAPACK/BLAS, please define ESSL in Common.h

.SUFFIXES: .c .o

C_COMPILE_PSAILS_FLAGS= -DPARASAILS_DEBUG -DPARASAILS_CG_PRINT
 
# need to link to LAPACK and BLAS
# linking fortran versions needs fortran libs

PSAILS_LFLAGS =\
 -L. -lParaSails\
 $(LIBDIRS) $(INCLUDE) $(LIBS) $(LDFLAGS)

HEADERS =\
 Common.h\
 ConjGrad.h\
 DiagScale.h\
 Hash.h\
 LoadBal.h\
 Matrix.h\
 Mem.h\
 Numbering.h\
 OrderStat.h\
 ParaSails.h\
 PrunedRows.h\
 RowPatt.h\
 StoredRows.h

FILES =\
 ConjGrad.c\
 DiagScale.c\
 FlexGmres.c\
 Hash.c\
 LoadBal.c\
 Matrix.c\
 Mem.c\
 Numbering.c\
 OrderStat.c\
 ParaSails.c\
 PrunedRows.c\
 RowPatt.c\
 StoredRows.c

OBJS = ${FILES:.c=.o}

##################################################################
# Targets
##################################################################

all: libParaSails.a 

driver: driver.o libParaSails.a
	@echo  "Linking" $@ "... "
	${CC} -o $@ driver.o ${PSAILS_LFLAGS}

clean:
	@rm -f *.o libParaSails.a

##################################################################
# Rules
##################################################################

libParaSails.a: ${OBJS}
	@echo  "Building $@ ... "
	ar -rcu $@ ${OBJS}
	${RANLIB} $@

${OBJS}: ${HEADERS}

##################################################################
# Generic rules
##################################################################

.c.o:
	${CC} -o $@ -c ${CFLAGS} ${C_COMPILE_PSAILS_FLAGS} $<
