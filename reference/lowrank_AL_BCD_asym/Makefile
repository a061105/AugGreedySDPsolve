
all:
	g++ -O3 -fopenmp -o LowrankALBCD LowrankALBCD.cpp

so: 
	g++ -shared -fopenmp -Wl,-soname,LowrankALBCD -o LowrankALBCD.so -fPIC LowrankALBCD.cpp

clean:
	rm LowrankALBCD LowrankALBCD.so
