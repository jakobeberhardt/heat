CC     =  clang
OPT3 	= -O3
CFLAGS  = -I/opt/homebrew/opt/libomp/include -Xclang -fopenmp -Wall -std=c99
LFLAGS  = -L/opt/homebrew/opt/libomp/lib -lomp

ALL	= heat-omp 
all: $(ALL)

misc.o: misc.c
	$(CC) -c $(CFLAGS) $(OPT3) $< -o $@

heat-omp: heat-omp.c solver-omp.c misc.o
	$(CC) $(CFLAGS) $(OPT3) $+ $(LFLAGS) -o $@

clean:
	rm -fr $(ALL) *.o 

ultraclean:
	rm -fr $(ALL) *.o *.ppm sub*.o* sub*.e* *.prv *.pcf *.row time-heat-omp*boada*.txt

test_seidel_omp: 
	@bash run_test.sh seidel_omp

test_jacobi_omp: heat-omp 
	@bash run_test.sh jacobi_omp

test_jacobi_omp_512: heat-omp 
	@bash run_test.sh jacobi_omp_512

test: clean all test_seidel_omp test_jacobi_omp test_jacobi_omp_512