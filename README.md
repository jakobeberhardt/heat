# Solves the heat equation using several parallel programming models
As reference, `test` contains diff files created from sequential execution.
## Part 1 (`OMP`)
Make sure to run 
	`export OMP_NUM_THREADS=4`
### Running tests
```bash 
make test_jacobi
```
```bash 
# Runs jacobi on a higher resolution 
make test_jacobi_512
```
```bash 
make test_seidel
```
```bash 
# Runs all test
make test
```
