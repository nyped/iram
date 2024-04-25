# IRAM

IRAM (Implicitly restarted Arnoldi method), an iterative eigenvalue solver.

## Dependencies

- A `C` compiler with support for `OpenMP`
- `CBLAS` and `LAPACKE`
- `CMake`

## Compilation

```bash
cmake -B build src
cmake --build build
```

## Execution

```bash
# help message
./build/iram

# Setting the CPU infos
export OMP_NUM_THREADS=16 OMP_PROC_BIND=spread

# calling iram on:
# - the file af23560.mtx
# - a projection subspace of dimension 30
# - 2 target eigenvalues
# - 500 iterations at most
# - 1e-6 error threshold
# - 0 (non symmetric matrix)
./build/iram assets/matrix/unsym/af23560.mtx 30 1 500 1e-6 0
```
