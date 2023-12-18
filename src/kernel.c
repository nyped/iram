#include "kernel.h"

void
dgemv (CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
       const CBLAS_INT N, const double alpha, const double *A,
       const CBLAS_INT lda, const double *X, const CBLAS_INT incX,
       const double beta, double *Y, const CBLAS_INT incY)
{
#pragma omp parallel for
    for (size_t i = 0; i < N; i++)
        {
            double sum = 0.0;
            for (size_t j = 0; j < N; j++)
                {
                    sum += A[i * N + j] * X[j];
                }
            Y[i] = sum;
        }
}
