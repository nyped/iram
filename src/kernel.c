#include "kernel.h"

void
dgemv (CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const int32_t M,
       const int32_t N, const double alpha, const double *A, const int32_t lda,
       const double *X, const int32_t incX, const double beta, double *Y,
       const int32_t incY)
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
