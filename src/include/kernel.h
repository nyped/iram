#pragma once

#include <cblas.h>
#include <stdint.h>

// Double general matrix vector
void dgemv (CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const int32_t M,
            const int32_t N, const double alpha, const double *A,
            const int32_t lda, const double *X, const int32_t incX,
            const double beta, double *Y, const int32_t incY);
