#pragma once

#include <cblas.h>

// Double general matrix vector
void dgemv (CBLAS_LAYOUT layout, CBLAS_TRANSPOSE TransA, const CBLAS_INT M,
            const CBLAS_INT N, const double alpha, const double *A,
            const CBLAS_INT lda, const double *X, const CBLAS_INT incX,
            const double beta, double *Y, const CBLAS_INT incY);
