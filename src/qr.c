#include "qr.h"
#include "arnoldi.h"
#include "tools.h"
#include <lapacke.h>
#include <stdio.h>

double *
shifted_qr (double *restrict h, const size_t m, double *restrict Q,
            const eigen_infos *restrict w, const size_t s)
{
    //
    double *restrict _tau;
    double *restrict _Q, *restrict Q_TMP;

    //
    ALLOC (_tau, m * m);
    ALLOC (_Q, m * m);
    ALLOC (Q_TMP, m * m);

    // Fill Q with identity
    double zero = 0.0F;
    cblas_dcopy (m * m, &zero, 0, Q, 1);
    for (size_t i = 0; i < m; ++i)
        Q[i * m + i] = 1;

    // print_mat (h, m, m, m, "h in");
    for (size_t i = s; i < m; ++i)
        {
            // Copy h to _Q
            cblas_dcopy (m * m, h, 1, _Q, 1);

            // _Q = h - shift * I
            for (size_t j = 0; j < m; ++j)
                _Q[j * m + j] -= w[i].re;

            // QR factorization on _Q = h - I lambda_i
            LAPACKE_dgeqrf (LAPACK_ROW_MAJOR, m, m, _Q, m, _tau);
            LAPACKE_dorgqr (LAPACK_ROW_MAJOR, m, m, m, _Q, m, _tau);
            // print_mat (_Q, m, m, m, "Q");

            // h = Q^T * h * Q
            cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, m, m, m, 1.0,
                         _Q, m, h, m, 0.0, _tau, m);
            cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m,
                         1.0, _tau, m, _Q, m, 0.0, h, m);

            // Q = Q * _Q
            cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasNoTrans, m, m, m,
                         1.0, Q, m, _Q, m, 0.0, Q_TMP, m);
            SWAP_PTR (Q, Q_TMP);
        }
    // print_mat (h, m, m, m, "h out");

    //
    FREE (_tau);
    FREE (_Q);
    FREE (Q_TMP);

    return Q;
}
