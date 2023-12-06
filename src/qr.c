#include "qr.h"
#include "arnoldi.h"
#include "tools.h"
#include <lapacke.h>
#include <math.h>
#include <stdio.h>

void
shifted_qr (double *restrict h, const size_t m, double *restrict Q,
            const eigen_infos *restrict w, const size_t s, double *restrict _Q,
            double *restrict _tau)
{
    // Fill Q with identity
    fill (Q, 0, m * m);
    for (size_t i = 0; i < m; ++i)
        Q[i * m + i] = 1;

    for (size_t i = s; i < m; ++i)
        {
            // _Q = h - shift * I
            cblas_dcopy (m * m, h, 1, _Q, 1);
            for (size_t j = 0; j < m; ++j)
                _Q[j * m + j] -= w[i].re;

            // QR factorization on _Q = h - I lambda_i
            LAPACKE_dgeqrf (LAPACK_ROW_MAJOR, m, m, _Q, m, _tau);

            // h = Q^T * h * Q
            LAPACKE_dormqr (LAPACK_ROW_MAJOR, 'L', 'T', m, m, m, _Q, m, _tau,
                            h, m);
            LAPACKE_dormqr (LAPACK_ROW_MAJOR, 'R', 'N', m, m, m, _Q, m, _tau,
                            h, m);

            // Q = Q * _Q
            LAPACKE_dormqr (LAPACK_ROW_MAJOR, 'R', 'N', m, m, m, _Q, m, _tau,
                            Q, m);

            /*
             * Make h upper Hessenberg.
             * We need this because of numerical errors.
             */
            for (size_t i = 1; i < m; ++i)
                for (size_t j = 0; j < i - 1; ++j)
                    h[i * m + j] = 0.0F;
        }
}
