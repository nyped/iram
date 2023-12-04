#include "iram.h"
#include "qr.h"
#include "ra.h"
#include "tools.h"
#include <cblas.h>
#include <math.h>
#include <stddef.h>

void
iram (const double *restrict A, double *restrict v, const size_t n,
      const size_t s, const size_t m, const size_t iter_max, const double tol,
      eigen_infos *restrict w, double *restrict u)
{
    double err = 0;
    double *restrict _h, *restrict _ym, *restrict _wi, *restrict _wr,
                                                           *restrict _Q,
                                                               *restrict _v;

    // Allocation
    ALLOC (_h, m * (m + 1));
    ALLOC (_ym, m * m);
    ALLOC (_wi, m);
    ALLOC (_wr, m);
    ALLOC (_Q, m * m);
    ALLOC (_v, n * (m + 1));

    fill (_h, 0, m * (m + 1));

    // Print the header
    printf ("# %10s %10s\n", "iter", "err");

    // Call Ritz Arnoldi
    ritz_arnoldi (A, v, n, s, m, &err, u, w, _h, _ym, _wi, _wr, 0);

    double fem = 1.0F;

    size_t iter;
    for (iter = 0; iter < iter_max; ++iter)
        {
            // Print the error
            printf ("%10zu % 10e\n", iter, err);

            /* for (size_t i = 0; i < s; ++i)
                printf ("eig[%zu]  = %g; ", i, w[i].re);
            printf ("\n"); */

            // Check for "convergence"
            if (err < tol)
                break;

            // Call QR on h
            _Q = shifted_qr (_h, m, _Q, w, s);
            fem = _Q[m * m - 1];

            /*
             * We have, with v(i) the columns of V:
             * V = V * Q
             * In our case, v(i) are the lines of V,
             * so we need to transform it into:
             * V = Q^t * V
             */
            cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, m, n, m, 1.0,
                         _Q, m, v, n, 0.0, _v, n);
            SWAP_PTR (v, _v);

            // Call Ritz Arnoldi
            ritz_arnoldi (A, v, n, s, m, &err, u, w, _h, _ym, _wi, _wr, s - 1);

            // Update the error
            err = fabs (err * fem);
        }

    // TODO, FIXME: write it properly
    // Copy v into v_
    if (iter % 2 == 1)
        {
            cblas_dcopy (n * (m + 1), v, 1, _v, 1);

            SWAP_PTR (v, _v);
        }

#if 1
    // Print the eigenvalues
    for (size_t i = 0; i < s; ++i)
        printf ("eig[%zu]  = %g\n", i, w[i].re);
#endif

    // Memory management
    FREE (_h);
    FREE (_ym);
    FREE (_wi);
    FREE (_wr);
    FREE (_Q);
    FREE (_v);
}
