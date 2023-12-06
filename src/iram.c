#include "iram.h"
#include "qr.h"
#include "ra.h"
#include "tools.h"
#include <cblas.h>
#include <math.h>
#include <stddef.h>

void
iram (const double *restrict A, double *restrict *restrict v, const size_t n,
      const size_t s, const size_t m, const size_t iter_max, const double tol,
      eigen_infos *restrict w, double *restrict u)
{
    double err = 0;
    double *restrict _Q, *restrict _QQ;
    double *restrict _h, *restrict _hh;
    double *restrict _v;
    double *restrict _wi, *restrict _wr;
    double *restrict _ym;
    double *restrict _tau;

    // Allocation
    ALLOC (_Q, m * m);
    ALLOC (_QQ, m * m);
    ALLOC (_h, m * (m + 1));
    ALLOC (_hh, m * (m + 1));
    ALLOC (_tau, m);
    ALLOC (_v, n * (m + 1));
    ALLOC (_wi, m);
    ALLOC (_wr, m);
    ALLOC (_ym, m * m);

    // Print the header
    printf ("# %10s %10s\n", "iter", "err");

    // Call Ritz Arnoldi
    ritz_arnoldi (A, *v, n, s, m, &err, u, w, _h, _hh, _ym, _wi, _wr, 0);

    for (size_t iter = 1; iter <= iter_max; ++iter)
        {
            // Print the error
            printf ("%10zu % 10e\n", iter, err);

            // Check for "convergence"
            if (err < tol)
                break;

            // Call QR on h
            shifted_qr (_h, m, _Q, w, s, _QQ, _tau);

            /*
             * We have, with v(i) the columns of V:
             * V = V * Q
             * In our case, v(i) are the lines of V,
             * so we need to transform it into:
             * V = Q^t * V
             */
            cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, m, n, m, 1.0,
                         _Q, m, *v, n, 0.0, _v, n);
            SWAP_PTR (*v, _v);

            // Call Ritz Arnoldi
            ritz_arnoldi (A, *v, n, s, m, &err, u, w, _h, _hh, _ym, _wi, _wr,
                          s - 1);
        }

    // Memory management
    FREE (_Q);
    FREE (_QQ);
    FREE (_h);
    FREE (_hh);
    FREE (_tau);
    FREE (_v);
    FREE (_wi);
    FREE (_wr);
    FREE (_ym);
}
