#include "iram.h"
#include "qr.h"
#include "ra.h"
#include "tools.h"
#include <cblas.h>
#include <math.h>
#include <stddef.h>

void
iram (const double *restrict A, double *restrict v0, const size_t n,
      const size_t s, const size_t m, const size_t iter_max, const double tol,
      eigen_infos *restrict w, double *restrict u)
{
    double err = 0;
    double *restrict _Q, *restrict _QQ;
    double *restrict _h, *restrict _hh;
    double *restrict _v, *restrict _vv;
    double *restrict _wi, *restrict _wr;
    double *restrict _ym;
    double *restrict _tau;
    double *restrict _fs, *restrict _fm;

    // Allocation
    ALLOC (_Q, m * m);
    ALLOC (_QQ, m * m);
    ALLOC (_fm, n);
    ALLOC (_fs, n);
    ALLOC (_h, m * (m + 1));
    ALLOC (_hh, m * (m + 1));
    ALLOC (_tau, m);
    ALLOC (_v, n * (m + 1));
    ALLOC (_vv, n * (m + 1));
    ALLOC (_wi, m);
    ALLOC (_wr, m);
    ALLOC (_ym, m * m);

    // Copy v0 into v
    cblas_dcopy (n, v0, 1, _vv, 1);

    // Print the header
    printf ("# %10s %10s\n", "nrc", "err");

    // Start timer
    const double t1 = my_timer ();

    // Call Ritz Arnoldi
    ritz_arnoldi (A, _vv, n, s, m, &err, w, _h, m, _hh, _ym, _wi, _wr, 0);

    for (size_t iter = 0; iter < iter_max; ++iter)
        {
            // Print the error
            printf ("%10zu % 10e\n", iter, err);

            // Check for "convergence"
            if (err < tol)
                break;

            // Call QR on h
            shifted_qr (_h, m, m, _Q, w, s, _QQ, _tau);

            /*
             * We have, with v(i) the columns of V:
             * V = V * Q
             * In our case, v(i) are the lines of V,
             * so we need to transform it into:
             * V = Q^t * V
             */
            cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, m, n, m, 1.0,
                         _Q, m, _vv, n, 0.0, _v, n);
            SWAP_PTR (_vv, _v);

            // beta = h(s + 1, s), sigma = Q(m, s)
            const double beta = _h[m * (s - 1) + s - 2];
            const double sigma = _Q[m * (m - 1) + s - 2];

            // fs = beta fs + sigma fm
            cblas_dcopy (n, _v + n * (m - 1), 1, _fm, 1);
            cblas_dcopy (n, _vv + n * (s - 1), 1, _fs, 1);
            cblas_dscal (n, beta, _fs, 1);
            cblas_daxpy (n, sigma, _fm, 1, _fs, 1);
            cblas_dcopy (n, _fs, 1, _vv + n * (s - 1), 1);

            // Call Ritz Arnoldi
            ritz_arnoldi (A, _vv, n, s, m, &err, w, _h, m, _hh, _ym, _wi, _wr,
                          s - 1);
        }

    // Compute the vectors
    translate_eigv (_vv, _ym, u, m, n);

    // End timer
    const double t2 = my_timer ();
    printf ("# time   : % 15.4lfs\n", t2 - t1);

    // Memory management
    FREE (_Q);
    FREE (_QQ);
    FREE (_fm);
    FREE (_fs);
    FREE (_h);
    FREE (_hh);
    FREE (_tau);
    FREE (_v);
    FREE (_vv);
    FREE (_wi);
    FREE (_wr);
    FREE (_ym);
}
