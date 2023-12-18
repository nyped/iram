#include "ra.h"
#include "arnoldi.h"
#include "tools.h"
#include <cblas.h>
#include <lapack.h>
#include <lapacke.h>
#include <math.h>
#include <stdlib.h>

// Compare function, used when sorting the eigenvalues
static int
cmp (const void *a, const void *b)
{
    eigen_infos *A = (eigen_infos *)a, *B = (eigen_infos *)b;
    double mod_a = A->im * A->im + A->re * A->re,
           mod_b = B->im * B->im + B->re * B->re;

    if (mod_a < mod_b)
        return 1;
    if (mod_a > mod_b)
        return -1;
    return 0;
}

void
ritz_arnoldi (const double *restrict A, double *restrict v, const size_t n,
              const size_t s, const size_t m, double *restrict err,
              double *restrict u, eigen_infos *restrict w, double *restrict _h,
              const size_t ldh, double *restrict _hh, double *restrict _ym,
              double *restrict _wi, double *restrict _wr, const size_t jj)
{
    // Call the Arnoldi Reduction
    arnoldi_mgs (A, v, _h, ldh, jj, n, m);

    // Compute the eigenvalues of the Hessenberg matrix H
    cblas_dcopy (ldh * m, _h, 1, _hh, 1);
    LAPACKE_dgeev (LAPACK_ROW_MAJOR, 'N', 'V', m, _hh, ldh, _wr, _wi, NULL, 1,
                   _ym, m);

    // Computing Um = Vm Ym
    cblas_dgemm (CblasRowMajor, CblasTrans, CblasNoTrans, m, n, m, 1.0, _ym, m,
                 v, n, 0.0, u, n);

    /*
     * Sorting the vectors.
     * The idea is to sort the eigenvalues and a list of indices,
     * so we do not move the eigenvectors.
     */
    const double h_err = _h[ldh * m + m - 1];
    for (size_t i = 0; i < m; i++)
        {
            w[i].re = _wr[i];
            w[i].im = _wi[i];
            w[i].index = i;
            w[i].err = fabs (h_err * _ym[m * (m - 1) + i]);
        }
    qsort (w, m, sizeof (*w), cmp);

    // Computing the errors on the s first eigenvalues
    double err_ = 0.0F;
    for (int i = 0; i < s; i++)
        err_ = MAX (err_, w[i].err);
    *err = err_;
}
