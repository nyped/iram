#include "arnoldi.h"
#include <cblas.h>
#include <stdio.h>

// For readability purposes
#define V(k) (v + n * (k))
#define H(i, j) (h + ldh * (i) + (j))

void
arnoldi_mgs (const double *restrict A, double *restrict v, double *restrict h,
             const size_t ldh, const size_t jj, const size_t n, const size_t m)
{
    // v(jj) = v(jj) / ||v(jj)||
    const double norm = cblas_dnrm2 (n, V (jj), 1);
    cblas_dscal (n, 1.0 / norm, V (jj), 1);
    cblas_dcopy (n, V (jj), 1, V (jj), 1);

    if (jj)
        {
            *H (jj, jj - 1) = norm;
        }

    for (size_t j = jj; j < m; ++j)
        {
            // v(j + 1) = A * v(j)
            cblas_dgemv (CblasRowMajor, CblasNoTrans, n, n, 1.0, A, n, V (j),
                         1, 0.0, V (j + 1), 1);

            for (size_t i = 0; i <= j; ++i)
                {
                    // h(i,j) = <v(j + 1), v(i)>
                    const double r = cblas_ddot (n, V (j + 1), 1, V (i), 1);
                    *H (i, j) = r;

                    // v(j + 1) = v(j + 1) - h(i,j) * v(i)
                    cblas_daxpy (n, -r, V (i), 1, V (j + 1), 1);
                }

            // H(j + 1, j) = norm
            const double norm = cblas_dnrm2 (n, V (j + 1), 1);
            *H (j + 1, j) = norm;

            // v(j) = v(j) / || v(j) ||
            cblas_dscal (n, 1.0 / norm, V (j + 1), 1);

            /*
             * No invariant space check for faster execution.
             * In practice this does not happen.
             */
#if 0
            if (norm < 1E-16L)
                {
                    printf ("Espace invariant à l'étape %lu\n", j);
                    return;
                }
#endif
        }
}
