#include "tools.h"
#include <bits/time.h>
#include <math.h>
#include <time.h>

void
fill (double *restrict v, const double a, const size_t n)
{
    double cst = a;

    cblas_dcopy (n, &cst, 0, v, 1);
}

double *
read_mtz (const char *restrict filename, size_t *restrict n, const size_t sym)
{
    FILE *f = fopen (filename, "r");
    double *mat = NULL;
    size_t nb_entries;
    *n = 0;

#define BUFF_SIZE 1024
    char line[BUFF_SIZE];

    //
    if (!f)
        {
            perror ("fopen");
            exit (EXIT_FAILURE);
        }

    // Getting the infos of the matrix
    while (fgets (line, sizeof (line), f) != NULL)
        {
            size_t tmp;

            // Skip comments
            if (line[0] == '%')
                continue;

            // Initializing the dimensions
            if (3 != sscanf (line, " %lu %lu %lu", n, &tmp, &nb_entries))
                {
                    fprintf (stderr, "Missing header line. Aborting.\n");
                    exit (EXIT_FAILURE);
                }

            // Dimension check
            if (tmp != *n)
                {
                    fprintf (stderr, "Non square matrix. Aborting.\n");
                    exit (EXIT_FAILURE);
                }
            break;
        }

    // Matrix allocation
    ALLOC (mat, *n * *n);

    // Populating the matrix
    for (size_t _ = 0; _ < nb_entries; ++_)
        {
            size_t i, j;
            double val;

            if (fgets (line, sizeof (line), f) == NULL)
                {
                    fprintf (
                        stderr,
                        "Not enough entries in the mtz file. Aborting.\n");
                    exit (EXIT_FAILURE);
                }

            if (3 != sscanf (line, " %lu %lu %lf", &i, &j, &val))
                {
                    fprintf (stderr, "Ill formed entry %lu. Aborting.\n", _);
                    exit (EXIT_FAILURE);
                }
            mat[(i - 1) * *n + j - 1] = val;
            if (sym)
                mat[(j - 1) * *n + i - 1] = val;
        }

    //
    fclose (f);

    //
    return mat;
}

static void
plant_seed (void)
{
    static size_t rand_initialized = 0;

    if (rand_initialized)
        return;

    rand_initialized = 1;

    // srand (time (NULL));
    srand (0);
}

void
gen_vect (double *restrict x, const size_t n, const size_t dx)
{
    plant_seed ();

    for (size_t i = 0, j = 0; i < n; i++, j += dx)
        x[j] = rand () % RANDMAX;
}

void
gen_base (double *restrict x, const size_t n, const bool transpose)
{
    const size_t dx = transpose ? n : 1;
    const size_t dy = transpose ? 1 : n;

    plant_seed ();

    // First line
    gen_vect (x, n, dx);

    // Filling the other lines
    for (size_t i = 1; i < n; ++i)
        {
            // copy the previous line
            cblas_dcopy (n, x + (i - 1) * dy, dx, x + i * dy, dx);

            // adding the identity matrix
            *(x + i * dy + (i - 1) * dx) += 1.0;
        }
}

void
print_mat (const double *restrict v, const size_t n, const size_t m,
           const size_t ldv, const char *restrict desc)
{
#if DEBUG
    if (desc)
        printf ("%s:\n", desc);

    for (size_t i = 0; i < n; ++i)
        {
            for (size_t j = 0; j < m; ++j)
                printf ("% 6.2e ", v[i * ldv + j]);
            printf ("\n");
        }

    printf ("\n");
#endif /* DEBUG */
}

// Inspired by:
// https://www.intel.com/content/www/us/en/docs/onemkl/code-samples-lapack/
void
print_eigs (const double *restrict v, const eigen_infos *restrict w,
            const size_t n, const size_t s, const char *restrict desc)
{
#if DEBUG
    printf ("%s\n", desc);
    for (size_t i = 0; i < s;)
        {
            size_t k = w[i].index;

            if (w[i].im != 0.0F)
                {
#define PRINT_COMPLEX_EIG(SIGN)                                               \
    do                                                                        \
        {                                                                     \
            printf ("(% 6.2f %+6.2fj): ", w[i].re, SIGN w[i].im);             \
            for (size_t j = 0; j < n; ++j)                                    \
                printf (" (% 6.2f %+6.2fj)", v[k * n + j],                    \
                        SIGN v[(k + 1) * n + j]);                             \
        }                                                                     \
    while (0)

                    PRINT_COMPLEX_EIG (+);
                    printf ("\n");
                    PRINT_COMPLEX_EIG (-);

                    i += 2;
                }
            else
                {
                    printf (" % 6.2f         : ", w[i].re);
                    for (size_t j = 0; j < n; ++j)
                        printf (" % 6.2f", v[k * n + j]);

                    i++;
                }
            printf ("\n");
        }
    printf ("\n");
#endif
}

double
residual_norm (const double *restrict a, const double *restrict v,
               const eigen_infos *restrict w, const size_t n, const size_t s)
{
    double max_res = 0.0F;
    /*
     * lre_vre meaning lambda_real * real(v_k)
     * The same goes for the other variables.
     */
    double *restrict lre_vre = NULL, *restrict lim_vre = NULL;
    double *restrict lim_vim = NULL, *restrict lre_vim = NULL;
    double *restrict lv_re = NULL, *restrict lv_im = NULL;

    /*
     * A_vre meaning A * real(v_k)
     * The same goes for the A_vim.
     */
    double *restrict A_vre = NULL, *restrict A_vim = NULL;

    //
    ALLOC (lre_vre, n);
    ALLOC (A_vre, n);
    ALLOC (lv_re, n);
    ALLOC (lv_im, n);
    ALLOC (A_vim, n);
    ALLOC (lim_vre, n);
    ALLOC (lim_vim, n);
    ALLOC (lre_vim, n);

    /*
     * NOTE: this is not the most optimized way to compute the residual.
     * But we call it only once, so it is not a big deal. Who cares.
     */

    // Iterating over the eigenvalues
    for (size_t i = 0; i < s;)
        {
            const double lambda_re = w[i].re;
            const size_t k = w[i].index;

            /*
             * Complex eigenvalue
             * We have a complex eigenvalue and eigenvector.
             * On one hand we need to compute:
             * - A * v = A * v_re + i * A * v_im
             * On the other hand we need to compute:
             * - lambda * v = (lambda_re + i * lambda_im) * (v_re + i * v_im)
             *              = lre * v_re - lim * v_im + i * (lre * v_im + lim *
             * v_re)
             */
            if (w[i].im != 0.0F)
                {
                    const double lambda_im = w[i].im;
                    const size_t j = w[i + 1].index;

                    cblas_dcopy (n, v + k * n, 1, lre_vre, 1);
                    cblas_dcopy (n, v + k * n, 1, lim_vre, 1);
                    cblas_dcopy (n, v + j * n, 1, lre_vim, 1);
                    cblas_dcopy (n, v + j * n, 1, lim_vim, 1);

                    cblas_dscal (n, lambda_re, lre_vre, 1);
                    cblas_dscal (n, lambda_re, lre_vim, 1);
                    cblas_dscal (n, lambda_im, lim_vre, 1);
                    cblas_dscal (n, lambda_im, lim_vim, 1);

                    // lre_vre - lim_vim in lv_re
                    cblas_dcopy (n, lre_vre, 1, lv_re, 1);
                    cblas_daxpy (n, -1.0, lim_vim, 1, lv_re, 1);

                    // lre_vim + lim_vre in lv_im
                    cblas_dcopy (n, lre_vim, 1, lv_im, 1);
                    cblas_daxpy (n, 1.0, lim_vre, 1, lv_im, 1);

                    // Compute A * v_k for real and imaginary parts
                    cblas_dgemv (CblasRowMajor, CblasNoTrans, n, n, 1.0, a, n,
                                 v + k * n, 1, 0.0, A_vre, 1);
                    cblas_dgemv (CblasRowMajor, CblasNoTrans, n, n, 1.0, a, n,
                                 v + j * n, 1, 0.0, A_vim, 1);

                    // Compute the difference for real and imaginary
                    cblas_daxpy (n, -1.0, A_vre, 1, lv_re, 1);
                    cblas_daxpy (n, -1.0, A_vim, 1, lv_im, 1);

                    // L2 norm of the difference
                    const double res_real = cblas_ddot (n, lv_re, 1, lv_re, 1);
                    const double res_imag = cblas_ddot (n, lv_im, 1, lv_im, 1);
                    const double res = sqrt (res_real + res_imag);

                    // Update the maximum residual
                    max_res = MAX (max_res, res);

                    // Skip the next eigenvalue, since it is the conjugate
                    i += 2;
                    continue;
                }
            /*
             * Real eigenvalue.
             * Straightforward computation of the residual.
             */
            else
                {
                    // Computing lambda * v_k
                    cblas_dcopy (n, v + k * n, 1, lre_vre, 1);
                    cblas_dscal (n, lambda_re, lre_vre, 1);

                    // Computing A * v_k
                    cblas_dgemv (CblasRowMajor, CblasNoTrans, n, n, 1, a, n,
                                 v + k * n, 1, 0.0, A_vre, 1);

                    // Computing the difference
                    cblas_daxpy (n, -1.0, A_vre, 1, lre_vre, 1);

                    // Computing the norm
                    const double res = cblas_dnrm2 (n, lre_vre, 1);

                    // Updating the maximum
                    max_res = MAX (max_res, res);

                    //
                    i++;
                }
        }

    //
    FREE (lre_vre);
    FREE (A_vre);
    FREE (lv_re);
    FREE (lv_im);
    FREE (A_vim);
    FREE (lim_vre);
    FREE (lim_vim);
    FREE (lre_vim);

    //
    return max_res;
}

double
my_timer (void)
{
    struct timespec t;

    clock_gettime (CLOCK_MONOTONIC_RAW, &t);

    return t.tv_sec + 1E-9L * t.tv_nsec;
}
