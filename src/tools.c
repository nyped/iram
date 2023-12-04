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
read_matrix (const char *filename, size_t *restrict n)
{
    FILE *f = fopen (filename, "r");
    double *mat;

    //
    if (!f)
        {
            perror ("fopen");
            exit (EXIT_FAILURE);
        }

    // Reading the dimension of the matrix
    fscanf (f, " %lu", n);
    ALLOC (mat, *n * *n);

    for (size_t i = 0; i < *n * *n; ++i)
        {
            if (fscanf (f, "%lf", mat + i) != 1)
                {
                    fprintf (stderr, "Error: could not read matrix\n");
                    exit (EXIT_FAILURE);
                }
        }

    fclose (f);

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
gen_vect (double *x, const size_t n, const size_t dx)
{
    plant_seed ();

    for (size_t i = 0, j = 0; i < n; i++, j += dx)
        x[j] = rand () % RANDMAX;
}

void
gen_base (double *x, const size_t n, const bool transpose)
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
// TODO: print the s best one
void
print_eigs (const double *v, const eigen_infos *w, const size_t n,
            const size_t s, const char *desc)
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
mean (const double *a, const size_t n)
{
    double m = 0.0;

    for (size_t i = 0; i < n; i++)
        m += a[i];

    m /= (double)n;

    return m;
}

double
stddev (const double *restrict a, const double mean, const size_t n)
{
    double d = 0.0;

    for (size_t i = 0; i < n; i++)
        d += (a[i] - mean) * (a[i] - mean);

    d /= (double)n;

    return sqrt (d);
}

void
orthogonality_mat (const double *restrict a, double *restrict o,
                   const size_t n)
{
    /*
     * Computing the dot product between each pair of vector.
     * This is equivalent to computing the matrix product
     * between the matrix and its transpose.
     */
    cblas_dgemm (CblasRowMajor, CblasNoTrans, CblasTrans, n, n, n, 1, a, n, a,
                 n, 0, o, n);
}

double
my_timer (void)
{
    struct timespec t;

    clock_gettime (CLOCK_MONOTONIC_RAW, &t);

    return t.tv_sec + 1E-9L * t.tv_nsec;
}
