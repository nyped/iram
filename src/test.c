#include "iram.h"
#include "miram.h"
#include "tools.h"
#include <omp.h>
#include <string.h>

static inline void
load_mat_diag (size_t n, double *restrict A)
{
    for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                A[i * n + j] = 0;
            A[i * n + i] = i + 1;
        }
}

static inline void
load_mat_a (size_t n, double *restrict A)
{

    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = 0;

    double a = 3;
    double b = 6;

    for (int i = 1; i < n - 1; i++)
        for (int j = 1; j < n - 1; j++)
            {
                A[i * n + i - 1] = b;
                A[i * n + i] = a;
                A[i * n + i + 1] = b;
            }

    A[0] = a;
    A[1] = b;
    A[n * n - 2] = b;
    A[n * n - 1] = a;
}

static inline void
usage (const char *restrict bin)
{
    fprintf (stderr,
             "usage: %s n m s iter_max algo tol solver\n\n"
             "with:\n"
             "- n        the target matrix dimension\n"
             "- m        the projection space dimension\n"
             "- s        the number of eigenvalues to estimate\n"
             "- iter_max the maximum number of iterations\n"
             "- algo     the matrix generation algo in: a, random, or diag\n"
             "- tol      the solution tolerance\n"
             "- solver   the solver to use: iram or miram\n",
             bin);
}

int
main (int argc, char *argv[])
{
    size_t n, m, s, iter_max;
    double *restrict A, *restrict v, *restrict v0, *restrict u, tol;
    eigen_infos *restrict w;
    char *restrict algo;
    char *restrict solver;
    const size_t nb_threads = omp_get_max_threads ();
    int ret = 0;

    //
    if (argc != 8)
        return usage (*argv), 255;

    //
    n = atol (argv[1]);
    m = atol (argv[2]);
    s = atol (argv[3]);
    iter_max = atol (argv[4]);
    algo = argv[5];
    solver = argv[7];
    if (!sscanf (argv[6], " %lf", &tol) || !n || !s || !m || !iter_max)
        {
            fprintf (stderr, "Error: wrong argument values\n");
            return 254;
        }

    //
    if (m > n || s > m)
        return fprintf (stderr, "s <= m <= n\n"), 253;

    // FIXME: Implementation quirk
    if (m < 2)
        return fprintf (stderr, "m >= 2\n"), 253;

    // Generating the matrix A
    ALLOC (A, n * n);
    if (!strcmp (algo, "a"))
        {
            load_mat_a (n, A);
        }
    else if (!strcmp (algo, "random"))
        {
            gen_vect (A, n * n, 1);
        }
    else if (!strcmp (algo, "diag"))
        {
            load_mat_diag (n, A);
        }
    else
        {
            FREE (A);
            return usage (*argv), 252;
        }

    //
    ALLOC (u, (m + nb_threads) * n);
    ALLOC (v, n * (m + 1));
    ALLOC (v0, n);
    ALLOC (w, m);

    // Generate vector for v
    fill (v0, 1.0, n);

    // Calling the solver
    if (!strcmp (solver, "iram"))
        {
            iram (A, &v, v0, n, s, m, iter_max, tol, w, u);
        }
    else if (!strcmp (solver, "miram"))
        {
            miram (A, v, v0, n, s, m, iter_max, tol, w, u);
        }
    else
        {
            usage (*argv);
            ret = 252;
            goto clean_up;
        }

    // Print the eigenvalues
    for (size_t i = 0; i < s; ++i)
        printf ("# eig[%zu] : % 15.6lf + % 15.6lf j\n", i, w[i].re, w[i].im);

    // Compute the residual norm
    printf ("# res    : % 15e\n", residual_norm (A, u, w, n, s));

    //
clean_up:
    FREE (A);
    FREE (u);
    FREE (v);
    FREE (v0);
    FREE (w);

    //
    return ret;
}
