#include "iram.h"
#include "tools.h"
#include <stdio.h>

static inline void
usage (const char *restrict bin)
{
    fprintf (stderr,
             "usage: %s file m s iter_max tol\n\n"
             "with:\n"
             "- file     the path to a mtz matrix\n"
             "- m        the projection space dimension\n"
             "- s        the number of eigenvalues to estimate\n"
             "- iter_max the maximum number of iterations\n"
             "- tol      the solution tolerance\n",
             bin);
}

int
main (int argc, char *argv[])
{
    size_t n, m, s, iter_max;
    double *restrict A, *restrict v, *restrict u, tol;
    eigen_infos *restrict w;

    //
    if (argc != 6)
        return usage (*argv), 255;

    //
    A = read_mtz (argv[1], &n);
    s = atol (argv[3]);
    m = atol (argv[2]);
    iter_max = atol (argv[4]);
    if (!sscanf (argv[5], " %lf", &tol) || !A || !s || !m || !iter_max)
        {
            fprintf (stderr, "Error: wrong argument values\n");
            return 254;
        }

    //
    if (m > n || s > m)
        return fprintf (stderr, "s <= m <= n\n"), 253;

    //
    ALLOC (u, m * n);
    ALLOC (v, n * (m + 1));
    ALLOC (w, m);

    // Generate vector for v
    fill (v, 1.0, n);

    // Calling iram
    iram (A, &v, n, s, m, iter_max, tol, w, u);

    // Print the eigenvalues
    for (size_t i = 0; i < s; ++i)
        printf ("# eig[%zu] : % 15.6lf + % 15.6lf j\n", i, w[i].re, w[i].im);

    // Compute the residual norm
    printf ("# res    : % 15e\n", residual_norm (A, u, w, n, s));

    //
    FREE (A);
    FREE (u);
    FREE (v);
    FREE (w);

    //
    return 0;
}
