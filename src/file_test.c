#include "iram.h"
#include "miram.h"
#include "tools.h"
#include <omp.h>
#include <stdio.h>

static inline void
usage (const char *restrict bin)
{
    fprintf (stderr,
             "usage: %s n m s iter_max algo tol solver [delta_m]\n\n"
             "with:\n"
             "- file     the path to a mtz matrix\n"
             "- m        the projection space dimension\n"
             "           If the solver is miram, this will be"
             "           the dimension of the smaller mi\n"
             "- s        the number of eigenvalues to estimate\n"
             "- iter_max the maximum number of iterations\n"
             "- tol      the solution tolerance\n"
             "- sym      boolean that tells if the matrix is symmetric\n"
             "- solver   the solver to use: iram or miram\n"
             "- delta_m  the offset between each mi when iram is used.\n"
             "           This is set to 1 by default.\n",
             bin);
}

int
main (int argc, char *argv[])
{
    size_t n, m, s, iter_max, delta_m;
    double *restrict A, *restrict v0, *restrict u, tol;
    eigen_infos *restrict w;
    size_t symmetric;
    const size_t nb_threads = omp_get_max_threads ();
    char *restrict solver;
    int ret = 0;

    //
    if (argc != 8 && argc != 9)
        return usage (*argv), 255;

    //
    symmetric = atol (argv[6]);
    A = read_mtz (argv[1], &n, symmetric);
    s = atol (argv[3]);
    m = atol (argv[2]);
    iter_max = atol (argv[4]);
    solver = argv[7];
    if (!sscanf (argv[5], " %lf", &tol) || !A || !s || !m || !iter_max)
        {
            fprintf (stderr, "Error: wrong argument values\n");
            return 254;
        }
    if (argc == 9)
        delta_m = atol (argv[8]);
    else
        delta_m = 1;

    //
    if (m > n || s > m)
        return fprintf (stderr, "s <= m <= n\n"), 253;

    // FIXME: Implementation quirk
    if (m < 2)
        return fprintf (stderr, "m >= 2\n"), 253;

    //
    ALLOC (u, (m + nb_threads * delta_m) * n);
    ALLOC (v0, n);
    ALLOC (w, m);

    // Generate vector for v0
    fill (v0, 1.0, n);

    // Calling the solver
    if (!strcmp (solver, "iram"))
        {
            iram (A, v0, n, s, m, iter_max, tol, w, u);
        }
    else if (!strcmp (solver, "miram"))
        {
            miram (A, v0, n, s, m, delta_m, iter_max, tol, w, u);
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
    FREE (v0);
    FREE (w);

    //
    return ret;
}
