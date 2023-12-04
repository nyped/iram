#include "iram.h"
#include "tools.h"
#include <string.h>

void
load_mat_diag (size_t n, double *A)
{
    for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
                A[i * n + j] = 0;
            A[i * n + i] = i + 1;
        }
}

void
load_mat_tri (size_t n, double *A)
{
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            A[i * n + j] = 0;

    for (int i = 0; i < n; i++)
        {
            for (int j = i; j < n; j++)
                A[i * n + j] = j;
            A[i * n + i] = i + 1;
        }
}

void
load_mat_b (size_t n, double *A)
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

void
usage (const char *restrict bin)
{
    fprintf (stderr,
             "usage: %s n m s iter_max algo\n"
             "with algo in: b, random, diag or tri\n",
             bin);
}

int
main (int argc, char *argv[])
{
    size_t n, m, s, iter_max;
    double *restrict A, *restrict v, *restrict u;
    eigen_infos *restrict w;
    char *restrict algo;

    //
    if (argc != 6)
        return usage (*argv), 255;
    n = atol (argv[1]);
    m = atol (argv[2]);
    s = atol (argv[3]);
    iter_max = atol (argv[4]);
    algo = argv[5];

    //
    if (m > n || s > m)
        return fprintf (stderr, "s <= m <= n\n"), 254;

    //
    ALLOC (A, n * n);
    ALLOC (u, m * n);
    ALLOC (v, n * (m + 1));
    ALLOC (w, m);

    //
    fill (v, 1, n);

    if (!strcmp (algo, "b"))
        {
            load_mat_b (n, A);
        }
    else if (!strcmp (algo, "random"))
        {
            gen_vect (A, n * n, 1);
        }
    else if (!strcmp (algo, "diag"))
        {
            load_mat_diag (n, A);
        }
    else if (!strcmp (algo, "tri"))
        {
            load_mat_tri (n, A);
        }
    else
        goto error;

    // Calling iram
    iram (A, v, n, s, m, iter_max, 1e-16F, w, u);

    //
    FREE (A);
    FREE (u);
    FREE (v);
    FREE (w);

    //
    return 0;

error:
    FREE (A);
    FREE (u);
    FREE (v);
    FREE (w);

    return usage (*argv), 255;
}
