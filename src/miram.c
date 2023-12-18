#include "miram.h"
#include "qr.h"
#include <math.h>
#include <omp.h>
#include <stdbool.h>

void
miram (const double *restrict A, double *restrict v, double *restrict v0,
       const size_t n, const size_t s, const size_t m0, const size_t iter_max,
       const double tol, eigen_infos *restrict w, double *restrict u)
{
    const size_t nb_threads = omp_get_max_threads ();
    const size_t m_max = m0 + nb_threads;
    double *restrict _hbest, *restrict _vbest;

    // Shared variables
    double best_er = 1.0e16L;
    bool stop = false;
    size_t count = 0;
    size_t nrc = 0;

    //
    ALLOC (_hbest, m_max * (m_max + 1));
    ALLOC (_vbest, n * (m_max + 1));

    // Print the header
    printf ("# %10s %10s %10s %10s\n", "nrc global", "err", "nrc local",
            "tid");

#pragma omp parallel
    {
        const size_t tid = omp_get_thread_num ();
        const size_t m = m0 + tid;

        double err = 0;
        double *restrict _Q, *restrict _QQ;
        double *restrict _h, *restrict _hh;
        double *restrict _vv, *restrict _v;
        double *restrict _wi, *restrict _wr;
        double *restrict _ym;
        double *restrict _tau;
        double *restrict _fs, *restrict _fm;
        double *restrict _u;
        eigen_infos *restrict _w;

        // Allocation
        ALLOC (_Q, m_max * m_max);
        ALLOC (_QQ, m_max * m_max);
        ALLOC (_fm, n);
        ALLOC (_fs, n);
        ALLOC (_h, m_max * (m_max + 1));
        ALLOC (_hh, m_max * (m_max + 1));
        ALLOC (_tau, m_max);
        ALLOC (_vv, n * (m_max + 1));
        ALLOC (_v, n * (m_max + 1));
        ALLOC (_wi, m_max);
        ALLOC (_wr, m_max);
        ALLOC (_w, m);
        ALLOC (_ym, m_max * m_max);
        ALLOC (_u, m * n);

        // Copy v0 into v
        cblas_dcopy (n, v0, 1, _v, 1);

        // Call Ritz Arnoldi
        ritz_arnoldi (A, _v, n, s, m, &err, _u, _w, _h, m_max, _hh, _ym, _wi,
                      _wr, 0);

        //
        for (size_t iter = 0; iter < iter_max; ++iter)
            {
                // Check for stop
                if (stop)
                    break;

                // Check for "convergence"
                if (err < tol)
                    {
#pragma omp critical
                        stop = true;

                        // Copy the result
                        memcpy (w, _w, s * sizeof (*w));
                        cblas_dcopy (m * n, _u, 1, u, 1);

                        break;
                    }

#pragma omp critical
                {
                    /*
                     * Avoid repeating the same subspace.
                     * The problem is not convex, so we may have
                     * to change the subspace after a while.
                     */
                    if (count > nb_threads)
                        {
                            best_er = 1.0e16L;
                        }

                    /*
                     * If we have a better error, we consider the current
                     * subspace as the best one.
                     */
                    if (err <= best_er)
                        {
                            // Print the error
                            printf ("%10zu % 10e %10zu %10zu\n", nrc, err,
                                    iter, tid);

                            // Call QR on h
                            shifted_qr (_h, m_max, m, _Q, _w, s, _QQ, _tau);

                            /*
                             * We have, with v(i) the columns of V:
                             * V = V * Q
                             * In our case, v(i) are the lines of V,
                             * so we need to transform it into:
                             * V = Q^t * V
                             */
                            cblas_dgemm (CblasRowMajor, CblasTrans,
                                         CblasNoTrans, m, n, m, 1.0, _Q, m_max,
                                         _v, n, 0.0, _vv, n);
                            SWAP_PTR (_v, _vv);

                            // beta = h(s + 1, s), sigma = Q(m, s)
                            const double beta
                                = _h[m_max * (s - 1) + s - 1 - 1];
                            const double sigma
                                = _Q[m_max * (m - 1) + s - 1 - 1];

                            // fs = beta fs + sigma fm
                            cblas_dcopy (n, _vv + n * (m - 1), 1, _fm, 1);
                            cblas_dcopy (n, _v + n * (s - 1), 1, _fs, 1);
                            cblas_dscal (n, beta, _fs, 1);
                            cblas_daxpy (n, sigma, _fm, 1, _fs, 1);
                            cblas_dcopy (n, _fs, 1, _v + n * (s - 1), 1);

                            //
                            best_er = err;
                            cblas_dcopy (n * (s), _v, 1, _vbest, 1);
                            cblas_dcopy (m_max * (s), _h, 1, _hbest, 1);
                            count = 0;
                            nrc++;
                        }
                    /*
                     * Else, we copy the best subspace into the current one.
                     */
                    else
                        {
                            cblas_dcopy (n * (s), _vbest, 1, _v, 1);
                            cblas_dcopy (m_max * (s), _hbest, 1, _h, 1);
                            count++;
                        }
                }

                // Call Ritz Arnoldi
                ritz_arnoldi (A, _v, n, s, m, &err, _u, _w, _h, m_max, _hh,
                              _ym, _wi, _wr, s - 1);
            }

        //
        FREE (_Q);
        FREE (_QQ);
        FREE (_fm);
        FREE (_fs);
        FREE (_h);
        FREE (_hh);
        FREE (_tau);
        FREE (_vv);
        FREE (_v);
        FREE (_wi);
        FREE (_wr);
        FREE (_w);
        FREE (_ym);
        FREE (_u);
    }

    //
    FREE (_hbest);
    FREE (_vbest);
}
