#pragma once

#include "tools.h"

// Algorithme Ritz Arnoldi
void ritz_arnoldi (const double *restrict A, double *restrict v,
                   const size_t n, const size_t s, const size_t m,
                   double *restrict err, double *restrict u,
                   eigen_infos *restrict w, double *restrict _h,
                   double *restrict _vr, double *restrict _wi,
                   double *restrict _wr, const size_t jj);
