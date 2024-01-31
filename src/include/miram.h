#pragma once

#include "ra.h"
#include <stddef.h>

void miram (const double *restrict A, double *restrict v0, const size_t n,
            const size_t s, const size_t m0, const size_t delta_m,
            const size_t iter_max, const double tol, eigen_infos *restrict w,
            double *restrict u);
