#pragma once

#include "ra.h"
#include <stddef.h>

void iram (const double *restrict A, double *restrict *restrict v,
           double *restrict v0, const size_t n, const size_t s, const size_t m,
           const size_t iter_max, const double tol, eigen_infos *restrict w,
           double *restrict u);
