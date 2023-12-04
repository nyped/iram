#pragma once

#include <stddef.h>

// Arnoldi reduction using mgs
void arnoldi_mgs (const double *restrict A, double *restrict v,
                  double *restrict h, const size_t jj, const size_t n,
                  const size_t m);
