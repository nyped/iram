#pragma once

#include "tools.h"
#include <stddef.h>

void shifted_qr (double *restrict h, const size_t ldh, const size_t m,
                 double *restrict Q, const eigen_infos *restrict w,
                 const size_t s, double *restrict _Q, double *restrict _tau);
