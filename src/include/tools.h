#pragma once

#include <cblas.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define RANDMAX 255
#define DEBUG 1

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define ALLOC(ptr, size)                                                      \
    do                                                                        \
        {                                                                     \
            ptr = malloc (sizeof (*(ptr)) * size);                            \
            if (!(ptr))                                                       \
                {                                                             \
                    perror ("calloc");                                        \
                    exit (254);                                               \
                }                                                             \
            memset ((ptr), 0, sizeof (*(ptr)) * size);                        \
        }                                                                     \
    while (0)

#define FREE(ptr)                                                             \
    do                                                                        \
        {                                                                     \
            free ((ptr));                                                     \
        }                                                                     \
    while (0)

#define SWAP_PTR(a, b)                                                        \
    do                                                                        \
        {                                                                     \
            void *tmp = (a);                                                  \
            (a) = (b);                                                        \
            (b) = tmp;                                                        \
        }                                                                     \
    while (0)

// Infos about eigenvalues
typedef struct
{
    double re, im, err;
    size_t index; // Index in the vectors infos
} eigen_infos;

// Bandwidth constants
#define GIB ((double)(1024 << 20))
#define MIB ((double)(1024 << 10))

// Fill a vector
void fill (double *restrict v, const double a, const size_t n);

// Creates a random vector
void gen_vect (double *restrict x, const size_t n, const size_t dx);

// Creates a family of `dim` independent vectors
void gen_base (double *restrict v, const size_t n, const bool transpose);

// Prints the matrix `v` of size `n`x`m`
void print_mat (const double *restrict v, const size_t n, const size_t m,
                const size_t ldv, const char *restrict desc);

// Print the eigenvalues and the eigenvectors
void print_eigs (const double *restrict v, const eigen_infos *restrict w,
                 const size_t n, const size_t s, const char *restrict desc);

// Compute the max residual norm on the s eigenvectors
double residual_norm (const double *restrict a, const double *restrict v,
                      const eigen_infos *restrict w, const size_t n,
                      const size_t s);

// Quick clock_gettime wrapper
double my_timer (void);

// Read a mtz file
double *read_mtz (const char *restrict filename, size_t *restrict n,
                  const size_t sym);
