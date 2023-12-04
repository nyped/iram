#pragma once

#include <cblas.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>
#include <time.h>

#define RANDMAX 255
#define DEBUG 1

#define MAX(a, b) ((a) > (b) ? (a) : (b))
#define MIN(a, b) ((a) < (b) ? (a) : (b))

#define ALLOC(ptr, size)                                                      \
    do                                                                        \
        {                                                                     \
            ptr = malloc (sizeof (*ptr) * size);                              \
            if (!ptr)                                                         \
                {                                                             \
                    perror ("calloc");                                        \
                    exit (254);                                               \
                }                                                             \
        }                                                                     \
    while (0)

#define FREE(ptr)                                                             \
    do                                                                        \
        {                                                                     \
            free (ptr);                                                       \
        }                                                                     \
    while (0)

#define SWAP_PTR(a, b)                                                        \
    do                                                                        \
        {                                                                     \
            void *tmp = a;                                                    \
            a = b;                                                            \
            b = tmp;                                                          \
        }                                                                     \
    while (0)

// Infos about eigenvalues
typedef struct
{
    double re, im, err;
    size_t index; // Index in the vectors infos
} eigen_infos;

#define GIB ((double)(1024 << 20))
#define MIB ((double)(1024 << 10))

// Fill a vector
void fill (double *restrict v, const double a, const size_t n);

// Creates a random vector
void gen_vect (double *x, const size_t n, const size_t dx);

// Creates a family of `dim` independent vectors
void gen_base (double *restrict v, const size_t n, const bool transpose);

// Prints the matrix `v` of size `n`x`m`
void print_mat (const double *v, const size_t n, const size_t m,
                const size_t ldv, const char *desc);

// Print the eigenvalues and the eigenvectors
void print_eigs (const double *v, const eigen_infos *w, const size_t n,
                 const size_t s, const char *desc);

// Quick clock_gettime wrapper
double my_timer (void);

// Compute the mean of an array
double mean (const double *a, const size_t n);

// Compute the stddev of an array
double stddev (const double *restrict a, const double mean, const size_t n);

// Read a matrix from a file
double *read_matrix (const char *filename, size_t *restrict n);

// Compute the orthogonality matrix
void orthogonality_mat (const double *restrict a, double *restrict o,
                        const size_t n);
