/* MATPROD - A LIBRARY FOR MATRIX MULTIPLICATION WITH OPTIONAL PIPELINING
             Task Procedures for Matrix Multiplication With Pipelining

   Copyright (c) 2013, 2014, 2017 Radford M. Neal.

   The matprod library is free software; you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation; either version 2 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License along
   with this program; if not, write to the Free Software Foundation, Inc.,
   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
*/


#ifdef MATPROD_APP_INCLUDED
#include "matprod-app.h"
#endif

#include "helpers-app.h"
#include "piped-matprod.h"

#define SCOPE static

#define EXTRAC ,
#define EXTRAD double *start_z, double *last_z, int threshold
#define EXTRAZ 0, 0, 0
#define EXTRAN start_z, last_z, threshold

#define AMTOUT(_z_) do { \
    if (start_z != 0 && (_z_) - last_z >= threshold) { \
        helpers_amount_out ((_z_) - start_z); \
        last_z = (_z_); \
    } \
} while (0)


#include "matprod.c"


/* Dot product of two vectors, with pipelining of input y. */

void task_piped_matprod_vec_vec (helpers_op_t op, helpers_var_ptr sz, 
                                 helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k = LENGTH(sx);

    if (k == 0) {
        z[0] = 0;
        return;
    }

    helpers_size_t a = 0;
    double s = 0.0;

    while (a < k) {
        helpers_size_t oa = a;
        helpers_size_t na = k-a > 3 ? a+3  : k-1;
        HELPERS_WAIT_IN2 (a, na, k);
        if (a < k) a &= ~3;
        s = matprod_vec_vec_sub (x+oa, y+oa, a-oa, s);
    }

    z[0] = s;
}


/* Product of row vector (x) of length k and k x m matrix (y), result stored 
   in z, with pipelining of the input y and of the output. */

void task_piped_matprod_vec_mat (helpers_op_t op, helpers_var_ptr sz, 
                                 helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k_times_m = LENGTH(sy);
    helpers_size_t k = LENGTH(sx);
    helpers_size_t m = LENGTH(sz);

    helpers_size_t a = 0;

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_vec_mat (x, y, z, k, m, z, z, 64);
}


/* Product of n x k matrix (x) and column vector of length k (y) with result 
   stored in z, with pipelining of input y. */

void task_piped_matprod_mat_vec (helpers_op_t op, helpers_var_ptr sz, 
                                 helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k = LENGTH(sy);
    helpers_size_t n = LENGTH(sz);

    helpers_size_t a = 0;

    if (k != 0) {
        HELPERS_WAIT_IN2 (a, k-1, k);
    }

    matprod_mat_vec (x, y, z, n, k);
}

/* Product of an n x 1 matrix (x) and a 1 x m matrix (y) with result stored 
   in z, with pipelining of the input y and the output (by column). */

void task_piped_matprod_outer (helpers_op_t op, helpers_var_ptr sz, 
                               helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t n = LENGTH(sx);
    helpers_size_t m = LENGTH(sy);

    helpers_size_t a = 0;

    if (m != 0) {
        HELPERS_WAIT_IN2 (a, m-1, m);
    }

    matprod_outer (x, y, z, n, m, z, z, 64);
}


/* Product of an n x k matrix (x) and a k x m matrix (y) with result stored 
   in z, with pipelining of the input y and the output (by column). */

void task_piped_matprod_mat_mat (helpers_op_t op, helpers_var_ptr sz, 
                                 helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k = op;
    helpers_size_t n_times_k = LENGTH(sx);
    helpers_size_t k_times_m = LENGTH(sy);
    helpers_size_t n = n_times_k / k;
    helpers_size_t m = k_times_m / k;

    helpers_size_t a = 0;

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_mat_mat (x, y, z, n, k, m, z, z, 64);
}


/* Product of the transpose of a k x n matrix (x) and a k x m matrix (y) 
   with result stored in z, with pipelining of the input y and the output
   (by column). */

void task_piped_matprod_trans1 (helpers_op_t op, helpers_var_ptr sz, 
                                helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k = op;
    helpers_size_t n_times_k = LENGTH(sx);
    helpers_size_t k_times_m = LENGTH(sy);
    helpers_size_t n = n_times_k / k;
    helpers_size_t m = k_times_m / k;

    helpers_size_t a = 0;

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_trans1 (x, y, z, n, k, m, z, z, 64);
}


/* Product of an n x k matrix (x) and the transpose of an m x k matrix (y) 
   with result stored in z, with pipelining of the output (by column). */

void task_piped_matprod_trans2 (helpers_op_t op, helpers_var_ptr sz, 
                                helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k = op;
    helpers_size_t n_times_k = LENGTH(sx);
    helpers_size_t k_times_m = LENGTH(sy);
    helpers_size_t n = n_times_k / k;
    helpers_size_t m = k_times_m / k;

    helpers_size_t a = 0;

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_trans2 (x, y, z, n, k, m, z, z, 64);
}
