/* MATPROD - A LIBRARY FOR MATRIX MULTIPLICATION WITH OPTIONAL PIPELINING
             Task Procedures for Matrix Multiplication With Pipelining

   Copyright (c) 2013, 2014, 2017, 2018 Radford M. Neal.

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


#define PIPED_MATPROD  /* Tell matprod.c it's being included from here */

#define SCOPE static

#define EXTRAD , double *start_z, double *last_z, int threshold
#define EXTRAZ , 0, 0, 0
#define EXTRAN , start_z, last_z, threshold

#define AMTOUT(_z_) do { \
    if (start_z != 0 && (_z_) - last_z >= threshold) { \
        helpers_amount_out ((_z_) - start_z); \
        last_z = (_z_); \
    } \
} while (0)

#define THRESH 64


#include "matprod.c"


#define OP_K(op) (op & 0x7fffffff)


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
        helpers_size_t na = k-a <= 4 ? k : a+4;
        HELPERS_WAIT_IN2 (a, na-1, k);
        if (a < k) a &= ~3;

        if (a == oa) continue;

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

    if (k_times_m == 0) {
        matprod_vec_mat (x, y, z, k, m, z, z, THRESH);
        return;
    }

    helpers_size_t a = 0;
    helpers_size_t d = 0;

    while (d < m) {

        helpers_size_t oa = a;
        helpers_size_t od = d;
        helpers_size_t na = m-d <= 4 ? k_times_m : k*(d+4);
        HELPERS_WAIT_IN2 (a, na-1, k_times_m);
        d = a/k;
        if (d < m) d &= ~3;

        if (d == od) continue;

        matprod_vec_mat (x, y+od*k, z+od*k, k, d-od, z, z+od*k, THRESH);
    }
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

    if (k == 0) {
        matprod_mat_vec (x, y, z, n, k);
        return;
    }

    helpers_size_t a = 0;
    int add = 0;

    while (a < k) {

        helpers_size_t oa = a;
        helpers_size_t na = k-a <= 4 ? k : a+4;
        HELPERS_WAIT_IN2 (a, na-1, k);
        if (a < k) a &= ~3;

        matprod_mat_vec_sub (x+oa*n, y+oa, z, n, a-oa, add);
        add = 1;
    }
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

    matprod_outer (x, y, z, n, m, z, z, THRESH);
}


/* Product of an n x k matrix (x) and a k x m matrix (y) with result stored 
   in z, with pipelining of the input y and the output (by column). */

void task_piped_matprod_mat_mat (helpers_op_t op, helpers_var_ptr sz, 
                                 helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k = OP_K(op);
    helpers_size_t n_times_k = LENGTH(sx);
    helpers_size_t k_times_m = LENGTH(sy);
    helpers_size_t n = n_times_k / k;
    helpers_size_t m = k_times_m / k;

    helpers_size_t a = 0;

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_mat_mat (x, y, z, n, k, m, z, z, THRESH);
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

    helpers_size_t k = OP_K(op);
    helpers_size_t n_times_k = LENGTH(sx);
    helpers_size_t k_times_m = LENGTH(sy);
    helpers_size_t n = n_times_k / k;
    helpers_size_t m = k_times_m / k;

    helpers_size_t a = 0;

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_trans1 (x, y, z, n, k, m, z, z, THRESH);
}


/* Product of an n x k matrix (x) and the transpose of an m x k matrix (y) 
   with result stored in z, with pipelining of the output (by column). */

void task_piped_matprod_trans2 (helpers_op_t op, helpers_var_ptr sz, 
                                helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k = OP_K(op);
    helpers_size_t n_times_k = LENGTH(sx);
    helpers_size_t k_times_m = LENGTH(sy);
    helpers_size_t n = n_times_k / k;
    helpers_size_t m = k_times_m / k;

    helpers_size_t a = 0;

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_trans2 (x, y, z, n, k, m, z, z, THRESH);
}
