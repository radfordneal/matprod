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


#include <stdint.h>

#ifdef MATPROD_APP_INCLUDED
#include "matprod-app.h"
#endif

#include "helpers-app.h"
#include "piped-matprod.h"


#define PIPED_MATPROD  /* Tell matprod.c it's being included from here */

#define SCOPE static

#define EXTRAD , double *start_z, double *last_z, int input_first, int threshold
#define EXTRAZ , 0, 0, 0, 0
#define EXTRAN , start_z, last_z, input_first, threshold

#define AMTOUT(_z_) do { \
    if (start_z == 0) break; \
    if (input_first) { \
        helpers_size_t a = helpers_avail0 (last_z - start_z); \
        if (a == 0) break; \
        helpers_amount_out(a); \
        if (a < last_z - start_z) break; \
        input_first = 0; \
    } \
    if ((_z_) - last_z >= threshold) { \
        helpers_amount_out ((_z_) - start_z); \
        last_z = (_z_); \
    } \
} while (0)

#define THRESH 64


#include "matprod.c"


#define OP_K(op) (op & 0x7fffffff)
#define OP_S(op) (1 + ((op >> 32) & 0xff))
#define OP_W(op) (op >> 40)

#define MAKE_OP(w,s,k) \
    (((helpers_op_t)(w)<<40) | ((helpers_op_t)((s)-1)<<32) | k)

#define ALIGNED8(z) ((((uintptr_t)(z))&7) == 0)
#define CACHE_ALIGN(z) ((double *) (((uintptr_t)(z)+0x18) & ~0x3f))

#define WAIT_FOR_EARLIER_TASKS(sz) \
    do { \
        ; /* Wait for tasks doing earlier portions to complete. */ \
    } while (helpers_avail0(LENGTH(sz)) < LENGTH(sz))


/* Dot product of two vectors, with pipelining of input y. */

void task_piped_matprod_vec_vec (helpers_op_t op, helpers_var_ptr sz, 
                                 helpers_var_ptr sx, helpers_var_ptr sy)
{
    double * MATPROD_RESTRICT x = REAL(sx);
    double * MATPROD_RESTRICT y = REAL(sy);
    double * MATPROD_RESTRICT z = REAL(sz);

    helpers_size_t k = LENGTH(sx);

    if (op && OP_W(op)!=OP_S(op)-1) return;  /* return if mistakenly split */

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

    int s = OP_S(op);
    int w = OP_W(op);

    if (k_times_m == 0) {
        if (w == s-1)  /* do in only the last thread */
            matprod_vec_mat (x, y, z, k, m, z, z, 0, THRESH);
        return;
    }

    while (4*s > m) s -= 1;

    if (s > 1) {  /* do in more than one thread */

        if (w < s) {

            helpers_size_t d, d1, a, a1;

            d = w == 0 ? 0 : (helpers_size_t) ((double)m * w / s) & ~3;
            d1 = w == s-1 ? m : (helpers_size_t) ((double)m * (w+1) / s) & ~3;
            a = d * k;
            a1 = d1 * k;

            while (d < d1) {

                helpers_size_t od = d;
                helpers_size_t oa = a;
                helpers_size_t na = d1-d <= 4 ? a1 : k*(d+4);
                HELPERS_WAIT_IN2 (a, na-1, a1);
                d = a/k;
                if (d < m) d &= ~3;

                if (d == od) continue;

                matprod_vec_mat (x, y+od*k, z+od, k, d-od, z, z+od, w, THRESH);
            }
        }
    }

    else if (w == 0) {  /* either no split, or better done in only one thread */

        helpers_size_t d = 0;
        helpers_size_t a = 0;

        while (d < m) {

            helpers_size_t od = d;
            helpers_size_t oa = a;
            helpers_size_t na = m-d <= 4 ? k_times_m : k*(d+4);
            HELPERS_WAIT_IN2 (a, na-1, k_times_m);
            d = a/k;
            if (d < m) d &= ~3;

            if (d == od) continue;

            matprod_vec_mat (x, y+od*k, z+od, k, d-od, z, z+od, 0, THRESH);
        }
    }

    if (w != 0) WAIT_FOR_EARLIER_TASKS(sz);
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

    int s = OP_S(op);
    int w = OP_W(op);

    if (k <= 1) {
        if (w == s-1)  /* do in only the last thread */
            matprod_mat_vec (x, y, z, n, k);
        return;
    }

    if (s > 1 && n > 16) {  /* do in more than one thread */

        if (s > (n+15)/16) s = (n+15)/16;

        if (w < s) {

            double *z0 = z + (size_t) ((double)n * w / s);
            double *z1 = z + (size_t) ((double)n * (w+1) / s);

            if (ALIGNED8(z)) {
                if (w != 0) z0 = CACHE_ALIGN(z0);
                if (w != s-1) z1 = CACHE_ALIGN(z1);
            }

            int xrows = z1 - z0;
            x += z0 - z;
            z += z0 - z;

            helpers_size_t a = 0;
            int add = 0;

            while (a < k) {

                helpers_size_t oa = a;
                helpers_size_t na = k-a <= 4 ? k : a+4;
                HELPERS_WAIT_IN2 (a, na-1, k);
                if (a < k) a &= ~3;

                matprod_mat_vec_sub_xrows0 (x+oa*n, y+oa, z0, 
                                            n, a-oa, xrows, add);
                add = 1;
            }
        }
    }

    else if (w == 0) {  /* either no split, or better done in only one thread */

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

    if (w != 0) WAIT_FOR_EARLIER_TASKS(sz);
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

    int s = OP_S(op);
    int w = OP_W(op);

    if (m != 0) {
        HELPERS_WAIT_IN2 (a, m-1, m);
    }

    matprod_outer (x, y, z, n, m, z, z, w, THRESH);
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

    int s = OP_S(op);
    int w = OP_W(op);

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_mat_mat (x, y, z, n, k, m, z, z, w, THRESH);
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

    int s = OP_S(op);
    int w = OP_W(op);

    if (k_times_m == 0) {
        if (w == s-1)  /* do in only the last thread */
            matprod_trans1 (x, y, z, n, k, m, z, z, 0, THRESH);
        return;
    }
        
    while (4*s > m) s -= 1;

    if (s > 1) {  /* do in more than one thread */

        if (w < s) {

            helpers_size_t d, d1, a, a1;

            d = w == 0 ? 0 : (helpers_size_t) ((double)m * w / s) & ~3;
            d1 = w == s-1 ? m : (helpers_size_t) ((double)m * (w+1) / s) & ~3;
            a = d * k;
            a1 = d1 * k;

            while (d < d1) {

                helpers_size_t od = d;
                helpers_size_t oa = a;
                helpers_size_t na = d1-d <= 4 ? a1 : k*(d+4);
                HELPERS_WAIT_IN2 (a, na-1, a1);

                d = a/k;
                if (d < m) d &= ~3;
                if (d == od) continue;
                if (d > d1) d = d1;

                matprod_trans1 (x, y+od*k, z+od*n, n, k, d-od, 
                                z, z+od*n, w, THRESH);
            }
        }
    }

    else if (w == 0) {  /* either no split, or better done in only one thread */

        helpers_size_t d = 0;
        helpers_size_t a = 0;

        while (d < m) {

            helpers_size_t od = d;
            helpers_size_t oa = a;
            helpers_size_t na = m-d <= 4 ? k_times_m : k*(d+4);
            HELPERS_WAIT_IN2 (a, na-1, k_times_m);
            d = a/k;
            if (d < m) d &= ~3;

            if (d == od) continue;

            matprod_trans1 (x, y+od*k, z+od*n, n, k, d-od, 
                            z, z+od*n, 0, THRESH);
        }
    }

    if (w != 0) WAIT_FOR_EARLIER_TASKS(sz);
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

    int s = OP_S(op);
    int w = OP_W(op);

    if (k_times_m != 0) {
        HELPERS_WAIT_IN2 (a, k_times_m-1, k_times_m);
    }

    matprod_trans2 (x, y, z, n, k, m, z, z, w, THRESH);
}


#define SPLIT_LIMIT(s,u) ((s) > (u) ? (u) : (s) < 1 ? 1 : (s))

void par_matprod_vec_vec (helpers_var_ptr z, helpers_var_ptr x, 
                          helpers_var_ptr y, int split)
{
    helpers_do_task (HELPERS_PIPE_IN2,
                     task_piped_matprod_vec_vec,
                     0, z, x, y);
}

void par_matprod_vec_mat (helpers_var_ptr z, helpers_var_ptr x, 
                          helpers_var_ptr y, int split)
{
    helpers_size_t k = LENGTH(z);
    helpers_size_t m = LENGTH(y) / k;

    int s = SPLIT_LIMIT(split,m);

    if (s > 1) {
        int w;
        for (w = 0; w < s; w++) {
            helpers_do_task (w == 0 ? HELPERS_PIPE_IN2_OUT : 
                                      HELPERS_PIPE_IN02_OUT,
                             task_piped_matprod_vec_mat,
                             MAKE_OP(w,s,0), z, x, y);
        }
    }
    else {
        helpers_do_task (HELPERS_PIPE_IN2, 
                         task_piped_matprod_vec_mat,
                         0, z, x, y);
    }
}

void par_matprod_mat_vec (helpers_var_ptr z, helpers_var_ptr x, 
                          helpers_var_ptr y, int split)
{
    helpers_size_t n = LENGTH(z);

    int s = SPLIT_LIMIT(split,n);

    if (s > 1) {
        int w;
        for (w = 0; w < s; w++) {
            helpers_do_task (w == 0  ? HELPERS_PIPE_IN2_OUT :
                             w < s-1 ? HELPERS_PIPE_IN02_OUT :
                                       HELPERS_PIPE_IN02,
                             task_piped_matprod_mat_vec,
                             MAKE_OP(w,s,0), z, x, y);
        }
    }
    else {
        helpers_do_task (HELPERS_PIPE_IN2, 
                         task_piped_matprod_mat_vec,
                         0, z, x, y);
    }
}

void par_matprod_outer (helpers_var_ptr z, helpers_var_ptr x, 
                        helpers_var_ptr y, int split)
{
    helpers_do_task (HELPERS_PIPE_IN2_OUT,
                     task_piped_matprod_outer,
                     0, z, x, y);
}

void par_matprod_mat_mat (helpers_var_ptr z, helpers_var_ptr x, 
                          helpers_var_ptr y, int k, int split)
{
    helpers_do_task (HELPERS_PIPE_IN2_OUT,
                     task_piped_matprod_mat_mat,
                     k, z, x, y);
}

void par_matprod_trans1 (helpers_var_ptr z, helpers_var_ptr x, 
                         helpers_var_ptr y, int k, int split)
{
    helpers_size_t m = LENGTH(y) / k;

    int s = SPLIT_LIMIT(split,m);

    if (REAL(x) == REAL(y) && LENGTH(x) == LENGTH(y)) {
        s = 1;  /* Don't split for the symmetric case */
    }

    if (s > 1) {
        int w;
        for (w = 0; w < s; w++) {
            helpers_do_task (w == 0 ? HELPERS_PIPE_IN2_OUT : 
                                      HELPERS_PIPE_IN02_OUT,
                             task_piped_matprod_trans1,
                             MAKE_OP(w,s,k), z, x, y);
        }
    }
    else {
        helpers_do_task (HELPERS_PIPE_IN2_OUT, 
                         task_piped_matprod_trans1,
                         k, z, x, y);
    }
}

void par_matprod_trans2 (helpers_var_ptr z, helpers_var_ptr x, 
                         helpers_var_ptr y, int k, int split)
{
    helpers_do_task (HELPERS_PIPE_IN2_OUT,
                     task_piped_matprod_trans2,
                     k, z, x, y);
}
