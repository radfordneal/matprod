/* MATPROD - A LIBRARY FOR MATRIX MULTIPLICATION WITH OPTIONAL PIPELINING
             C Procedures for Matrix Multiplication Without Pipelining

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

#include "matprod.h"


/* Set up alignment definitions. */

#ifndef ALIGN
#define ALIGN 1
#endif

#ifndef ALIGN_OFFSET
#define ALIGN_OFFSET 0
#endif

#if ALIGN >= 8 && __GNUC__
#define CAN_ASSUME_ALIGNED 1
#else
#define CAN_ASSUME_ALIGNED 0
#endif


/* Set up SIMD definitions. */

#if __SSE2__ && !defined(DISABLE_SIMD_CODE)
#define CAN_USE_SSE2 1
#else
#define CAN_USE_SSE2 0
#endif

#if __AVX__ && !defined(DISABLE_SIMD_CODE) && !defined(DISABLE_AVX_CODE)
#define CAN_USE_AVX 1
#else
#define CAN_USE_AVX 0
#endif

#if CAN_USE_SSE2 || CAN_USE_AVX
#include <immintrin.h>
#endif


/* Dot product of two vectors of length k. 

   Two implementations are provided.  One uses the obvious loop, which
   maybe the compiler will optimize well.  In the other, the loop is
   unrolled to add four products to the sum each iteration, perhaps
   using SSE2 instructions.

   Use -DALT_MATPROD_VEC_VEC to switch between these two implementations.
   Change #ifdef to #ifndef or vice versa below to change the default. */

double matprod_vec_vec (double * MATPROD_RESTRICT x, 
                        double * MATPROD_RESTRICT y, int k)
{
#   if CAN_ASSUME_ALIGNED
        x = __builtin_assume_aligned (x, ALIGN, ALIGN_OFFSET);
        y = __builtin_assume_aligned (y, ALIGN, ALIGN_OFFSET);
#   endif

#   ifdef ALT_MATPROD_VEC_VEC
    {
        double s = 0.0;
        int i;

        for (i = 0; i < k; i++)
            s += x[i] * y[i];

        return s;
    }
#   else
    {
        double s;
        int i;

#       if (ALIGN_OFFSET & 8)
            s = x[0] * y[0];
            i = 1;
            k -= 1;
#           if CAN_ASSUME_ALIGNED
                x = __builtin_assume_aligned (x, ALIGN,
                        (ALIGN_OFFSET+8)&(ALIGN-1));
                y = __builtin_assume_aligned (y, ALIGN,
                        (ALIGN_OFFSET+8)&(ALIGN-1));
#           endif
#       else
            s = 0.0;
            i = 0;
#       endif

        int e = k&(~3);

#       if CAN_USE_SSE2
        {
            __m128d S = _mm_load_sd(&s);
            __m128d A;
            while (i < e) {
                A = ALIGN >= 16 
                     ? _mm_mul_pd(_mm_load_pd(x+i),_mm_load_pd(y+i))
                     : _mm_mul_pd(_mm_loadu_pd(x+i),_mm_loadu_pd(y+i));
                S = _mm_add_sd (A, S);
                A = _mm_unpackhi_pd (A, A);
                S = _mm_add_sd (A, S);
                A  = ALIGN >= 16 
                      ? _mm_mul_pd(_mm_load_pd(x+i+2),_mm_load_pd(y+i+2))
                      : _mm_mul_pd(_mm_loadu_pd(x+i+2),_mm_loadu_pd(y+i+2));
                S = _mm_add_sd (A, S);
                A = _mm_unpackhi_pd (A, A);
                S = _mm_add_sd (A, S);
                i += 4;
            }
            if (k & 2) {
                A = ALIGN >= 16 
                     ? _mm_mul_pd(_mm_load_pd(x+i),_mm_load_pd(y+i))
                     : _mm_mul_pd(_mm_loadu_pd(x+i),_mm_loadu_pd(y+i));
                S = _mm_add_sd (A, S);
                A = _mm_unpackhi_pd (A, A);
                S = _mm_add_sd (A, S);
                i += 2;
            }
            _mm_store_sd (&s, S);
            if (k & 1) {
                s += x[i+0] * y[i+0];
            }
        }
#       else  /* non-SIMD */
        {
            while (i < e) {
                s += x[i+0] * y[i+0];
                s += x[i+1] * y[i+1];
                s += x[i+2] * y[i+2];
                s += x[i+3] * y[i+3];
                i += 4;
            }
            if (k & 2) {
                s += x[i+0] * y[i+0];
                s += x[i+1] * y[i+1];
                i += 2;
            }
            if (k & 1) {
                s += x[i+0] * y[i+0];
            }
        }
#       endif
        return s;
    }
#   endif
}


/* Product of row vector (x) of length k and k x m matrix (y), result
   stored in z.

   The inner loop is a set of four vector dot products.  Two
   implementations are provided, one with loop unrolling within each
   dot product, the other without (since maybe the compiler does a
   better job of this).  The loop unrolling to do four dot products at
   one time is done manually in both implementations.

   Cases where k is 0 or 1 are handled specially.

   Use -DALT_MATPROD_VEC_MAT to switch between these two implementations.
   Change #ifdef to #ifndef or vice versa below to change the default. */

void matprod_vec_mat (double * MATPROD_RESTRICT x, 
                      double * MATPROD_RESTRICT y, 
                      double * MATPROD_RESTRICT z, int k, int m)
{
#   if CAN_ASSUME_ALIGNED
        x = __builtin_assume_aligned (x, ALIGN, ALIGN_OFFSET);
        y = __builtin_assume_aligned (y, ALIGN, ALIGN_OFFSET);
        z = __builtin_assume_aligned (z, ALIGN, ALIGN_OFFSET);
#   endif

    /* Specially handle scalar times row vector, and zero-length matrix. 
       Assumes the compiler can optimize this with SIMD instructions if
       that is desirable. */

    if (k <= 1) {
        int j;
        if (k == 0) {
            for (j = 0; j < m; j++) {
                z[j] = 0.0;
            }
        }
        else {
            double t = x[0];
            for (j = 0; j < m; j++) {
                z[j] = t * y[j];
            }
        }
        return;
    }

    double *p;             /* Pointer that goes along pairs in x */
    double *e = x+(k&~1);  /* Position after last complete pair in x */

#   if CAN_USE_AVX
    int use_AVX = (double)k*m>160; /* Use AVX only if significant computation */
                                   /*   due to possible transition cost       */
#   endif

    /* In this loop, compute four consecutive elements of the result vector,
       by doing four dot products of x with columns of y.  Adjust y, z, and
       m as we go. */

    while (m >= 4) {

#       ifdef ALT_MATPROD_VEC_MAT
        {
            /* SIMPLE CODE */

            double s[4] = { 0, 0, 0, 0 };
            int i;

            /* Each time around this loop, add one product to the sum
               for each of the four dot products. */

            for (i = 0; i<k; i++) {
                double t = x[i];
                s[0] += t * y[i];
                s[1] += t * y[i+k];
                s[2] += t * y[i+2*k];
                s[3] += t * y[i+3*k];
            }

            y += 4*k;

            z[0] = s[0];
            z[1] = s[1];
            z[2] = s[2];
            z[3] = s[3];
        }

        /* The various versions of the loop below all add two products
           to the sums for each of the four dot products, adjusting p
           and y as we go.  A possible final set of four products is
           then added afterwards.  The sums may be initialized to zero
           or to the result of one product, if that helps
           alignment. */

#       elif CAN_USE_SSE2  /* or AVX */
        {
#           if CAN_USE_AVX
            if (use_AVX)
            {
                /* AVX CODE */

                __m256d S, B;

#               if (ALIGN_OFFSET & 8)
                    S = _mm256_set_pd (y[3*k], y[2*k], y[k], y[0]);
                    S = _mm256_mul_pd (_mm256_set1_pd(x[0]), S);
                    p = x+1;
                    y += 1;
#                   if CAN_ASSUME_ALIGNED
                        p = __builtin_assume_aligned (p, ALIGN, 
                                 (ALIGN_OFFSET+8)&(ALIGN-1));
                        y = __builtin_assume_aligned (y, ALIGN,
                                 (ALIGN_OFFSET+8)&(ALIGN-1));
#                   endif
                    e = p+((k-1)&~1);
#               else
                    S = _mm256_setzero_pd ();
                    p = x;
#               endif

                while (p < e) {
                    __m128d Y0, Y1, Y2, Y3;
                    __m256d T0, T1;
                    Y0 = ALIGN >= 16 ? _mm_load_pd(y) : _mm_loadu_pd(y);
                    Y1 = _mm_loadu_pd(y+k);
                    Y2 = ALIGN >= 16 ? _mm_load_pd(y+2*k) : _mm_loadu_pd(y+2*k);
                    Y3 = _mm_loadu_pd(y+3*k);
                    T0 = _mm256_castpd128_pd256 (Y0);
                    T0 = _mm256_insertf128_pd (T0, Y2, 1);
                    T1 = _mm256_castpd128_pd256 (Y1);
                    T1 = _mm256_insertf128_pd (T1, Y3, 1);
                    B = _mm256_unpacklo_pd (T0, T1);
                    B = _mm256_mul_pd (_mm256_set1_pd(p[0]), B);
                    S = _mm256_add_pd (B, S);
                    B = _mm256_unpackhi_pd (T0, T1);
                    B = _mm256_mul_pd (_mm256_set1_pd(p[1]), B);
                    S = _mm256_add_pd (B, S);
                    p += 2;
                    y += 2;
                }

                if (p < x+k) {
                    B = _mm256_set_pd (y[3*k], y[2*k], y[k], y[0]);
                    B = _mm256_mul_pd (_mm256_set1_pd(p[0]), B);
                    S = _mm256_add_pd (B, S);
                    y += 1;
                }

                y += 3*k;

                _mm256_storeu_pd (z, S);
            }
            else   /* End of AVX code, which is done conditionally, */
#           endif  /*   so the SSE2 code below may be used instead. */
            {
                /* SSE2 CODE */

                __m128d S0, S1, B, P;

#               if (ALIGN_OFFSET & 8)
                {
                    P = _mm_set1_pd(x[0]);
                    S0 = _mm_mul_pd (P, _mm_set_pd (y[k], y[0]));
                    S1 = _mm_mul_pd (P, _mm_set_pd (y[3*k], y[2*k]));
                    p = x+1;
                    y += 1;
#                   if CAN_ASSUME_ALIGNED
                        p = __builtin_assume_aligned (p, ALIGN,
                                 (ALIGN_OFFSET+8)&(ALIGN-1));

                        y = __builtin_assume_aligned (y, ALIGN,
                                 (ALIGN_OFFSET+8)&(ALIGN-1));
#                   endif
                    e = p+((k-1)&~1);
                }
#               else
                {
                    S0 = _mm_setzero_pd ();
                    S1 = _mm_setzero_pd ();
                    p = x;
                }
#               endif

#               if ALIGN >= 16
                {
                    __m128d Z = _mm_setzero_pd();
                    while (p < e) {
                        __m128d T0, T1, Q;
                        P = _mm_load_pd (p);
                        Q = _mm_unpacklo_pd (P, P);
                        T0 = _mm_load_pd(y);
                        B = _mm_mul_pd (Q, _mm_loadh_pd (T0, y+k));
                        S0 = _mm_add_pd (S0, B);
                        T1 = _mm_load_pd(y+2*k);
                        Q = _mm_mul_pd (Q, _mm_loadh_pd (T1, y+3*k));
                        S1 = _mm_add_pd (S1, Q);
                        P = _mm_unpackhi_pd (P, P);
                        Z = _mm_loadh_pd (Z, y+k+1);
                        T0 = _mm_unpackhi_pd (T0, Z);
                        B = _mm_mul_pd (P, T0);
                        S0 = _mm_add_pd (S0, B);
                        Z = _mm_loadh_pd (Z, y+3*k+1);
                        T1 = _mm_unpackhi_pd (T1, Z);
                        B = _mm_mul_pd (P, T1);
                        S1 = _mm_add_pd (S1, B);
                        p += 2;
                        y += 2;
                    }
                }
#               else
                {
                    while (p < e) {
                        P = _mm_set1_pd(p[0]);
                        B = _mm_set_pd (y[k], y[0]);
                        B = _mm_mul_pd (P, B);
                        S0 = _mm_add_pd (S0, B);
                        B = _mm_set_pd (y[3*k], y[2*k]);
                        B = _mm_mul_pd (P, B);
                        S1 = _mm_add_pd (S1, B);
                        P = _mm_set1_pd(p[1]);
                        B = _mm_set_pd (y[k+1], y[1]);
                        B = _mm_mul_pd (P, B);
                        S0 = _mm_add_pd (S0, B);
                        B = _mm_set_pd (y[3*k+1], y[2*k+1]);
                        B = _mm_mul_pd (P, B);
                        S1 = _mm_add_pd (S1, B);
                        p += 2;
                        y += 2;
                    }
                }
#               endif

                if (p < x+k) {
                    P = _mm_set1_pd(p[0]);
                    B = _mm_mul_pd (P, _mm_set_pd (y[k], y[0]));
                    S0 = _mm_add_pd (S0, B);
                    B = _mm_mul_pd (P, _mm_set_pd (y[3*k], y[2*k]));
                    S1 = _mm_add_pd (S1, B);
                    y += 1;
                }

                y += 3*k;

                _mm_storeu_pd (z, S0);
                _mm_storeu_pd (z+2, S1);
            }
        }
#       else
        {
            /* NON-SIMD CODE */

            double s[4] = { 0, 0, 0, 0 };
            p = x;
            while (p < e) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[2] += p[0] * y[2*k];
                s[3] += p[0] * y[3*k];
                s[0] += p[1] * y[1];
                s[1] += p[1] * y[k+1];
                s[2] += p[1] * y[2*k+1];
                s[3] += p[1] * y[3*k+1];
                y += 2;
                p += 2;
            }

            if (k & 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[2] += p[0] * y[2*k];
                s[3] += p[0] * y[3*k];
                y += 1;
            }

            y += 3*k;

            z[0] = s[0];
            z[1] = s[1];
            z[2] = s[2];
            z[3] = s[3];
        }
#       endif

        z += 4;
        m -= 4;
    }

    /* Compute the final few dot products left over from the loop above. 
       There appears to be no advantage to explicitly using AVX or SSE2 
       to do this. */

    if (m & 2) {  /* Do two more dot products */

#       ifdef ALT_MATPROD_VEC_MAT
        {
            double s[2] = { 0, 0 };
            int i;
            for (i = 0; i<k; i++) {
                s[0] += x[i] * y[i];
                s[1] += x[i] * y[i+k];
            }
            y += 2*k;
            z[0] = s[0];
            z[1] = s[1];
        }
#       else
        {
            double s[2] = { 0, 0 };
            p = x;

            while (p < e) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[0] += p[1] * y[1];
                s[1] += p[1] * y[k+1];
                p += 2;
                y += 2;
            }

            if (k & 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                y += 1;
            }

            y += k;
            z[0] = s[0];
            z[1] = s[1];
        }
#       endif

        z += 2;
    }

    if (m & 1) {  /* Do one final dot product */

#       ifdef ALT_MATPROD_VEC_MAT
        {
            double s = 0.0;
            int i;
            for (i = 0; i < k; i++)
                s += x[i] * y[i];
            z[0] = s;
        }
#       else
        {
            double s = 0.0;
            p = x;

            while (p < e) {
                s += p[0] * y[0];
                s += p[1] * y[1];
                p += 2;
                y += 2;
            }

            if (k & 1) {
                s += p[0] * y[0];
            }

            z[0] = s;
        }
#       endif
    }
}


/* Product of n x k matrix (x) and column vector of length k (y) with result 
   stored in z. 

   The product is computed using an outer loop that accumulates the sums for 
   all elements of the result vector, iterating over columns of x, in order
   to produce sequential accesses.  This loop is unrolled to accumulate from
   two columns of x at once.

   The case of n=2 may be handled specially, accumulating sums in a
   local variable rather than in the result vector, and then storing
   them in the result at the end.  Whether this is done can be
   controlled using -DALT_MATPROD_MAT_VEC.  Change #ifdef to #ifndef
   or vice versa below to change the default. */

void matprod_mat_vec (double * MATPROD_RESTRICT x, 
                      double * MATPROD_RESTRICT y, 
                      double * MATPROD_RESTRICT z, int n, int k)
{
    if (n <= 0) return;

#   if CAN_ASSUME_ALIGNED
        x = __builtin_assume_aligned (x, ALIGN, ALIGN_OFFSET);
        y = __builtin_assume_aligned (y, ALIGN, ALIGN_OFFSET);
        z = __builtin_assume_aligned (z, ALIGN, ALIGN_OFFSET);
#   endif

    /* Specially handle scalar times row vector and zero-length matrix. */

    if (k <= 1) {
        int j;
        if (k == 0) {
            for (j = 0; j < n; j++) {
                z[j] = 0.0;
            }
        }
        else {
            double t = y[0];
            for (j = 0; j < n; j++) {
                z[j] = t * x[j];
            }
        }
        return;
    }

    double *e = y+(k&~1);  /* point to stop at in/after vector y */

#   ifndef ALT_MATPROD_MAT_VEC
    {
        if (n == 2) { 

#           if CAN_USE_SSE2

                __m128d S;  /* sums for the two values in the result */
                __m128d A;

                /* Each time around this loop, add the products of two
                   columns of x with two elements of y to S.  Adjust x
                   and y to account for this. */

                S = _mm_setzero_pd (); 

                while (y < e) {
#                   if ALIGN >= 16 && ALIGN_OFFSET == 0
                        A = _mm_mul_pd (_mm_load_pd(x), _mm_load1_pd(y));
                        S = _mm_add_pd (A, S);
                        A = _mm_mul_pd (_mm_load_pd(x+2), _mm_load1_pd(y+1));
                        S = _mm_add_pd (A, S);
#                   else
                        A = _mm_mul_pd (_mm_loadu_pd(x), _mm_load1_pd(y));
                        S = _mm_add_pd (A, S);
                        A = _mm_mul_pd (_mm_loadu_pd(x+2), _mm_load1_pd(y+1));
                        S = _mm_add_pd (A, S);
#                   endif
                    x += 4;
                    y += 2;
                }

                if (k & 1) {
#                   if ALIGN >= 16 && ALIGN_OFFSET == 0
                        A = _mm_mul_pd (_mm_load_pd(x), _mm_load1_pd(y));
                        S = _mm_add_pd (A, S);
#                   else
                        A = _mm_mul_pd (_mm_loadu_pd(x), _mm_load1_pd(y));
                        S = _mm_add_pd (A, S);
#                   endif
                }

                /* Store the two sums in S in the result vector. */

                _mm_storeu_pd (z, S);

#           else

                double s[2];  /* sums for the two values in the result */

                s[0] = s[1] = 0;

                /* Each time around this loop, add the products of two
                   columns of x with two elements of y to s[0] and s[1].
                   Adjust x and y to account for this. */

                while (y < e) {
                    s[0] = (s[0] + (x[0] * y[0])) + (x[2] * y[1]);
                    s[1] = (s[1] + (x[1] * y[0])) + (x[3] * y[1]);
                    x += 4;
                    y += 2;
                }

                if (k & 1) {
                    s[0] += x[0] * y[0];
                    s[1] += x[1] * y[0];
                }

                /* Store the two sums in s[0] and s[1] in the result vector. */

                z[0] = s[0];
                z[1] = s[1];


#           endif

            return;
        }
    }
#   endif

    /* To start, set result to all zeros. */

    int i = 0;
    do { z[i] = 0.0; } while (i++ < n);

    /* Each time around this loop, add the products of two columns of x 
       with two elements of y to the result vector, z.  Adjust x and y
       to account for this. */

    double *q, *f;

#   if CAN_USE_AVX

        while (y < e) {
            q = z;
            f = z + (n&~3);
            __m256d Y0b = _mm256_set1_pd(y[0]);
            __m256d Y1b = _mm256_set1_pd(y[1]);
            while (q < f) { 
                __m256d Q = _mm256_loadu_pd(q);
                __m256d X = _mm256_mul_pd (_mm256_loadu_pd(x), Y0b);
                __m256d P = _mm256_mul_pd (_mm256_loadu_pd(x+n), Y1b);
                Q = _mm256_add_pd (_mm256_add_pd (Q, X), P);
                _mm256_storeu_pd (q, Q);
                x += 4;
                q += 4;
            }
            if (n & 2) {
                __m128d Y0 = _mm_set1_pd(y[0]);
                __m128d Y1 = _mm_set1_pd(y[1]);
                __m128d Q = _mm_loadu_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadu_pd(x), Y0);
                __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
                Q = _mm_add_pd (_mm_add_pd (Q, X), P);
                _mm_storeu_pd (q, Q);
                x += 2;
                q += 2;
            }
            if (n & 1) {
                q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
                x += 1;
            }
            x += n;
            y += 2;
        }

        if (k & 1) {
            q = z;
            f = z + (n&~3);
            __m256d Yb = _mm256_set1_pd(y[0]);
            while (q < f) { 
                __m256d Q = _mm256_loadu_pd(q);
                __m256d X = _mm256_mul_pd (_mm256_loadu_pd(x), Yb);
                Q = _mm256_add_pd (Q, X);
                _mm256_storeu_pd (q, Q);
                x += 4;
                q += 4;
            }
            if (n & 2) { 
                __m128d Y = _mm_set1_pd(y[0]);
                __m128d Q = _mm_loadu_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadu_pd(x), Y);
                Q = _mm_add_pd (Q, X);
                _mm_storeu_pd (q, Q);
                x += 2;
                q += 2;
            }
            if (n & 1) {
                q[0] += x[0] * y[0];
            }
        }


#   elif CAN_USE_SSE2

        while (y < e) {
            q = z;
            f = z + (n&~3);
            __m128d Y0 = _mm_load1_pd(y);
            __m128d Y1 = _mm_load1_pd(y+1);
            while (q < f) { 
                __m128d Q, X, P;
                Q = _mm_loadu_pd(q);
                X = _mm_mul_pd (_mm_loadu_pd(x), Y0);
                P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
                Q = _mm_add_pd (_mm_add_pd (Q, X), P);
                _mm_storeu_pd (q, Q);
                Q = _mm_loadu_pd(q+2);
                X = _mm_mul_pd (_mm_loadu_pd(x+2), Y0);
                P = _mm_mul_pd (_mm_loadu_pd(x+n+2), Y1);
                Q = _mm_add_pd (_mm_add_pd (Q, X), P);
                _mm_storeu_pd (q+2, Q);
                x += 4;
                q += 4;
            }
            if (n & 2) {
                __m128d Q = _mm_loadu_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadu_pd(x), Y0);
                __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
                Q = _mm_add_pd (_mm_add_pd (Q, X), P);
                _mm_storeu_pd (q, Q);
                x += 2;
                q += 2;
            }
            if (n & 1) {
                q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
                x += 1;
            }
            x += n;
            y += 2;
        }

        if (k & 1) {
            q = z;
            f = z + (n&~3);
            __m128d Y = _mm_load1_pd(y);
            while (q < f) { 
                __m128d Q, X;
                Q = _mm_loadu_pd(q);
                X = _mm_mul_pd (_mm_loadu_pd(x), Y);
                Q = _mm_add_pd (Q, X);
                _mm_storeu_pd (q, Q);
                Q = _mm_loadu_pd(q+2);
                X = _mm_mul_pd (_mm_loadu_pd(x+2), Y);
                Q = _mm_add_pd (Q, X);
                _mm_storeu_pd (q+2, Q);
                x += 4;
                q += 4;
            }
            if (n & 2) { 
                __m128d Q = _mm_loadu_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadu_pd(x), Y);
                Q = _mm_add_pd (Q, X);
                _mm_storeu_pd (q, Q);
                x += 2;
                q += 2;
            }
            if (n & 1) {
                q[0] += x[0] * y[0];
            }
        }

#   else

        while (y < e) {
            q = z;
            f = z + (n&~3);
            while (q < f) { 
                q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
                q[1] = (q[1] + x[1] * y[0]) + x[n+1] * y[1];
                q[2] = (q[2] + x[2] * y[0]) + x[n+2] * y[1];
                q[3] = (q[3] + x[3] * y[0]) + x[n+3] * y[1];
                x += 4;
                q += 4;
            }
            if (n & 2) {
                q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
                q[1] = (q[1] + x[1] * y[0]) + x[n+1] * y[1];
                x += 2;
                q += 2;
            }
            if (n & 1) {
                q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
                x += 1;
            }
            x += n;
            y += 2;
        }

        if (k & 1) {
            q = z;
            f = z + (n&~3);
            while (q < f) { 
                q[0] += x[0] * y[0];
                q[1] += x[1] * y[0];
                q[2] += x[2] * y[0];
                q[3] += x[3] * y[0];
                x += 4;
                q += 4;
            }
            if (n & 2) { 
                q[0] += x[0] * y[0];
                q[1] += x[1] * y[0];
                x += 2;
                q += 2;
            }
            if (n & 1) {
                q[0] += x[0] * y[0];
            }
        }

#   endif
}


/* Product of an n x k matrix (x) and a k x m matrix (y) with result stored 
   in z. 

   The inner loop does two matrix-vector products each time, implemented 
   much as in matprod_mat_vec above, except for computing two columns. This
   gives a reasonably efficient implementation of an outer product (where
   k is one).

   The case of n=2 may be handled specially, accumulating sums in two
   local variables rather than in a column of the result, and then storing
   them in the result column at the end.  Whether this is done can be 
   controlled using -DALT_MATPROD.  Change #ifdef to #ifndef or 
   vice versa below to change the default. */

void matprod_mat_mat (double * MATPROD_RESTRICT x, 
                      double * MATPROD_RESTRICT y, 
                      double * MATPROD_RESTRICT z, int n, int k, int m)
{
    if (n <= 0) return;

#   if CAN_ASSUME_ALIGNED
        x = __builtin_assume_aligned (x, ALIGN, ALIGN_OFFSET);
        y = __builtin_assume_aligned (y, ALIGN, ALIGN_OFFSET);
        z = __builtin_assume_aligned (z, ALIGN, ALIGN_OFFSET);
#   endif

#   ifndef ALT_MATPROD

        if (n == 2) { /* Treated specially */
    
            /* If m is odd, compute the first column of the result, and
               update y, z, and m accordingly. */
    
            if (m & 1) {
    
                double s1, s2;    /* sums for a column of the result */
                double *r = x;    /* r set to x, and then incremented */
                double *e = y+k;  /* stop when y reaches here */
    
                /* Initialize s1 and s2 to zero, if k is even, or to the 
                   products of the first element of the first column of y with
                   the first column of x.  Adjust r and y accordingly. */
    
                if (k & 1) {
                    double b = *y++;
                    s1 = *r++ * b;
                    s2 = *r++ * b;
                }
                else
                    s1 = s2 = 0.0;
    
                /* Each time around this loop, add the products of two columns
                   of x and two elements of the first column of y to s1 and s2.
                   Adjust r and y to account for this.  Note that e-y will 
                   be even when we start. */
    
                while (y < e) {
                    double b1 = y[0];
                    double b2 = y[1];
                    s1 = (s1 + (r[0] * b1)) + (r[2] * b2);
                    s2 = (s2 + (r[1] * b1)) + (r[3] * b2);
                    r += 4;
                    y += 2;
                }
    
                /* Store s1 and s2 in the result column. */
    
                z[0] = s1;
                z[1] = s2;
    
                /* Move to next column of the result. */
    
                z += 2;
                m -= 1;
    
            }
    
            /* Compute two columns of the result each time around this loop, 
               updating y, z, and m accordingly.  Note that m is now even. */
    
            while (m > 0) {
    
                double s11, s12, s21, s22;
                double *y2 = y + k;
                double *e = y2;     /* where y should stop */
                double *r = x;      /* r set to x, and then incrementd */
    
                /* Initialize sums for columns to zero, if k is even, or to the
                   products of the first elements of the next two columns of
                   y with the first column of x. Adjust r and y accordingly. */
    
                if (k & 1) {
                    double b1 = *y++;
                    double b2 = *y2++;
                    s11 = r[0] * b1;
                    s12 = r[1] * b1;
                    s21 = r[0] * b2;
                    s22 = r[1] * b2;
                    r += 2;
                }
                else
                    s11 = s12 = s21 = s22 = 0.0;
    
                /* Each time around this loop, add the products of two columns
                   of x with elements of the next two columns of y to the sums.
                   Adjust r and y to account for this. Note that e-y will
                   be even.*/
    
                while (y < e) {
                    double b11 = y[0];
                    double b12 = y[1];
                    double b21 = y2[0];
                    double b22 = y2[1];
                    s11 = (s11 + (r[0] * b11)) + (r[2] * b12);
                    s12 = (s12 + (r[1] * b11)) + (r[3] * b12);
                    s21 = (s21 + (r[0] * b21)) + (r[2] * b22);
                    s22 = (s22 + (r[1] * b21)) + (r[3] * b22);
                    r += 4;
                    y2 += 2;
                    y += 2;
                }
    
                /* Store sums in the next two result columns. */
    
                z[0] = s11;
                z[1] = s12;
                z[2] = s21;
                z[3] = s22;

                /* Move forward two to next column of the result and the next
                   column of y. */

                y = y2;
                z += 4;
                m -= 2;
            }
    
            return;
        }

#   endif

    /* If m is odd, compute the first column of the result, updating y, z, and 
       m to account for this column having been computed (so that the situation
       is the same as if m had been even to start with). */

    if (m & 1) {

        double *r = x;    /* r set to x, then modified as columns of x summed */
        double *e = y+k;  /* where y should stop */

        /* Initialize sums in z to zero, if k is even, or to the product of
           the first element of the next column of y with the first column 
           of x (in which case adjust r and y accordingly). */

        if (k & 1) {
            double b = *y++;
            double *q = z;
            double *f = z+n;
            do { *q++ = *r++ * b; } while (q < f);
        }
        else {
            double *q = z;
            double *f = z+n;
            do { *q++ = 0.0; } while (q < f);
        }

        /* Each time around this loop, add the products of two columns of x 
           with two elements of the next column of y to the result vector, z.
           Adjust r and y to account for this.  Note that e-y will be even 
           when we start. */

        while (y < e) {
            double *q = z;
            double *f = z+n;
            double b1 = y[0];
            double b2 = y[1];
            do {
                *q = (*q + (*r * b1)) + (*(r+n) * b2);
                r += 1;
                q += 1;
            } while (q < f);
            r += n;
            y += 2;
        }

        /* Move to next column of the result. */

        z += n;
        m -= 1;
    }

    /* Compute two columns of the result each time around this loop, updating
       y, z, and m accordingly.  Note that m will be even.  (At the start
       of each loop iteration, the work remaining to be done is the same as 
       if y, z, and m (and x, n, and k, which don't change) had been the 
       original arguments.) */

    while (m > 0) {

        double *y2 = y + k;
        double *e = y2; /* where y should stop */
        double *r = x;  /* r set to x, then modified as columns of x summed */

        /* Initialize sums in next two columns of z to zero, if k is even, 
           or to the products of the first elements of the next two columns
           of y with the first column of x, if k is odd (in which case adjust 
           r and y accordingly). */

        if (k & 1) {
            double b1 = *y++;
            double b2 = *y2++;
            double *q = z;
            double *f = x+n;
            do { *q++ = *r++ * b1; } while (r < f);
            r = x;
            do { *q++ = *r++ * b2; } while (r < f);
        }
        else {
            double *q = z;
            double *f = z+2*n;
            do { *q++ = 0.0; } while (q < f);
        }

        /* Each time around this loop, add the products of two columns of x 
           with two elements of the next two columns of y to the next two
           columns of the result vector, z.  Adjust r and y to account 
           for this.  Note that e-y will be even. */

        while (y < e) {
            double *q = z;
            double *f = z+n;
            double b11 = y[0];
            double b12 = y[1];
            double b21 = y2[0];
            double b22 = y2[1];
            do {
                double s1 = *r;
                double s2 = *(r+n);
                *q = (*q + (s1 * b11)) + (s2 * b12);
                *(q+n) = (*(q+n) + (s1 * b21)) + (s2 * b22);
                r += 1;
                q += 1;
            } while (q < f);
            r += n;
            y2 += 2;
            y += 2;
        }

        /* Move to the next two columns. */

        y = y2;
        z += 2*n;
        m -= 2;
    }
}


/* Product of the transpose of a k x n matrix (x) and a k x m matrix (y) 
   with result stored in z.  

   Each element of the result is the dot product of a column of x and a
   column of y.  The result is computed two columns at a time, which 
   allows the memory accesses to columns of x to be used for two such
   dot products (with two columns of y).  Two columns of x are also done 
   at once, again so memory accesses can be re-used.  The result is that,
   except perhaps for the first column or first element in a column, four
   elements of the result are computed at a time, using four accesses to
   columns of x and y (half the number of accesses that would be needed
   for doing four dot products in the obvious way).

   When the two operands are the same, the result will be a symmetric
   matrix.  After computation of each column or pair of columns, they are
   copied to the corresponding rows; hence each column need be computed
   only from the diagonal element down.

   There is no alternate implementation for this procedure.
*/

void matprod_trans1 (double * MATPROD_RESTRICT x, 
                     double * MATPROD_RESTRICT y, 
                     double * MATPROD_RESTRICT z, int n, int k, int m)
{
    if (n <= 0) return;

#   if CAN_ASSUME_ALIGNED
        x = __builtin_assume_aligned (x, ALIGN, ALIGN_OFFSET);
        y = __builtin_assume_aligned (y, ALIGN, ALIGN_OFFSET);
        z = __builtin_assume_aligned (z, ALIGN, ALIGN_OFFSET);
#   endif

    int sym = x==y && n==m;  /* same operands, so symmetric result? */
    int j = 0;               /* number of columns of result produced so far */

    /* Set result to zeros if k is zero. */

    if (k <= 0) {
        double *e = z + n*m;
        while (z < e) *z++ = 0;
        return;
    }

    /* If m is odd, compute the first column of the result, updating y, z, and 
       m to account for this column having been computed (so that the situation
       is the same as if m had been even to start with).  If the result is
       symmetric, also copy the first column to the first row. */

    if (m & 1) {

        double *r = x;
        double *e = z+n;
        double *rz = z;

        /* If n is odd, compute the first element of the first column of the
           result here.  Also, move r to point to the second column of x, and
           increment z.  For use if result is symmetric, advance rz to second
           element of the first row (no need to copy 1st element to itself). */

        if (n & 1) {
            double s = 0;
            double *q = y;
            double *e = y+k;
            do { s += *r++ * *q++; } while (q < e);
            *z++ = s;
            rz += n;
        }

        /* Compute the remainder of the first column of the result two
           elements at a time (looking at two columns of x).  Note that 
           e-z will be even.  If result is symmetric, copy elements to
           the first row as well. */

        while (z < e) {
            double s0 = 0;
            double s1 = 0;
            double *q = y;
            double *f = y+k;
            do {
                double t = *q;
                s0 += *r * t;
                s1 += *(r+k) * t;
                r += 1;
                q += 1;
            } while (q < f);
            r += k;
            *z++ = s0;
            *z++ = s1;
            if (sym) {
                *rz = s0;
                rz += n;
                *rz = s1;
                rz += n;
            }
        }

        y += k;
        j += 1;
    }

    /* Compute two columns of the result each time around this loop, updating
       y, z, and j accordingly.  Note that m-j will be even. */

    while (j < m) {

        double *z2 = z+n;
        double *e = z2;
        double *r = x;
        int nn = n;
        double *rz;

        /* If the result is symmetric, skip down to the diagonal element
           of the first column.  Also, let nn be the number of elements to 
           compute for these column, and set r to the start of the column
           of x to use. */
           
        if (sym) {
            z += j;
            z2 += j;
            nn -= j;
            r += j*k;
            rz = z;
        }

        /* If an odd number of elements are to be computed in the two columns,
           compute the first elements here.  Also, if result is symmetric,
           advance rz (but no need to store, since it would be redundant). */

        if (nn & 1) {
            double s0 = 0;
            double s1 = 0;
            double *q = y;
            double *f = y+k;
            do {
                double t = *r++;
                s0 += t * *q;
                s1 += t * *(q+k);
                q += 1;
            } while (q < f);
            *z++ = s0;
            *z2++ = s1;
            if (sym) rz += n;
        }

        /* Compute the remainder of the two columns of the result, two elements
           at a time.  Copy them to the corresponding rows too, if the result
           is symmetric. */

        while (z < e) {
#           if CAN_USE_AVX
                __m256d S = _mm256_setzero_pd();
                double *q = y;
                int i = k;
                do {
                    __m256d X = _mm256_set_pd (r[k], r[0], r[k], r[0]);
                    __m256d Y = _mm256_set_pd (q[k], q[k], q[0], q[0]);
                    S = _mm256_add_pd (_mm256_mul_pd(X,Y), S);
                    r += 1;
                    q += 1;
                } while (--i > 0);
                __m128d H = _mm256_extractf128_pd(S,1);
                _mm_storeu_pd (z, _mm256_extractf128_pd(S,0));
                _mm_storeu_pd (z2, H);
                if (sym) {
                    _mm_storeu_pd (rz, 
                       _mm_unpacklo_pd(_mm256_extractf128_pd(S,0),H));
                    rz += n;
                    _mm_storeu_pd (rz, 
                       _mm_unpackhi_pd(_mm256_extractf128_pd(S,0),H));
                    rz += n;
                }
                z += 2;
                z2 += 2;
                r += k;
#           else
                double s[4] = { 0, 0, 0, 0 };
                double *q = y;
                int i = k;
                do {
                    double t = *r;
                    double t2 = *(r+k);
                    double u = *q;
                    double u2 = *(q+k);
                    s[0] += t * u;
                    s[1] += t * u2;
                    s[2] += t2 * u;
                    s[3] += t2 * u2;
                    r += 1;
                    q += 1;
                } while (--i > 0);
                *z++ = s[0];
                *z2++ = s[1];
                *z++ = s[2];
                *z2++ = s[3];
                if (sym) {
                    rz[0] = s[0];
                    rz[1] = s[1];
                    rz[n] = s[2];
                    rz[n+1] = s[3];
                    rz += 2*n;
                }
                r += k;
#           endif
        }

        /* Go on to next two columns of y. */

        z = z2;
        y += 2*k;
        j += 2;
    }
}


/* Product of an n x k matrix (x) and the transpose of an m x k matrix (y) 
   with result stored in z.

   When the two operands are the same, the result will be a symmetric
   matrix.  Only the lower-triangular part of the result is computed,
   with the elements in columns that are computed then being copied to 
   the corresponding elements in rows above the diagonal.

   The case of n=2 may be handled specially, accumulating sums in two
   local variables rather than in a column of the result, and then storing
   them in the result column at the end.  Whether this is done can be 
   controlled using -DALT_MATPROD_TRANS2.  Change #ifdef to #ifndef or
   vice versa below to change the default. */

void matprod_trans2 (double * MATPROD_RESTRICT x, 
                     double * MATPROD_RESTRICT y, 
                     double * MATPROD_RESTRICT z, int n, int k, int m)
{
    if (n <= 0) return;

#   if CAN_ASSUME_ALIGNED
        x = __builtin_assume_aligned (x, ALIGN, ALIGN_OFFSET);
        y = __builtin_assume_aligned (y, ALIGN, ALIGN_OFFSET);
        z = __builtin_assume_aligned (z, ALIGN, ALIGN_OFFSET);
#   endif

    int sym = x==y && n==m;  /* same operands, so symmetric result? */
    double *ex = x + n*k;    /* point past end of x */
    int j = 0;               /* number of columns of result produced so far */

#   ifndef ALT_MATPROD_MAT_TRANS2
    {
        if (n == 2) {
    
            /* If m is odd, compute the first column of the result, and
               update y, z, and mt accordingly. */
    
            if (m & 1) {
    
                double s1, s2;
                double *q = y;
                double *r = x;
                double *e = x+2*k;
    
                /* Initialize s1 and s2 to zero, if k is even, or to the 
                   products of the first element of the first row of y with
                   the first column of x.  Adjust r and q accordingly. */
    
                if (k & 1) {
                    double b = *q;
                    s1 = *r++ * b;
                    s2 = *r++ * b;
                    q += m;
                }
                else
                    s1 = s2 = 0.0;
    
                /* Each time around this loop, add the products of two columns
                   of x and two elements of the first row of y to s1 and s2.
                   Adjust r and q to account for this. */
    
                while (r < e) {
                    double b1, b2;
                    b1 = *q;
                    q += m;
                    b2 = *q;
                    q += m;
                    s1 = (s1 + (r[0] * b1)) + (r[2] * b2);
                    s2 = (s2 + (r[1] * b1)) + (r[3] * b2);
                    r += 4;
                }
    
                /* Store s1 and s2 in the result column. */
    
                z[0] = s1;
                z[1] = s2;
    
                /* Move to next column of the result, and next row of y. */

                z += 2;
                y += 1;
                j += 1;
            }
    
            /* Compute two columns of the result each time around this loop, 
               updating y, z, and j accordingly.  Note that m-j is now even. */
    
            while (j < m) {
    
                double s11, s12, s21, s22;
                double *q = y;
                double *r = x;

                /* Initialize sums for columns to zero, if k is even, or to the 
                   products of the first elements of the next two rows of y with
                   the first column of x.  Adjust r and q accordingly. */
    
                if (k & 1) {
                    double b = *q;
                    double b2 = *(q+1);
                    q += m;
                    s11 = r[0] * b;
                    s12 = r[1] * b;
                    s21 = r[0] * b2;
                    s22 = r[1] * b2;
                    r += 2;
                }
                else
                    s11 = s12 = s21 = s22 = 0.0;
    
                /* Each time around this loop, add the products of two columns
                   of x with elements of the next two rows of y to the sums.
                   Adjust r and q to account for this. */
    
                while (r < ex) {
                    double b11, b12, b21, b22;
                    b11 = *q;
                    b21 = *(q+1);
                    q += m;
                    b12 = *q;
                    b22 = *(q+1);
                    q += m;
                    s11 = (s11 + (r[0] * b11)) + (r[2] * b12);
                    s12 = (s12 + (r[1] * b11)) + (r[3] * b12);
                    s21 = (s21 + (r[0] * b21)) + (r[2] * b22);
                    s22 = (s22 + (r[1] * b21)) + (r[3] * b22);
                    r += 4;
                }
    
                /* Store sums in the next two result columns. */
    
                z[0] = s11;
                z[1] = s12;
                z[2] = s21;
                z[3] = s22;

                /* Move forward two to the next column of the result and 
                   the next row of y. */

                z += 4;
                y += 2;
                j += 2;
            }
    
            return;
        }
    }
#   endif

    /* If m is odd, compute the first column of the result, updating y, z, and 
       j to account for this column having been computed.  Also, if result
       is symmetric, copy this column to the first row. */

    if (m & 1) {

        double *q = y;
        double *r = x;
        double *ez = z+n;

        /* Initialize sums in z to zero, if k is even, or to the product of
           the first element of the first row of y with the first column 
           of x (in which case adjust r and q accordingly). */

        if (k & 1) {
            double *t = z;
            double *f = z+n;
            double b = *q;
            do { *t++ = *r++ * b; } while (t < f);
            q += m;
        }
        else {
            double *t = z;
            double *f = z+n;
            do { *t++ = 0.0; } while (t < f);
        }

        /* Each time around this loop, add the products of two columns of x 
           with two elements of the first row of y to the result vector, z.
           Adjust r and y to account for this. */

        while (r < ex) {
            double *t = z;
            double b1, b2;
            b1 = *q;
            q += m;
            b2 = *q;
            q += m;
            do {
                *t = (*t + (*r * b1)) + (*(r+n) * b2);
                r += 1;
                t += 1;
            } while (t < ez);
            r += n;
        }

        /* Copy first column to first row, if result is symmetric. */

        if (sym) {
            double *t = z+1;
            double *q = z+n;
            while (t < ez) {
                *q = *t;
                t += 1;
                q += n;
            }
        }

        /* Move to next column of the result and the next row of y. */

        z += n;
        y += 1;
        j += 1;
    }

    /* Compute two columns of the result each time around this loop, updating
       y, z, and j accordingly.  Note that m-j will be even.  If the result
       is symmetric, only the parts of the columns at and below the diagonal
       are computed (except one element above the diagonal is computed for
       the second column), and these parts are then copied to the corresponding 
       rows. */

    while (j < m) {

        double *zs = sym ? z+j : z;   /* Where to start storing sums */
        double *ez = z+n;             /* Where we stop storing sums */
        double *xs = x;
        double *t1 = zs;
        double *t2 = t1 + n;
        double *q = y;

        /* Initialize sums in the next two columns of z to zero, if k is 
           even, or to the products of the first elements of the next two
           rows of y with the first column of x (in which case adjust r and 
           q accordingly). */

        if (k & 1) {
            double b1 = *q;
            double b2 = *(q+1);
            double *r = sym ? xs+j : xs;
            do {
                double s = *r++;
                *t1++ = s * b1;
                *t2++ = s * b2;
            } while (t1 < ez);
            xs += n;
            q += m;
        }
        else {
            do {
                *t1++ = 0;
                *t2++ = 0;
            } while (t1 < ez);
        }

        /* Each time around this loop, add the products of two columns of x 
           with elements of the next two rows of y to the next two columns
           the result vector, z.  Adjust r and y to account for this. */

        while (xs < ex) {
            double b11, b12, b21, b22;
            double *t1 = zs;
            double *t2 = t1 + n;
            double *r = sym ? xs+j : xs;
            b11 = *q;
            b21 = *(q+1);
            q += m;
            b12 = *q;
            b22 = *(q+1);
            q += m;
            do {
                double s1 = *r;
                double s2 = *(r+n);
                *t1 = (*t1 + (s1 * b11)) + (s2 * b12);
                *t2 = (*t2 + (s1 * b21)) + (s2 * b22);
                t1 += 1;
                t2 += 1;
                r += 1;
            } while (t1 < ez);
            xs += 2*n;
        }

        /* If the result is symmetric, copy the columns just computed
           to the corresponding rows. */

        if (sym) {
            double *t1 = zs + 2;
            double *t2 = t1 + n;
            double *q = zs + 2*n;
            while (t1 < ez) {
                q[0] = *t1;
                q[1] = *t2;
                t1 += 1;
                t2 += 1;
                q += n;
            }
        }

        /* Move forward two, to the next column of the result and the
           next row of y. */

        z += 2*n;
        y += 2;
        j += 2;
    }
}
