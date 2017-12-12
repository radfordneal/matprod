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

#if ALIGN < 0 || (ALIGN & (ALIGN-1)) != 0
#error "alignment must be a power of two"
#endif

#if ALIGN_OFFSET >= ALIGN
#error "alignment offset must be less than alignment"
#endif

#if ALIGN_OFFSET % 8 != 0
#error "alignment offset must be a multiple of eight"
#endif

#define ALIGN_FORWARD ((ALIGN - ALIGN_OFFSET) % ALIGN)

#if ALIGN >= 8 && __GNUC__
#define CAN_ASSUME_ALIGNED 1
#define ASSUME_ALIGNED(x,a,o) __builtin_assume_aligned(x,a,o);
#else
#define CAN_ASSUME_ALIGNED 0
#define ASSUME_ALIGNED(x,a,o) (x)
#endif

#if 0  /* Enable for debug check */
#   include <stdint.h>
#   include <stdlib.h>
#   define CHK_ALIGN(p) \
      do { if (((uintptr_t)(p)&(ALIGN-1)) != ALIGN_OFFSET) abort(); } while (0)
#else
#   define CHK_ALIGN(p) \
      do {} while (0)
#endif


/* Set up SIMD definitions. */

#if __SSE2__ && !defined(DISABLE_SIMD_CODE)
#define CAN_USE_SSE2 1
#else
#define CAN_USE_SSE2 0
#endif

#if __SSE3__ && !defined(DISABLE_SIMD_CODE)
#define CAN_USE_SSE3 1
#else
#define CAN_USE_SSE3 0
#endif

#if __AVX__ && !defined(DISABLE_SIMD_CODE) && !defined(DISABLE_AVX_CODE)
#define CAN_USE_AVX 1
#else
#define CAN_USE_AVX 0
#endif

#if CAN_USE_SSE2 || CAN_USE_SSE3 || CAN_USE_AVX
#include <immintrin.h>
#endif


/* Versions of load and store that take advantage of known alignment.
   The loadA and storeA macros do an aligned load/store if ALIGN is
   suitably large, assuming that any offset has been compensated for.
   The loadAA and storeAA macro do an unalign load/store only if ALIGN
   is suitably large and ALIGN_OFFSET is zero, as is appropriate for
   an address that is one of the arguments plus a multiple of of ALIGN. */

#define _mm_loadA_pd(w) \
   (ALIGN>=16 ? _mm_load_pd(w) : _mm_loadu_pd(w))
#define _mm_storeA_pd(w,v) \
   (ALIGN>=16 ? _mm_store_pd(w,v) : _mm_storeu_pd(w,v))
#define _mm256_loadA_pd(w) \
   (ALIGN>=32 ? _mm256_load_pd(w) : _mm256_loadu_pd(w))
#define _mm256_storeA_pd(w,v) \
   (ALIGN>=32 ? _mm256_store_pd(w,v) : _mm256_storeu_pd(w,v))

#define _mm_loadAA_pd(w) \
   (ALIGN>=16 && ALIGN_OFFSET%16==0 ? _mm_load_pd(w) : _mm_loadu_pd(w))
#define _mm_storeAA_pd(w,v) \
   (ALIGN>=16 && ALIGN_OFFSET%16==0 ? _mm_store_pd(w,v) : _mm_storeu_pd(w,v))
#define _mm256_loadAA_pd(w) \
   (ALIGN>=32 && ALIGN_OFFSET==0 ? _mm256_load_pd(w) : _mm256_loadu_pd(w))
#define _mm256_storeAA_pd(w,v) \
   (ALIGN>=32 && ALIGN_OFFSET==0 ? _mm256_store_pd(w,v) : _mm256_storeu_pd(w,v))


/* Set vector z of length n to all zeros.  This is a degenerate special
   case of other operations. */

static inline void set_to_zeros (double * MATPROD_RESTRICT z, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        z[i] = 0.0;
    }
}


/* Multiply vector x of length n by scalar s, storing result in vector z. 
   This is a degenerate special case of other operations. */

static inline void scalar_multiply (double s, 
                                    double * MATPROD_RESTRICT x,
                                    double * MATPROD_RESTRICT z, int n)
{
    int i;
    for (i = 0; i < n; i++) {
        z[i] = s * x[i];
    }
}


/* Dot product of two vectors of length k. 

   An unrolled loop is used that adds four products to the sum each
   iteration, perhaps using SSE2 or AVX instructions. */

double matprod_vec_vec (double * MATPROD_RESTRICT x, 
                        double * MATPROD_RESTRICT y, int k)
{
    /* Handle k = 0, 1, or 2 specially. */

    if (k <= 2) {
        if (k == 2)
            return x[0] * y[0] + x[1] * y[1];
        if (k == 1) 
            return x[0] * y[0];
        else  /* k <= 0 */
            return 0.0;
    }

    CHK_ALIGN(x); CHK_ALIGN(y);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);

    double s;
    int i;

    /* Initialize the sum, s, perhaps doing a few products to improve alignment.
       Set i to the number of products done.  Note that k is at least 3 here. */

#   if (ALIGN_FORWARD & 8)
        s = x[0] * y[0];
        i = 1;
#   else
        s = 0.0;
        i = 0;
#   endif

#   if (ALIGN_FORWARD & 16)
        s += x[i] * y[i];
        s += x[i+1] * y[i+1];
        i += 2;
#   endif

    /* Use an unrolled loop to add (most) remaining products to s,
       perhaps using SSE2 or AVX instructions.  A possible final
       product is handled after main code below. */

    int k2 = k-i;  /* number of products left to do */

#   if CAN_USE_AVX
    {
        __m128d S = _mm_load_sd(&s);
        __m256d AA;
        __m128d A;
        while (i < k-7) {
            AA = _mm256_mul_pd (_mm256_loadA_pd(x+i), _mm256_loadA_pd(y+i));
            A = _mm256_castpd256_pd128(AA);
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            A = _mm256_extractf128_pd(AA,1);
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            AA = _mm256_mul_pd (_mm256_loadA_pd(x+i+4), _mm256_loadA_pd(y+i+4));
            A = _mm256_castpd256_pd128(AA);
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            A = _mm256_extractf128_pd(AA,1);
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            i += 8;
        }
        if (k2 & 4) {
            AA = _mm256_mul_pd (_mm256_loadA_pd(x+i), _mm256_loadA_pd(y+i));
            A = _mm256_castpd256_pd128(AA);
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            A = _mm256_extractf128_pd(AA,1);
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            i += 4;
        }
        if (k2 & 2) {
            __m128d A;
            A = _mm_mul_pd (_mm_loadA_pd(x+i), _mm_loadA_pd(y+i));
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            i += 2;
        }
        if (k2 & 1) {
            __m128d A;
            A = _mm_mul_sd (_mm_load_sd(x+i), _mm_load_sd(y+i));
            S = _mm_add_sd (A, S);
        }
        _mm_store_sd (&s, S);
    }

#   elif CAN_USE_SSE2
    {
        __m128d S = _mm_load_sd(&s);
        __m128d A;
        while (i < k-3) {
            A = _mm_mul_pd (_mm_loadA_pd(x+i), _mm_loadA_pd(y+i));
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            A  = _mm_mul_pd (_mm_loadA_pd(x+i+2), _mm_loadA_pd(y+i+2));
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            i += 4;
        }
        if (k2 & 2) {
            A = _mm_mul_pd (_mm_loadA_pd(x+i), _mm_loadA_pd(y+i));
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            i += 2;
        }
        if (k2 & 1) {
            A = _mm_mul_sd (_mm_load_sd(x+i), _mm_load_sd(y+i));
            S = _mm_add_sd (A, S);
        }
        _mm_store_sd (&s, S);
    }

#   else  /* non-SIMD code */
    {
        while (i < k-3) {
            s += x[i+0] * y[i+0];
            s += x[i+1] * y[i+1];
            s += x[i+2] * y[i+2];
            s += x[i+3] * y[i+3];
            i += 4;
        }
        if (k2 & 2) {
            s += x[i+0] * y[i+0];
            s += x[i+1] * y[i+1];
            i += 2;
        }
        if (k2 & 1) {
            s += x[i+0] * y[i+0];
        }
    }
#   endif

    return s;
}


/* Product of row vector (x) of length k and k x m matrix (y), result
   stored in z.

   Cases where k is 0, 1, or 2 are handled specially, perhaps using
   SSE3 or AVX instructions.  Otherwise, the inner loop is unrolled to
   do a set of four vector dot products, and these dot products are also
   done with loop unrolling, perhaps using SSE2 or AVX instructions. */

void matprod_vec_mat (double * MATPROD_RESTRICT x, 
                      double * MATPROD_RESTRICT y, 
                      double * MATPROD_RESTRICT z, int k, int m)
{
    if (m <= 0) return;

    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    /* Specially handle cases where y has two or fewer rows. */

    if (k <= 2) {

        if (k != 2) {
            if (k == 1)
                scalar_multiply (x[0], y, z, m);
            else  /* k == 0 */
                set_to_zeros (z, m);
        }
        else {  /* k == 2 */
#           if CAN_USE_AVX
            {
                __m256d T = _mm256_set_pd (x[1], x[0], x[1], x[0]);
#               if ALIGN_FORWARD & 8
                {
                    __m128d A = _mm_mul_pd (_mm256_castpd256_pd128(T),
                                            _mm_loadAA_pd(y));
                    _mm_store_sd (z, _mm_hadd_pd(A,A));
                    z += 1;
                    y += 2;
                    m -= 1;
                }
#               endif
                while (m >= 4) {
                     __m256d A = _mm256_mul_pd (T, _mm256_loadAA_pd(y));
                     __m256d B = _mm256_mul_pd (T, _mm256_loadAA_pd(y+4));
                     _mm256_storeAA_pd(z,_mm256_hadd_pd
                                         (_mm256_permute2f128_pd (A, B, 0x20),
                                          _mm256_permute2f128_pd (A, B, 0x31)));
                    y += 8;
                    z += 4;
                    m -= 4;
                }
                if (m > 1) {
                     __m128d A = _mm_mul_pd (_mm256_castpd256_pd128(T), 
                                             _mm_loadAA_pd(y));
                     __m128d B = _mm_mul_pd (_mm256_castpd256_pd128(T), 
                                             _mm_loadAA_pd(y+2));
                     _mm_storeA_pd (z, _mm_hadd_pd(A,B));
                    y += 4;
                    z += 2;
                    m -= 2;
                }
                if (m >= 1) {
                    __m128d A = _mm_mul_pd (_mm256_castpd256_pd128(T),
                                            _mm_loadAA_pd(y));
                    _mm_store_sd (z, _mm_hadd_pd(A,A));
                }
            }
#           elif CAN_USE_SSE3
            {
                __m128d T = _mm_set_pd (x[1], x[0]);
                __m128d A, B;
#               if ALIGN_FORWARD & 8
                     A = _mm_mul_pd (T, _mm_loadAA_pd(y));
                    _mm_store_sd (z, _mm_hadd_pd(A,A));
                    z += 1;
                    y += 2;
                    m -= 1;
#               endif
                while (m >= 4) {
                     A = _mm_mul_pd (T, _mm_loadAA_pd(y));
                     B = _mm_mul_pd (T, _mm_loadAA_pd(y+2));
                     _mm_storeA_pd (z, _mm_hadd_pd(A,B));
                     A = _mm_mul_pd (T, _mm_loadAA_pd(y+4));
                     B = _mm_mul_pd (T, _mm_loadAA_pd(y+6));
                     _mm_storeA_pd (z+2, _mm_hadd_pd(A,B));
                    y += 8;
                    z += 4;
                    m -= 4;
                }
                if (m > 1) {
                     A = _mm_mul_pd (T, _mm_loadAA_pd(y));
                     B = _mm_mul_pd (T, _mm_loadAA_pd(y+2));
                     _mm_storeA_pd (z, _mm_hadd_pd(A,B));
                    y += 4;
                    z += 2;
                    m -= 2;
                }
                if (m >= 1) {
                     A = _mm_mul_pd (T, _mm_loadAA_pd(y));
                    _mm_store_sd (z, _mm_hadd_pd(A,A));
                }
            }
#           else  /* non-SIMD code */
            {
                double t[2] = { x[0], x[1] };
                while (m >= 4) {
                    z[0] = t[0] * y[0] + t[1] * y[1];
                    z[1] = t[0] * y[2] + t[1] * y[3];
                    z[2] = t[0] * y[4] + t[1] * y[5];
                    z[3] = t[0] * y[6] + t[1] * y[7];
                    y += 8;
                    z += 4;
                    m -= 4;
                }
                if (m > 1) {
                    z[0] = t[0] * y[0] + t[1] * y[1];
                    z[1] = t[0] * y[2] + t[1] * y[3];
                    y += 4;
                    z += 2;
                    m -= 2;
                }
                if (m >= 1) {
                    z[0] = t[0] * y[0] + t[1] * y[1];
                }
            }
#           endif
        }

        return;
    }

    /* In this loop, compute four consecutive elements of the result vector,
       by doing four dot products of x with columns of y.  Adjust y, z, and
       m as we go. */

    while (m >= 4) {

        /* The various versions of the loop below all add two products
           to the sums for each of the four dot products, adjusting p
           and y as they go.  A possible final set of four products is
           then added afterwards.  The sums may be initialized to zero
           or to the result of one product, if that helps alignment 
           (k is at least one, zero k being handled above). */

        double *p;               /* Pointer that goes along pairs in x */

#       if CAN_USE_AVX
        {
            __m256d S, B;
            int k2;

#           if (ALIGN_FORWARD & 8)
                S = _mm256_set_pd (y[3*k], y[2*k], y[k], y[0]);
                S = _mm256_mul_pd (_mm256_set1_pd(x[0]), S);
                p = x+1;
                y += 1;
                k2 = k-1;
#           else
                S = _mm256_setzero_pd ();
                p = x;
                k2 = k;
#           endif

            while (k2 >= 4) {
                __m256d Y0, Y1, Y2, Y3;
                __m256d T0, T1;
                Y0 = _mm256_loadu_pd(y);
                Y1 = _mm256_loadu_pd(y+k);
                Y2 = _mm256_loadu_pd(y+2*k);
                Y3 = _mm256_loadu_pd(y+3*k);
                T0 = _mm256_permute2f128_pd (Y0, Y2, 0x20);
                T1 = _mm256_permute2f128_pd (Y1, Y3, 0x20);
                B = _mm256_unpacklo_pd (T0, T1);
                B = _mm256_mul_pd (_mm256_set1_pd(p[0]), B);
                S = _mm256_add_pd (B, S);
                B = _mm256_unpackhi_pd (T0, T1);
                B = _mm256_mul_pd (_mm256_set1_pd(p[1]), B);
                S = _mm256_add_pd (B, S);
                T0 = _mm256_permute2f128_pd (Y0, Y2, 0x31);
                T1 = _mm256_permute2f128_pd (Y1, Y3, 0x31);
                B = _mm256_unpacklo_pd (T0, T1);
                B = _mm256_mul_pd (_mm256_set1_pd(p[2]), B);
                S = _mm256_add_pd (B, S);
                B = _mm256_unpackhi_pd (T0, T1);
                B = _mm256_mul_pd (_mm256_set1_pd(p[3]), B);
                S = _mm256_add_pd (B, S);
                p += 4;
                y += 4;
                k2 -= 4;
            }

            if (k2 > 1) {
                __m128d Y0, Y1, Y2, Y3;
                __m256d T0, T1;
                Y0 = _mm_loadA_pd(y);
                Y1 = _mm_loadu_pd(y+k);
                Y2 = _mm_loadA_pd(y+2*k);
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
                k2 -= 2;
            }

            if (k2 >= 1) {
                B = _mm256_set_pd (y[3*k], y[2*k], y[k], y[0]);
                B = _mm256_mul_pd (_mm256_set1_pd(p[0]), B);
                S = _mm256_add_pd (B, S);
                y += 1;
            }

            y += 3*k;

           _mm256_storeu_pd (z, S);
        }

#       elif CAN_USE_SSE2 && ALIGN >= 16 /* works, but slower, when unaligned */
        {
            __m128d S0, S1, B, P;
            int k2;

#           if (ALIGN_FORWARD & 8)
            {
                P = _mm_set1_pd(x[0]);
                S0 = _mm_mul_pd (P, _mm_set_pd (y[k], y[0]));
                S1 = _mm_mul_pd (P, _mm_set_pd (y[3*k], y[2*k]));
                p = x+1;
                y += 1;
                k2 = k-1;
            }
#           else
            {
                S0 = _mm_setzero_pd ();
                S1 = _mm_setzero_pd ();
                p = x;
                k2 = k;
            }
#           endif

            if (k & 1) {  /* second column not aligned if first is */
                while (k2 > 1) {
                    __m128d T0, T1;
                    __m128d P = _mm_loadA_pd(p);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                    T1 = _mm_mul_pd (_mm_loadu_pd(y+k), P);
                    S0 = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S0);
                    S0 = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S0);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y+2*k), P);
                    T1 = _mm_mul_pd (_mm_loadu_pd(y+3*k), P);
                    S1 = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S1);
                    S1 = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S1);
                    p += 2;
                    y += 2;
                    k2 -= 2;
                }
            }
            else {  /* second column has same 16-byte alignment as first */
                while (k2 > 1) {
                    __m128d T0, T1;
                    __m128d P = _mm_loadA_pd(p);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                    T1 = _mm_mul_pd (_mm_loadA_pd(y+k), P);
                    S0 = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S0);
                    S0 = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S0);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y+2*k), P);
                    T1 = _mm_mul_pd (_mm_loadA_pd(y+3*k), P);
                    S1 = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S1);
                    S1 = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S1);
                    p += 2;
                    y += 2;
                    k2 -= 2;
                }
            }

            if (k2 >= 1) {
                __m128d B;
                __m128d P = _mm_set1_pd(p[0]);
                B = _mm_mul_pd (P, _mm_set_pd(y[k],y[0]));
                S0 = _mm_add_pd (B, S0);
                B = _mm_mul_pd (P, _mm_set_pd(y[3*k],y[2*k]));
                S1 = _mm_add_pd (B, S1);
                y += 1;
            }

            y += 3*k;

            _mm_storeu_pd (z, S0);
            _mm_storeu_pd (z+2, S1);
        }

#       else  /* non-SIMD code */
        {
            double s[4] = { 0, 0, 0, 0 };
            int k2 = k;
            p = x;
            while (k2 > 1) {
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
                k2 -= 2;
            }

            if (k2 >= 1) {
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

    /* Compute the final few dot products left over from the loop above. */

    if (m == 3) {  /* Do three more dot products */

        double *p;               /* Pointer that goes along pairs in x */
#       if CAN_USE_AVX
        {
            __m128d S, S2;
            int k2;

#           if (ALIGN_FORWARD & 8)
            {
                __m128d P = _mm_set1_pd(x[0]);
                S = _mm_mul_pd (P, _mm_set_pd (y[k], y[0]));
                S2 = _mm_mul_sd (P, _mm_set_sd (y[2*k]));
                p = x+1;
                y += 1;
                k2 = k-1;
            }
#           else
            {
                S = _mm_setzero_pd ();
                S2 = _mm_setzero_pd ();
                p = x;
                k2 = k;
            }
#           endif

            while (k2 >= 4) {
                __m256d P = _mm256_loadu_pd(p);
                __m256d T0 = _mm256_mul_pd (_mm256_loadu_pd(y), P);
                __m256d T1 = _mm256_mul_pd (_mm256_loadu_pd(y+k), P);
                __m256d T2 = _mm256_mul_pd (_mm256_loadu_pd(y+2*k), P);
                __m128d L2 = _mm256_castpd256_pd128 (T2);
                __m128d H2 = _mm256_extractf128_pd (T2, 1);
                __m256d L = _mm256_unpacklo_pd(T0,T1);
                __m256d H = _mm256_unpackhi_pd(T0,T1);
                S = _mm_add_pd (_mm256_castpd256_pd128(L), S);
                S = _mm_add_pd (_mm256_castpd256_pd128(H), S);
                S = _mm_add_pd (_mm256_extractf128_pd(L,1), S);
                S = _mm_add_pd (_mm256_extractf128_pd(H,1), S);
                S2 = _mm_add_sd (L2, S2);
                S2 = _mm_add_sd (_mm_unpackhi_pd(L2,L2), S2);
                S2 = _mm_add_sd (H2, S2);
                S2 = _mm_add_sd (_mm_unpackhi_pd(H2,H2), S2);
                p += 4;
                y += 4;
                k2 -= 4;
            }

            if (k2 > 1) {
                __m128d P = _mm_loadA_pd(p);
                __m128d T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                __m128d T1 = _mm_mul_pd (_mm_loadu_pd(y+k), P);
                __m128d T2 = _mm_mul_pd (_mm_loadA_pd(y+2*k), P);
                S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                S2 = _mm_add_sd (T2, S2);
                S2 = _mm_add_sd (_mm_unpackhi_pd(T2,T2), S2);
                p += 2;
                y += 2;
                k2 -= 2;
            }

            if (k2 >= 1) {
                __m128d P = _mm_set1_pd(p[0]);
                __m128d B;
                B = _mm_mul_pd (P, _mm_set_pd(y[k],y[0]));
                S = _mm_add_pd (B, S);
                B = _mm_mul_sd (P, _mm_set_sd (y[2*k]));
                S2 = _mm_add_sd (B, S2);
                y += 1;
            }

           _mm_storeu_pd (z, S);
           _mm_store_sd (z+2, S2);
        }

#       elif CAN_USE_SSE2 /* && ALIGN >= 16 works, but slower?, when unaligned*/
        {
            __m128d S, S2;
            int k2;

#           if (ALIGN_FORWARD & 8)
            {
                __m128d P = _mm_set1_pd(x[0]);
                S = _mm_mul_pd (P, _mm_set_pd (y[k], y[0]));
                S2 = _mm_mul_sd (P, _mm_set_sd (y[2*k]));
                p = x+1;
                y += 1;
                k2 = k-1;
            }
#           else
            {
                S = _mm_setzero_pd ();
                S2 = _mm_setzero_pd ();
                p = x;
                k2 = k;
            }
#           endif

            if (k & 1) {  /* second column not aligned if first is */
                while (k2 >= 4) {
                    __m128d P, T0, T1, T2;
                    P = _mm_loadA_pd(p);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                    T1 = _mm_mul_pd (_mm_loadu_pd(y+k), P);
                    T2 = _mm_mul_pd (_mm_loadA_pd(y+2*k), P);
                    S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                    S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                    S2 = _mm_add_sd (T2, S2);
                    S2 = _mm_add_sd (_mm_unpackhi_pd(T2,T2), S2);
                    P = _mm_loadA_pd(p+2);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y+2), P);
                    T1 = _mm_mul_pd (_mm_loadu_pd(y+k+2), P);
                    T2 = _mm_mul_pd (_mm_loadA_pd(y+2*k+2), P);
                    S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                    S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                    S2 = _mm_add_sd (T2, S2);
                    S2 = _mm_add_sd (_mm_unpackhi_pd(T2,T2), S2);
                    p += 4;
                    y += 4;
                    k2 -= 4;
                }
           }
           else {  /* second column has same 16-byte alignment as first */
                while (k2 >= 4) {
                    __m128d P, T0, T1, T2;
                    P = _mm_loadA_pd(p);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                    T1 = _mm_mul_pd (_mm_loadA_pd(y+k), P);
                    T2 = _mm_mul_pd (_mm_loadA_pd(y+2*k), P);
                    S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                    S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                    S2 = _mm_add_sd (T2, S2);
                    S2 = _mm_add_sd (_mm_unpackhi_pd(T2,T2), S2);
                    P = _mm_loadA_pd(p+2);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y+2), P);
                    T1 = _mm_mul_pd (_mm_loadA_pd(y+k+2), P);
                    T2 = _mm_mul_pd (_mm_loadA_pd(y+2*k+2), P);
                    S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                    S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                    S2 = _mm_add_sd (T2, S2);
                    S2 = _mm_add_sd (_mm_unpackhi_pd(T2,T2), S2);
                    p += 4;
                    y += 4;
                    k2 -= 4;
                }
           }

           if (k2 > 1) {
                __m128d P = _mm_loadA_pd(p);
                __m128d T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                __m128d T1 = _mm_mul_pd (_mm_loadu_pd(y+k), P);
                __m128d T2 = _mm_mul_pd (_mm_loadA_pd(y+2*k), P);
                S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                S2 = _mm_add_sd (T2, S2);
                S2 = _mm_add_sd (_mm_unpackhi_pd(T2,T2), S2);
                p += 2;
                y += 2;
                k2 -= 2;
            }

            if (k2 >= 1) {
                __m128d P = _mm_set1_pd(p[0]);
                __m128d B;
                B = _mm_mul_pd (P, _mm_set_pd(y[k],y[0]));
                S = _mm_add_pd (B, S);
                B = _mm_mul_sd (P, _mm_set_sd (y[2*k]));
                S2 = _mm_add_sd (B, S2);
                y += 1;
            }

           _mm_storeu_pd (z, S);
           _mm_store_sd (z+2, S2);
        }

#       else  /* non-SIMD code */
        {
            double s[3] = { 0, 0, 0 };
            int k2 = k;

            p = x;

            while (k2 > 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[2] += p[0] * y[2*k];
                s[0] += p[1] * y[1];
                s[1] += p[1] * y[k+1];
                s[2] += p[1] * y[2*k+1];
                p += 2;
                y += 2;
                k2 -= 2;
            }

            if (k2 >= 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[2] += p[0] * y[2*k];
                y += 1;
            }

            z[0] = s[0];
            z[1] = s[1];
            z[2] = s[2];
        }
#       endif
    }

    else if (m == 2) {  /* Do two more dot products */

        double *p;               /* Pointer that goes along pairs in x */

#       if CAN_USE_AVX
        {
            __m128d S;
            int k2;

#           if (ALIGN_FORWARD & 8)
            {
                S = _mm_mul_pd (_mm_set1_pd(x[0]), _mm_set_pd (y[k], y[0]));
                p = x+1;
                y += 1;
                k2 = k-1;
            }
#           else
            {
                S = _mm_setzero_pd ();
                p = x;
                k2 = k;
            }
#           endif

            while (k2 >= 4) {
                __m256d P = _mm256_loadAA_pd(p);
                __m256d T0 = _mm256_mul_pd (_mm256_loadAA_pd(y), P);
                __m256d T1 = _mm256_mul_pd (_mm256_loadAA_pd(y+k), P);
                __m256d L = _mm256_unpacklo_pd(T0,T1);
                __m256d H = _mm256_unpackhi_pd(T0,T1);
                S = _mm_add_pd (_mm256_castpd256_pd128(L), S);
                S = _mm_add_pd (_mm256_castpd256_pd128(H), S);
                S = _mm_add_pd (_mm256_extractf128_pd(L,1), S);
                S = _mm_add_pd (_mm256_extractf128_pd(H,1), S);
                p += 4;
                y += 4;
                k2 -= 4;
            }

            if (k2 > 1) {
                __m128d P = _mm_loadA_pd(p);
                __m128d T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                __m128d T1 = _mm_mul_pd (_mm_loadu_pd(y+k), P);
                S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                p += 2;
                y += 2;
                k2 -= 2;
            }

            if (k2 >= 1) {
                __m128d B;
                B = _mm_mul_pd (_mm_set1_pd(p[0]), _mm_set_pd(y[k],y[0]));
                S = _mm_add_pd (B, S);
                y += 1;
            }

           _mm_storeu_pd (z, S);
        }

#       elif CAN_USE_SSE2 && ALIGN >= 16 /* works, but slower, when unaligned */
        {
            __m128d S;
            int k2;

#           if (ALIGN_FORWARD & 8)
            {
                S = _mm_mul_pd (_mm_set1_pd(x[0]), _mm_set_pd (y[k], y[0]));
                p = x+1;
                y += 1;
                k2 = k-1;
            }
#           else
            {
                S = _mm_setzero_pd ();
                p = x;
                k2 = k;
            }
#           endif

            if (k & 1) {  /* second column not aligned if first is */
                while (k2 >= 4) {
                    __m128d P, T0, T1;
                    P = _mm_loadA_pd(p);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                    T1 = _mm_mul_pd (_mm_loadu_pd(y+k), P);
                    S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                    S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                    P = _mm_loadA_pd(p+2);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y+2), P);
                    T1 = _mm_mul_pd (_mm_loadu_pd(y+k+2), P);
                    S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                    S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                    p += 4;
                    y += 4;
                    k2 -= 4;
                }
           }
           else {  /* second column has same 16-byte alignment as first */
                while (k2 >= 4) {
                    __m128d P, T0, T1;
                    P = _mm_loadA_pd(p);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                    T1 = _mm_mul_pd (_mm_loadA_pd(y+k), P);
                    S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                    S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                    P = _mm_loadA_pd(p+2);
                    T0 = _mm_mul_pd (_mm_loadA_pd(y+2), P);
                    T1 = _mm_mul_pd (_mm_loadA_pd(y+k+2), P);
                    S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                    S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                    p += 4;
                    y += 4;
                    k2 -= 4;
                }
           }

           if (k2 > 1) {
                __m128d P = _mm_loadA_pd(p);
                __m128d T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                __m128d T1 = _mm_mul_pd (_mm_loadu_pd(y+k), P);
                S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                p += 2;
                y += 2;
                k2 -= 2;
            }

            if (k2 >= 1) {
                __m128d B;
                B = _mm_mul_pd (_mm_set1_pd(p[0]), _mm_set_pd(y[k],y[0]));
                S = _mm_add_pd (B, S);
                y += 1;
            }

           _mm_storeu_pd (z, S);
        }

#       else  /* non-SIMD code */
        {
            double s[2] = { 0, 0 };
            int k2 = k;

            p = x;

            while (k2 > 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[0] += p[1] * y[1];
                s[1] += p[1] * y[k+1];
                p += 2;
                y += 2;
                k2 -= 2;
            }

            if (k2 >= 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                y += 1;
            }

            z[0] = s[0];
            z[1] = s[1];
        }
#       endif
    }

    else if (m == 1) {  /* Do one final dot product */

        double s = 0.0;
        double *p = x;
        int k2 = k;

        while (k2 >= 4) {
            s += p[0] * y[0];
            s += p[1] * y[1];
            s += p[2] * y[2];
            s += p[3] * y[3];
            p += 4;
            y += 4;
            k2 -= 4;
        }

        if (k2 > 1) {
            s += p[0] * y[0];
            s += p[1] * y[1];
            p += 2;
            y += 2;
            k2 -= 2;
        }

        if (k2 >= 1) {
            s += p[0] * y[0];
        }

        z[0] = s;
    }
}


/* Product of n x k matrix (x) and column vector of length k (y) with result 
   stored in z. 

   The product is computed using an outer loop that accumulates the sums for 
   all elements of the result vector, iterating over columns of x, in order
   to produce sequential accesses.  This loop is unrolled to accumulate from
   two columns of x at once.

   Cases where k is 0 or 1 and cases where n is 2 are handled specially. */

void matprod_mat_vec (double * MATPROD_RESTRICT x, 
                      double * MATPROD_RESTRICT y, 
                      double * MATPROD_RESTRICT z, int n, int k)
{
    if (n <= 0) return;

    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    /* Specially handle scalar times row vector and zero-length matrix. */

    if (k <= 1) {
        if (k == 1)
            scalar_multiply (y[0], x, z, n);
        else /* k == 0 */
            set_to_zeros (z, n);
        return;
    }

    /* Handle matrices with 2 or 3 rows specially, holding the result vector
       in a local variable. */

    if (n == 2) { 

#       if CAN_USE_SSE2 && ALIGN >= 16 && ALIGN_OFFSET%16 == 0

            __m128d S;  /* sums for the two values in the result */
            __m128d A;

            /* Each time around this loop, add the products of two
               columns of x with two elements of y to S.  Adjust x
               and y to account for this. */

            S = _mm_setzero_pd (); 

            while (k > 1) {
                A = _mm_mul_pd (_mm_load_pd(x), _mm_load1_pd(y));
                S = _mm_add_pd (A, S);
                A = _mm_mul_pd (_mm_load_pd(x+2), _mm_load1_pd(y+1));
                S = _mm_add_pd (A, S);
                x += 4;
                y += 2;
                k -= 2;
            }

            if (k >= 1) {
                A = _mm_mul_pd (_mm_load_pd(x), _mm_load1_pd(y));
                S = _mm_add_pd (A, S);
            }

            /* Store the two sums in S in the result vector. */

            _mm_storeu_pd (z, S);

#       else

            double s[2];  /* sums for the two values in the result */

            s[0] = s[1] = 0;

            /* Each time around this loop, add the products of two
               columns of x with two elements of y to s[0] and s[1].
               Adjust x and y to account for this. */

            while (k > 1) {
                s[0] = (s[0] + (x[0] * y[0])) + (x[2] * y[1]);
                s[1] = (s[1] + (x[1] * y[0])) + (x[3] * y[1]);
                x += 4;
                y += 2;
                k -= 2;
            }

            if (k >= 1) {
                s[0] += x[0] * y[0];
                s[1] += x[1] * y[0];
            }

            /* Store the two sums in s[0] and s[1] in the result vector. */

            z[0] = s[0];
            z[1] = s[1];


#       endif

        return;
    }

    if (n == 3) { 

        double s[3];  /* sums for the three values in the result */

        s[0] = s[1] = s[2] = 0;

        /* Each time around this loop, add the products of two columns
           of x with two elements of y to s[0], s[1], and s[2].
           Adjust x and y to account for this. */

        while (k > 1) {
            s[0] = (s[0] + (x[0] * y[0])) + (x[3] * y[1]);
            s[1] = (s[1] + (x[1] * y[0])) + (x[4] * y[1]);
            s[2] = (s[2] + (x[2] * y[0])) + (x[5] * y[1]);
            x += 6;
            y += 2;
            k -= 2;
        }

        if (k >= 1) {
            s[0] += x[0] * y[0];
            s[1] += x[1] * y[0];
            s[2] += x[2] * y[0];
        }

        /* Store the three sums in s[0], s[1], and s[2] in the result vector. */

        z[0] = s[0];
        z[1] = s[1];
        z[2] = s[2];

        return;
    }

    /* To start, set the result, z, to the sum from the first two
       columns.  Note that k is at least 2 here, and that n is at
       least 4, since lower values are handled above. 

       A few initial products may be done to help alignment.  This is
       also done, in the same way, later, allowing reuse of n2. */

    int n2;     /* number of elements in z after those done to help alignment */
    int n3;
    double *q;  /* pointer going along z */

    q = z;
    n2 = n;

#   if ALIGN_FORWARD & 8
        q[0] = x[0] * y[0] + x[n] * y[1];
        n2 -= 1;
        x += 1;
        q += 1;
#   endif

#   if CAN_USE_AVX && (ALIGN_FORWARD & 16)
        /* q[0] = x[0] * y[0] + x[n] * y[1];
           q[1] = x[1] * y[0] + x[n+1] * y[1]; */
        __m128d Y0 = _mm_set1_pd(y[0]);
        __m128d Y1 = _mm_set1_pd(y[1]);
        __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
        __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
        _mm_storeA_pd (q, _mm_add_pd (X, P));
        n2 -= 2;
        x += 2;
        q += 2;
#   endif

    n3 = n2;

#   if CAN_USE_AVX
    {
        __m256d Y0b = _mm256_set1_pd(y[0]);
        __m256d Y1b = _mm256_set1_pd(y[1]);
        while (n3 >= 4) { 
            __m256d X = _mm256_mul_pd (_mm256_loadA_pd(x), Y0b);
            __m256d P = _mm256_mul_pd (_mm256_loadu_pd(x+n), Y1b);
            _mm256_storeA_pd (q, _mm256_add_pd (X, P));
            x += 4;
            q += 4;
            n3 -= 4;
        }
        if (n3 > 1) {
            __m128d Y0 = _mm256_castpd256_pd128(Y0b);
            __m128d Y1 = _mm256_castpd256_pd128(Y1b);
            __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
            __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
            _mm_storeA_pd (q, _mm_add_pd (X, P));
            x += 2;
            q += 2;
            n3 -= 2;
        }
    }

#   elif CAN_USE_SSE2
    {
        __m128d Y0 = _mm_load1_pd(y);
        __m128d Y1 = _mm_load1_pd(y+1);
        while (n3 >= 4) { 
            __m128d X, P;
            X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
            P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
            _mm_storeA_pd (q, _mm_add_pd (X, P));
            X = _mm_mul_pd (_mm_loadA_pd(x+2), Y0);
            P = _mm_mul_pd (_mm_loadu_pd(x+n+2), Y1);
            _mm_storeA_pd (q+2, _mm_add_pd (X, P));
            x += 4;
            q += 4;
            n3 -= 4;
        }
        if (n3 > 1) {
            __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
            __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
            _mm_storeA_pd (q, _mm_add_pd (X, P));
            x += 2;
            q += 2;
            n3 -= 2;
        }
    }

#   else  /* non-SIMD code */
    {
        while (n3 >= 4) { 
            q[0] = x[0] * y[0] + x[n] * y[1];
            q[1] = x[1] * y[0] + x[n+1] * y[1];
            q[2] = x[2] * y[0] + x[n+2] * y[1];
            q[3] = x[3] * y[0] + x[n+3] * y[1];
            x += 4;
            q += 4;
            n3 -= 4;
        }
        if (n3 > 1) {
            q[0] = x[0] * y[0] + x[n] * y[1];
            q[1] = x[1] * y[0] + x[n+1] * y[1];
            x += 2;
            q += 2;
            n3 -= 2;
        }
    }
#   endif

    if (n3 >= 1) {
        q[0] = x[0] * y[0] + x[n] * y[1];
        x += 1;
    }

    x += n;
    y += 2;
    k -= 2;

    /* Each time around this loop, add the products of two columns of x 
       with two elements of y to the result vector, z.  Adjust x and y
       to account for this. */

    while (k > 1) {

        q = z;

#       if ALIGN_FORWARD & 8
            q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
            x += 1;
            q += 1;
#       endif

#       if CAN_USE_AVX && (ALIGN_FORWARD & 16)
            /* q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
               q[1] = (q[1] + x[1] * y[0]) + x[n+1] * y[1]; */
            __m128d Y0 = _mm_set1_pd(y[0]);
            __m128d Y1 = _mm_set1_pd(y[1]);
            __m128d Q = _mm_loadA_pd(q);
            __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
            __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
            Q = _mm_add_pd (_mm_add_pd (Q, X), P);
            _mm_storeA_pd (q, Q);
            x += 2;
            q += 2;
#       endif

        n3 = n2;

#       if CAN_USE_AVX
        {
            __m256d Y0b = _mm256_set1_pd(y[0]);
            __m256d Y1b = _mm256_set1_pd(y[1]);
            while (n3 >= 4) { 
                __m256d Q = _mm256_loadA_pd(q);
                __m256d X = _mm256_mul_pd (_mm256_loadA_pd(x), Y0b);
                __m256d P = _mm256_mul_pd (_mm256_loadu_pd(x+n), Y1b);
                Q = _mm256_add_pd (_mm256_add_pd (Q, X), P);
                _mm256_storeA_pd (q, Q);
                x += 4;
                q += 4;
                n3 -= 4;
            }
            if (n3 > 1) {
                __m128d Y0 = _mm256_castpd256_pd128(Y0b);
                __m128d Y1 = _mm256_castpd256_pd128(Y1b);
                __m128d Q = _mm_loadA_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
                __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
                Q = _mm_add_pd (_mm_add_pd (Q, X), P);
                _mm_storeA_pd (q, Q);
                x += 2;
                q += 2;
                n3 -= 2;
            }
        }

#       elif CAN_USE_SSE2
        {
            __m128d Y0 = _mm_load1_pd(y);
            __m128d Y1 = _mm_load1_pd(y+1);
            while (n3 >= 4) { 
                __m128d Q, X, P;
                Q = _mm_loadA_pd(q);
                X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
                P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
                Q = _mm_add_pd (_mm_add_pd (Q, X), P);
                _mm_storeA_pd (q, Q);
                Q = _mm_loadA_pd(q+2);
                X = _mm_mul_pd (_mm_loadA_pd(x+2), Y0);
                P = _mm_mul_pd (_mm_loadu_pd(x+n+2), Y1);
                Q = _mm_add_pd (_mm_add_pd (Q, X), P);
                _mm_storeA_pd (q+2, Q);
                x += 4;
                q += 4;
                n3 -= 4;
            }
            if (n3 > 1) {
                __m128d Q = _mm_loadA_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
                __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
                Q = _mm_add_pd (_mm_add_pd (Q, X), P);
                _mm_storeA_pd (q, Q);
                x += 2;
                q += 2;
                n3 -= 2;
            }
        }

#       else  /* non-SIMD code */
        {
            while (n3 >= 4) { 
                q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
                q[1] = (q[1] + x[1] * y[0]) + x[n+1] * y[1];
                q[2] = (q[2] + x[2] * y[0]) + x[n+2] * y[1];
                q[3] = (q[3] + x[3] * y[0]) + x[n+3] * y[1];
                x += 4;
                q += 4;
                n3 -= 4;
            }
            if (n3 > 1) {
                q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
                q[1] = (q[1] + x[1] * y[0]) + x[n+1] * y[1];
                x += 2;
                q += 2;
                n3 -= 2;
            }
        }
#       endif

        if (n3 >= 1) {
            q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
            x += 1;
        }

        x += n;
        y += 2;
        k -= 2;
    }

    /* Add the last column if there are an odd number of columns. */

    if (k >= 1) {

        q = z;

#       if ALIGN_FORWARD & 8
            q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
            x += 1;
            q += 1;
#       endif

#       if CAN_USE_AVX && (ALIGN_FORWARD & 16)
            /* q[0] = (q[0] + x[0] * y[0]) + x[n] * y[1];
               q[1] = (q[1] + x[1] * y[0]) + x[n+1] * y[1]; */
            __m128d Y0 = _mm_set1_pd(y[0]);
            __m128d Y1 = _mm_set1_pd(y[1]);
            __m128d Q = _mm_loadA_pd(q);
            __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
            __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
            Q = _mm_add_pd (_mm_add_pd (Q, X), P);
            _mm_storeA_pd (q, Q);
            x += 2;
            q += 2;
#       endif

        n3 = n2;

#       if CAN_USE_AVX
        {
            __m256d Yb = _mm256_set1_pd(y[0]);
            while (n3 >= 4) { 
                __m256d Q = _mm256_loadA_pd(q);
                __m256d X = _mm256_mul_pd (_mm256_loadA_pd(x), Yb);
                Q = _mm256_add_pd (Q, X);
                _mm256_storeA_pd (q, Q);
                x += 4;
                q += 4;
                n3 -= 4;
            }
            if (n3 > 1) { 
                __m128d Y = _mm256_castpd256_pd128(Yb);
                __m128d Q = _mm_loadA_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y);
                Q = _mm_add_pd (Q, X);
                _mm_storeA_pd (q, Q);
                x += 2;
                q += 2;
                n3 -= 2;
            }
        }

#       elif CAN_USE_SSE2
        {
            __m128d Y = _mm_load1_pd(y);
            while (n3 >= 4) { 
                __m128d Q, X;
                Q = _mm_loadA_pd(q);
                X = _mm_mul_pd (_mm_loadA_pd(x), Y);
                Q = _mm_add_pd (Q, X);
                _mm_storeA_pd (q, Q);
                Q = _mm_loadA_pd(q+2);
                X = _mm_mul_pd (_mm_loadA_pd(x+2), Y);
                Q = _mm_add_pd (Q, X);
                _mm_storeA_pd (q+2, Q);
                x += 4;
                q += 4;
                n3 -= 4;
            }
            if (n3 > 1) { 
                __m128d Q = _mm_loadA_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y);
                Q = _mm_add_pd (Q, X);
                _mm_storeA_pd (q, Q);
                x += 2;
                q += 2;
                n3 -= 2;
            }
        }
#       else
        {
            while (n3 >= 4) { 
                q[0] += x[0] * y[0];
                q[1] += x[1] * y[0];
                q[2] += x[2] * y[0];
                q[3] += x[3] * y[0];
                x += 4;
                q += 4;
                n3 -= 4;
            }
            if (n3 > 1) { 
                q[0] += x[0] * y[0];
                q[1] += x[1] * y[0];
                x += 2;
                q += 2;
                n3 -= 2;
            }
        }

#       endif

        if (n3 >= 1) {
            q[0] += x[0] * y[0];
        }
    }
}


/* Outer product - n x 1 matrix x times 1 x m matrix y, with result stored
   in the n x m matrix z. */

void matprod_outer (double * MATPROD_RESTRICT x, 
                    double * MATPROD_RESTRICT y, 
                    double * MATPROD_RESTRICT z, int n, int m)
{
    if (n <= 1) {
        if (n == 1)
            scalar_multiply (x[0], y, z, m);
        return;
    }
    if (m <= 1) {
        if (m == 1)
            scalar_multiply (y[0], x, z, n);
        return;
    }

    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    if (n > 4) {

        const int n2 = (ALIGN_FORWARD & 8) ? n-1 : n;
        int j;

        for (j = 0; j < m; j++) {

            double t = y[0];
            double *p;

#           if ALIGN_FORWARD & 8
                z[0] = x[0] * t;
                z += 1;
                p = ASSUME_ALIGNED (x+1, ALIGN, (ALIGN_OFFSET+8)%ALIGN);
#           else
                p = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
#           endif

            double *f = p+(n2-3);

            if (CAN_ASSUME_ALIGNED && ALIGN >= 32 && (((j*n) & 3) == 0)) {
                z = ASSUME_ALIGNED (z, 32, (ALIGN_OFFSET & 8) 
                                             ? (ALIGN_OFFSET+8)%32
                                             : ALIGN_OFFSET%32);
                while (p < f) {
                    z[0] = p[0] * t;
                    z[1] = p[1] * t;
                    z[2] = p[2] * t;
                    z[3] = p[3] * t;
                    z += 4;
                    p += 4;
                }
            }
            else if (CAN_ASSUME_ALIGNED && ALIGN >= 16 && (((j&n) & 1) == 0)) {
                z = ASSUME_ALIGNED (z, 16, (ALIGN_OFFSET & 8) 
                                             ? (ALIGN_OFFSET+8)%16
                                             : ALIGN_OFFSET%16);
                while (p < f) {
                    z[0] = p[0] * t;
                    z[1] = p[1] * t;
                    z[2] = p[2] * t;
                    z[3] = p[3] * t;
                    z += 4;
                    p += 4;
                }
            }
            else {
                while (p < f) {
                    z[0] = p[0] * t;
                    z[1] = p[1] * t;
                    z[2] = p[2] * t;
                    z[3] = p[3] * t;
                    z += 4;
                    p += 4;
                }
            }

            if (n2 & 2) {
                z[0] = p[0] * t;
                z[1] = p[1] * t;
                z += 2;
                p += 2;
            }

            if (n2 & 1) {
                z[0] = p[0] * t;
                z += 1;
            }

            y += 1;
        }
    }

    else if (n == 4) {
        double *e = z + 4*m;
#       if CAN_USE_AVX
        {
            __m256d X = _mm256_loadu_pd (x);
            while (z < e) {
                __m256d Y0 = _mm256_set1_pd (y[0]);
                _mm256_storeu_pd (z, _mm256_mul_pd(X,Y0));
                z += 4;
                y += 1;
            }
        }
#       elif CAN_USE_SSE2
        {
            __m128d Xa = _mm_loadu_pd (x);
            __m128d Xb = _mm_loadu_pd (x+2);
            while (z < e) {
                __m128d Y0 = _mm_set1_pd (y[0]);
                _mm_storeu_pd (z, _mm_mul_pd(Xa,Y0));
                _mm_storeu_pd (z+2, _mm_mul_pd(Xb,Y0));
                z += 4;
                y += 1;
            }
        }
#       else
        {
            double X[4] = { x[0], x[1], x[2], x[3] };
            while (z < e) {
                double y0 = y[0];
                z[0] = y0 * X[0];
                z[1] = y0 * X[1];
                z[2] = y0 * X[2];
                z[3] = y0 * X[3];
                z += 4;
                y += 1;
            }
        }
#       endif
    }

    else if (n == 3) {
#       if CAN_USE_AVX && 0
        {
            __m256d Xa = _mm256_set_pd (x[0], x[2], x[1], x[0]);
            __m128d Xc = _mm_loadu_pd (x+1);
            double *e = z + 3*(m-1);
            while (z < e) {
                __m256d Ya = _mm256_set_pd (y[1], y[0], y[0], y[0]);
                _mm256_storeu_pd (z, _mm256_mul_pd(Xa,Ya));
                __m128d Yc = _mm_set1_pd (y[1]);
                _mm_storeu_pd (z+4, _mm_mul_pd(Xc,Yc));
                z += 6;
                y += 2;
            }
            if (m & 1) {
                __m128d Y = _mm_set1_pd (y[0]);
                _mm_storeu_pd (z, _mm_mul_pd(_mm256_castpd256_pd128(Xa),Y));
                _mm_store_sd (z+2, _mm_mul_sd(_mm256_extractf128_pd(Xa,1),Y));
            }
        }
#       elif CAN_USE_SSE2
        {
            __m128d Xa = _mm_set_pd (x[1], x[0]);
            __m128d Xb = _mm_set_pd (x[0], x[2]);
            __m128d Xc = _mm_set_pd (x[2], x[1]);
            __m128d Y;
            if (ALIGN_FORWARD & 8) {
                Y = _mm_set1_pd (y[0]);
                _mm_store_sd (z, _mm_mul_sd(Xa,Y));
                _mm_storeA_pd (z+1, _mm_mul_pd(Xc,Y));
                z += 3;
                y += 1;
                m -= 1;
            }
            double *e = z + 3*(m-1);
            while (z < e) {
                Y = _mm_set1_pd (y[0]);
                _mm_storeA_pd (z, _mm_mul_pd(Xa,Y));
                Y = _mm_set_pd (y[1],y[0]);
                _mm_storeA_pd (z+2, _mm_mul_pd(Xb,Y));
                Y = _mm_set1_pd (y[1]);
                _mm_storeA_pd (z+4, _mm_mul_pd(Xc,Y));
                z += 6;
                y += 2;
            }
            if (m & 1) {
                Y = _mm_set1_pd (y[0]);
                _mm_storeA_pd (z, _mm_mul_pd(Xa,Y));
                _mm_store_sd (z+2, _mm_mul_sd(Xb,Y));
            }
        }
#       else
        {
            double X[3] = { x[0], x[1], x[2] };
            double *e = z + 3*(m-1);
            while (z < e) {
                double y0 = y[0];
                z[0] = y0 * X[0];
                z[1] = y0 * X[1];
                z[2] = y0 * X[2];
                double y1 = y[1];
                z[3] = y1 * X[0];
                z[4] = y1 * X[1];
                z[5] = y1 * X[2];
                z += 6;
                y += 2;
            }
            if (m & 1) {
                double y0 = y[0];
                z[0] = y0 * X[0];
                z[1] = y0 * X[1];
                z[2] = y0 * X[2];
            }
        }
#       endif
    }

    else {  /* n == 2 */
#       if CAN_USE_AVX
        {
            __m256d X = _mm256_set_pd (x[1], x[0], x[1], x[0]);
            double *e = z + 2*(m-1);
            while (z < e) {
                __m256d Y = _mm256_set_pd (y[1], y[1], y[0], y[0]);
                _mm256_storeu_pd (z, _mm256_mul_pd(X,Y));
                z += 4;
                y += 2;
            }
            if (m & 1) {
                __m128d Y = _mm_set1_pd (y[0]);
                _mm_storeu_pd (z, _mm_mul_pd(_mm256_castpd256_pd128(X),Y));
            }
        }
#       elif CAN_USE_SSE2
        {
            __m128d X = _mm_loadu_pd (x);
            __m128d Y;
            double *e = z + 2*(m-1);
            while (z < e) {
                Y = _mm_set1_pd (y[0]);
                _mm_storeu_pd (z, _mm_mul_pd(X,Y));
                Y = _mm_set1_pd (y[1]);
                _mm_storeu_pd (z+2, _mm_mul_pd(X,Y));
                z += 4;
                y += 2;
            }
            if (m & 1) {
                Y = _mm_set1_pd (y[0]);
                _mm_storeu_pd (z, _mm_mul_pd(X,Y));
            }
        }
#       else
        {
            double X[3] = { x[0], x[1] };
            double *e = z + 2*m;
            while (z < e) {
                double y0 = y[0];
                z[0] = y0 * X[0];
                z[1] = y0 * X[1];
                z += 2;
                y += 1;
            }
        }
#       endif
    }
}


/* Product of an n x k matrix (x) and a k x m matrix (y) with result stored 
   in z. 

   The inner loop does two matrix-vector products each time, implemented 
   much as in matprod_mat_vec above, except for computing two columns. This
   gives a reasonably efficient implementation of an outer product (where
   k is one).

   Cases where n is two are handled specially, accumulating sums in two
   local variables rather than in a column of the result, and then storing
   them in the result column at the end.  Outer products (where k is one)
   are also handled specially, with the matprod_outer procedure. */

void matprod_mat_mat (double * MATPROD_RESTRICT x, 
                      double * MATPROD_RESTRICT y, 
                      double * MATPROD_RESTRICT z, int n, int k, int m)
{
    if (n <= 0 || m <= 0) return;

    if (k == 1) {
        matprod_outer (x, y, z, n, m);
        return;
    }

    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    /* Handle n=2 specially. */

    if (n == 2) {

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

   Each element of the result is the dot product of a column of x and
   a column of y.  Four elements of this result are computed at once,
   using two consecutive columns of x and two consecutive columns of y
   (except perhaps for odd columns at the end), thereby reducing the
   number of memory accesses.

   The case of k=2 is handled specially.

   When the two operands are the same, the result will be a symmetric
   matrix.  After computation of each column or pair of columns, they
   are copied to the corresponding rows; hence each column need be
   computed only from the diagonal element down. */

void matprod_trans1 (double * MATPROD_RESTRICT x, 
                     double * MATPROD_RESTRICT y, 
                     double * MATPROD_RESTRICT z, int n, int k, int m)
{
    if (n <= 0 || m <= 0) return;

    if (k <= 1) {
        if (k == 1)
            matprod_outer (x, y, z, n, m);
        else
            set_to_zeros (z, n*m);
        return;
    }

    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    /* Handle the case where k is two specially. */

    if (k == 2) {
#       if CAN_USE_AVX
        {
            double *e = y + 2*m;
            double *f = x + 2*(n-3);
            while (y < e) {
                __m256d Y = _mm256_set_pd (y[1], y[0], y[1], y[0]);
                double *p = x;
                while (p < f) {
                    __m256d M0 = _mm256_mul_pd (Y, _mm256_loadu_pd(p));
                    __m256d M1 = _mm256_mul_pd (Y, _mm256_loadu_pd(p+4));
                    __m256d Z = _mm256_hadd_pd 
                                  (_mm256_permute2f128_pd (M0, M1, 0x20),
                                   _mm256_permute2f128_pd (M0, M1, 0x31));
                    _mm256_storeu_pd (z, Z);
                    p += 8;
                    z += 4;
                }
                if (n & 2) {
                    __m128d M0 = _mm_mul_pd (_mm256_castpd256_pd128(Y),
                                             _mm_loadu_pd(p));
                    __m128d M1 = _mm_mul_pd (_mm256_castpd256_pd128(Y),
                                             _mm_loadu_pd(p+2));
                    _mm_storeu_pd (z, _mm_hadd_pd (M0, M1));
                    p += 4;
                    z += 2;
                }
                if (n & 1) {
                    __m128d M0 = _mm_mul_pd (_mm256_castpd256_pd128(Y),
                                             _mm_loadu_pd(p));
                    _mm_store_sd (z, _mm_hadd_pd (M0, M0));
                    p += 2;
                    z += 1;
                }
                y += 2;
            }
        }
#       elif CAN_USE_SSE3
        {
            double *e = y + 2*m;
            double *f = x + 2*(n-3);
            while (y < e) {
                __m128d Y = _mm_set_pd (y[1], y[0]);
                __m128d M0, M1;
                double *p = x;
                while (p < f) {
                    M0 = _mm_mul_pd (Y, _mm_loadAA_pd(p));
                    M1 = _mm_mul_pd (Y, _mm_loadAA_pd(p+2));
                    _mm_storeu_pd (z, _mm_hadd_pd (M0, M1));
                    M0 = _mm_mul_pd (Y, _mm_loadAA_pd(p+4));
                    M1 = _mm_mul_pd (Y, _mm_loadAA_pd(p+6));
                    _mm_storeu_pd (z+2, _mm_hadd_pd (M0, M1));
                    p += 8;
                    z += 4;
                }
                if (n & 2) {
                    M0 = _mm_mul_pd (Y, _mm_loadAA_pd(p));
                    M1 = _mm_mul_pd (Y, _mm_loadAA_pd(p+2));
                    _mm_storeu_pd (z, _mm_hadd_pd (M0, M1));
                    p += 4;
                    z += 2;
                }
                if (n & 1) {
                    M0 = _mm_mul_pd (Y, _mm_loadAA_pd(p));
                    _mm_store_sd (z, _mm_hadd_pd (M0, M0));
                    p += 2;
                    z += 1;
                }
                y += 2;
            }
        }
#       else
        {
            double *e = y + 2*m;
            double *f = x + 2*n;
            while (y < e) {
                double *p = x;
                while (p < f) {
                    z[0] = p[0] * y[0] + p[1] * y[1];
                    p += 2;
                    z += 1;
                }
                y += 2;
            }
        }
#       endif
        return;
    }

    int sym = x==y && n==m;  /* same operands, so symmetric result? */
    int j = 0;               /* number of columns of result produced so far */
    int me = m-1;            /* limit for cols that can be computed as pairs */

    /* Compute two columns of the result each time around this loop, updating
       y, z, and j accordingly. */

    while (j < me) {

        double *z2 = z+n;
        double *r = x;
        int nn = n;
        double *rz;

        /* If the result is symmetric, skip down to the diagonal element
           of the first column.  Also, let nn be the number of elements to 
           compute for this column, and set r to the start of the column
           of x to use.  However, skip down one less than this if it
           helps alignment. */
           
        if (sym) {
#           if CAN_USE_SSE2 && ALIGN >= 16
                int jj = j & ~1;
#           else
                int jj = j;
#           endif
            z += jj;
            z2 += jj;
            nn -= jj;
            r += jj*k;
            rz = z;
        }

        /* Compute pairs of elements in the two columns being computed.  Copy
           them to the corresponding rows too, if the result is symmetric. */

        double *ze = z + (nn-1);

        while (z < ze) {
#           if CAN_USE_AVX
            {
                double *q, *qe;
                __m256d S;
#               if ALIGN_FORWARD & 8
                    __m256d X = _mm256_set_pd (r[k], r[0], r[k], r[0]);
                    __m256d Y = _mm256_set_pd (y[k], y[k], y[0], y[0]);
                    S = _mm256_mul_pd(X,Y);
                        /* ie, _mm256_set_pd 
                                 (r[k]*y[k], r[0]*y[k], r[k]*y[0], r[0]*y[0]) */
                    r += 1;
                    q = y+1;
                    qe = q+((k-1)-3);
#               else
                    S = _mm256_setzero_pd();
                    q = y;
                    qe = q+(k-3);
#               endif
                while (q < qe) {
                    __m256d Q0 = _mm256_loadu_pd(q);
                    __m256d R0 = _mm256_loadu_pd(r);
                    __m256d Rk = _mm256_loadu_pd(r+k);
                    __m256d Qk = _mm256_loadu_pd(q+k);
                    __m256d M00 = _mm256_mul_pd (Q0, R0);
                    __m256d M0k = _mm256_mul_pd (Q0, Rk);
                    __m256d Mk0 = _mm256_mul_pd (Qk, R0);
                    __m256d Mkk = _mm256_mul_pd (Qk, Rk);
                    __m256d L0 = _mm256_unpacklo_pd (M00, M0k);
                    __m256d H0 = _mm256_unpackhi_pd (M00, M0k);
                    __m256d Lk = _mm256_unpacklo_pd (Mk0, Mkk);
                    __m256d Hk = _mm256_unpackhi_pd (Mk0, Mkk);
                    S = _mm256_add_pd (S, _mm256_permute2f128_pd(L0, Lk, 0x20));
                    S = _mm256_add_pd (S, _mm256_permute2f128_pd(H0, Hk, 0x20));
                    S = _mm256_add_pd (S, _mm256_permute2f128_pd(L0, Lk, 0x31));
                    S = _mm256_add_pd (S, _mm256_permute2f128_pd(H0, Hk, 0x31));
                    r += 4;
                    q += 4;
                }
                qe = y+k;
                while (q < qe) {
                    __m256d X = _mm256_set_pd (r[k], r[0], r[k], r[0]);
                    __m256d Y = _mm256_set_pd (q[k], q[k], q[0], q[0]);
                    S = _mm256_add_pd (_mm256_mul_pd(X,Y), S);
                    r += 1;
                    q += 1;
                }
                __m128d H = _mm256_extractf128_pd(S,1);
                _mm_storeu_pd (z, _mm256_castpd256_pd128(S));
                _mm_storeu_pd (z2, H);
                if (sym) {
                    _mm_storeu_pd (rz, 
                       _mm_unpacklo_pd(_mm256_castpd256_pd128(S),H));
                    rz += n;
                    _mm_storeu_pd (rz, 
                       _mm_unpackhi_pd(_mm256_castpd256_pd128(S),H));
                    rz += n;
                }
            }
#           elif CAN_USE_SSE2
            {
                double *q, *qe;
                __m128d S0, S1;
#               if ALIGN_FORWARD & 8
                    __m128d X, Y;
                    X = _mm_set_pd (r[k], r[0]);
                    Y = _mm_set_pd (y[0], y[0]);
                    S0 = _mm_mul_pd(X,Y);
                         /* ie, _mm_set_pd (r[k]*y[0], r[0]*y[0]) */
                    Y = _mm_set_pd (y[k], y[k]);
                    S1 = _mm_mul_pd(X,Y);
                         /* ie, _mm_set_pd (r[k]*y[k], r[0]*y[k]) */
                    r += 1;
                    q = y+1;
                    qe = q+((k-1)-1);
#               else
                    S0 = _mm_setzero_pd();
                    S1 = _mm_setzero_pd();
                    q = y;
                    qe = q+(k-1);
#               endif
                while (q < qe) {
                    __m128d Q0 = _mm_loadA_pd(q);
                    __m128d R0 = _mm_loadA_pd(r);
                    __m128d Rk = _mm_loadu_pd(r+k);
                    __m128d Qk = _mm_loadu_pd(q+k);
                    __m128d M00 = _mm_mul_pd (Q0, R0);
                    __m128d M0k = _mm_mul_pd (Q0, Rk);
                    __m128d Mk0 = _mm_mul_pd (Qk, R0);
                    __m128d Mkk = _mm_mul_pd (Qk, Rk);
                    __m128d L0 = _mm_unpacklo_pd (M00, M0k);
                    __m128d H0 = _mm_unpackhi_pd (M00, M0k);
                    __m128d Lk = _mm_unpacklo_pd (Mk0, Mkk);
                    __m128d Hk = _mm_unpackhi_pd (Mk0, Mkk);
                    S0 = _mm_add_pd (S0, L0);
                    S0 = _mm_add_pd (S0, H0);
                    S1 = _mm_add_pd (S1, Lk);
                    S1 = _mm_add_pd (S1, Hk);
                    r += 2;
                    q += 2;
                }
                if (q < y+k) {
                    __m128d X, Y;
                    X = _mm_set_pd (r[k], r[0]);
                    Y = _mm_set_pd (q[0], q[0]);
                    S0 = _mm_add_pd (_mm_mul_pd(X,Y), S0);
                    Y = _mm_set_pd (q[k], q[k]);
                    S1 = _mm_add_pd (_mm_mul_pd(X,Y), S1);
                    r += 1;
                    q += 1;
                }
#               if ALIGN >= 16 && ALIGN_OFFSET%16 == 0
                    _mm_store_pd (z, S0);
#               else
                    _mm_storeu_pd (z, S0);
#               endif
                _mm_storeu_pd (z2, S1);
                if (sym) {
                    _mm_storeu_pd (rz, _mm_unpacklo_pd(S0,S1));
                    rz += n;
                    _mm_storeu_pd (rz, _mm_unpackhi_pd(S0,S1));
                    rz += n;
                }
            }
#           else
            {
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
                z[0] = s[0];
                z2[0] = s[1];
                z[1] = s[2];
                z2[1] = s[3];
                if (sym) {
                    rz[0] = s[0];
                    rz[1] = s[1];
                    rz[n] = s[2];
                    rz[n+1] = s[3];
                    rz += 2*n;
                }
            }
#           endif

            z += 2;
            z2 += 2;
            r += k;
        }

        /* If an odd number of elements are to be computed in the two columns,
           compute the remaining elements here.  If result is symmetric, store
           in the symmetric places as well. */

        if (nn & 1) {
            double s0 = 0;
            double s1 = 0;
            double *q = y;
            int i = k;
            do {
                double t = *r++;
                s0 += t * *q;
                s1 += t * *(q+k);
                q += 1;
            } while (--i > 0);
            *z = s0;
            *z2 = s1;
            z2 += 1;
            if (sym) {
                rz[0] = s0;
                rz[1] = s1;
            }
        }

        /* Go on to next two columns of y. */

        z = z2;
        y += 2*k;
        j += 2;
    }

    /* If m is odd, compute the final column of the result. If the result is
       symmetric, only the final element needs to be computed, the others
       having already been filled in. */

    if (m & 1) {

        if (sym) {
            double *r = x+(n-1)*k;
            double *q = y;
            double *f = y+k;
            double s = 0;
            do {
                s += *r++ * *q++;
            } while (q < f);
            *(z+n-1) = s;
        }

        else {

            double *r = x;
            double *e = z+n;

            /* If n is odd, compute the first element of the first
               column of the result here.  Also, move r to point to
               the second column of x, and increment z. */

            if (n & 1) {
                double s = 0;
                double *q = y;
                double *e = y+k;
                do { s += *r++ * *q++; } while (q < e);
                *z++ = s;
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
            }
        }
    }
}


/* Product of an n x k matrix (x) and the transpose of an m x k matrix (y) 
   with result stored in z.

   When the two operands are the same, the result will be a symmetric
   matrix.  Only the lower-triangular part of the result is computed,
   with the elements in columns that are computed then being copied to 
   the corresponding elements in rows above the diagonal.

   Cases where n is two are handled specially, accumulating sums in two
   local variables rather than in a column of the result, and then storing
   them in the result column at the end. */

void matprod_trans2 (double * MATPROD_RESTRICT x, 
                     double * MATPROD_RESTRICT y, 
                     double * MATPROD_RESTRICT z, int n, int k, int m)
{
    if (n <= 0 || m <= 0) return;

    if (k <= 1) {
        if (k == 1)
            matprod_outer (x, y, z, n, m);
        else
            set_to_zeros (z, n*m);
        return;
    }

    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    int sym = x==y && n==m;  /* same operands, so symmetric result? */
    double *ex = x + n*k;    /* point past end of x */
    int j = 0;               /* number of columns of result produced so far */

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
