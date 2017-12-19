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
   is suitably large and ALIGN_OFFSET mod ALIGN is zero, as is appropriate 
   for an address that is one of the arguments plus a multiple of of ALIGN. */

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


/* Macro to cast a variable to __m128d if it is __m256d, which does nothing
   when AVX is not available, and hence the variable will already be __m128d.
   This facilitates sharing of code in AVX and SSE2 sections. */

#if CAN_USE_AVX
#   define cast128(x) _mm256_castpd256_pd128(x)
#else
#   define cast128(x) (x)
#endif


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
    CHK_ALIGN(x); CHK_ALIGN(y);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);

    /* Handle k = 0, 1, or 2 specially. */

    if (k <= 2) {
        if (k == 2)
            return x[0] * y[0] + x[1] * y[1];
        if (k == 1) 
            return x[0] * y[0];
        else  /* k <= 0 */
            return 0.0;
    }

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
            A = cast128(AA);
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            A = _mm256_extractf128_pd(AA,1);
            S = _mm_add_sd (A, S);
            A = _mm_unpackhi_pd (A, A);
            S = _mm_add_sd (A, S);
            AA = _mm256_mul_pd (_mm256_loadA_pd(x+i+4), _mm256_loadA_pd(y+i+4));
            A = cast128(AA);
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
            A = cast128(AA);
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
   done with loop unrolling, perhaps using SSE2 or AVX instructions. 

   If y has more than VEC_MAT_ROWS (defined below) rows, the operation
   is done on successive parts of the matrix, each consisting of at
   most MAT_VEC_ROWS of the rows.  The second and later parts add to
   the result found by earlier parts.  The matprod_sub_mat_vec
   procedure below does one such part. */

static void matprod_sub_vec_mat (double * MATPROD_RESTRICT x,
                                 double * MATPROD_RESTRICT y,
                                 double * MATPROD_RESTRICT z,
                                 int k, int m, int rows, int add);

void matprod_vec_mat (double * MATPROD_RESTRICT x, 
                      double * MATPROD_RESTRICT y, 
                      double * MATPROD_RESTRICT z, int k, int m)
{
    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    if (m <= 1) {
        if (m == 1)
            z[0] = matprod_vec_vec (x, y, k);
        return;
    }

    /* Specially handle cases where y has two or fewer rows. */

    if (k <= 2) {

        if (k != 2) {
            if (k == 1)
                scalar_multiply (x[0], y, z, m);
            else  /* k == 0 */
                set_to_zeros (z, m);
        }
        else {  /* k == 2 */
#           if CAN_USE_SSE3 || CAN_USE_AVX
            {
#               if CAN_USE_AVX
                    __m256d T = _mm256_set_pd (x[1], x[0], x[1], x[0]);
#               else  /* CAN_USE_SSE3 */
                    __m128d T = _mm_set_pd (x[1], x[0]);
#               endif
#               if ALIGN_FORWARD & 8
                {
                    __m128d A = _mm_mul_pd (cast128(T), _mm_loadAA_pd(y));
                    _mm_store_sd (z, _mm_hadd_pd(A,A));
                    z += 1;
                    y += 2;
                    m -= 1;
                }
#               endif
#               if CAN_USE_AVX
                    while (m >= 4) {
                         __m256d A = _mm256_mul_pd (T, _mm256_loadAA_pd(y));
                         __m256d B = _mm256_mul_pd (T, _mm256_loadAA_pd(y+4));
                         _mm256_storeAA_pd(z,_mm256_hadd_pd
                                            (_mm256_permute2f128_pd(A,B,0x20),
                                             _mm256_permute2f128_pd(A,B,0x31)));
                        y += 8;
                        z += 4;
                        m -= 4;
                    }
#               else  /* CAN_USE_SSE3 */
                    while (m >= 4) {
                         __m128d A, B;
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
#               endif
                if (m > 1) {
                     __m128d A = _mm_mul_pd (cast128(T), _mm_loadAA_pd(y));
                     __m128d B = _mm_mul_pd (cast128(T), _mm_loadAA_pd(y+2));
                     _mm_storeA_pd (z, _mm_hadd_pd(A,B));
                    y += 4;
                    z += 2;
                    m -= 2;
                }
                if (m >= 1) {
                    __m128d A = _mm_mul_pd (cast128(T), _mm_loadAA_pd(y));
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

    /* The general case with k > 2.  Calls matprod_sub_vec_mat to do parts
       (only one part if y has matrix with fewer than VEC_MAT_ROWS). */

#   define VEC_MAT_ROWS (4096+2048) /* be multiple of 64 to keep any alignment*/

    if (k <= VEC_MAT_ROWS || m <= 4)
        matprod_sub_vec_mat (x, y, z, k, m, k, 0);
    else {
        int rows = k;
        int add = 0;
        while (rows >= 2*VEC_MAT_ROWS) {
            matprod_sub_vec_mat (x, y, z, k, m, VEC_MAT_ROWS, add);
            x += VEC_MAT_ROWS;
            y += VEC_MAT_ROWS;
            rows -= VEC_MAT_ROWS;
            add = 1;
        }
        if (rows > VEC_MAT_ROWS) {
            int nr = (rows/2) & ~7; /* ensure any alignment of x, y preserved */
            matprod_sub_vec_mat (x, y, z, k, m, nr, add);
            x += nr;
            y += nr;
            rows -= nr;
            add = 1;
        }
        matprod_sub_vec_mat (x, y, z, k, m, rows, 1);
    }
}


/* Multiply the first 'rows' elements of vector x with the first
   'rows' rows of matrix y, storing the result in z if 'add' is zero,
   or adding the result to z if 'add' is non-zero.  Note that x and y
   may not be the start of the original vector/matrix.  The k argument
   is the number of rows in the original y, which is the amount to
   step to go right to an element in the same row and the next column.
   The m argument is the number of columns in y.

   The same alignment assumptions hold for x, y, and z as with the
   visible procedures.

   Note that k and 'rows' will be greater than 2. */

static void matprod_sub_vec_mat (double * MATPROD_RESTRICT x,
                                 double * MATPROD_RESTRICT y,
                                 double * MATPROD_RESTRICT z,
                                 int k, int m, int rows, int add)
{
    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    double *ys = y;

    /* In this loop, compute four consecutive elements of the result vector,
       by doing four dot products of x with columns of y.  Adjust y, z, and
       m as we go. */

    while (m >= 4) {

        double *p;               /* Pointer that goes along pairs in x */

#       if CAN_USE_AVX
        {
            /* This loop adds four products to the sums for each of
               the four dot products, adjusting p and y as it goes.
               More products are ossibly are added afterwards.  The
               sums may be initialized to one product or two producs,
               as helps alignment. */

            __m256d S, B;
            int rows2;

            B = _mm256_set_pd (y[3*k], y[2*k], y[k], y[0]);
            S = _mm256_mul_pd (_mm256_set1_pd(x[0]), B);
            y += 1;
            if (add)
                S = _mm256_add_pd (_mm256_loadu_pd(z), S);

#           if (ALIGN_FORWARD & 8) == 0
                B = _mm256_set_pd (y[3*k], y[2*k], y[k], y[0]);
                B = _mm256_mul_pd (_mm256_set1_pd(x[1]), B);
                S = _mm256_add_pd (B, S);
                y += 1;
                p = x+2;
                rows2 = rows-2;
#           else
                p = x+1;
                rows2 = rows-1;
#           endif

            while (rows2 >= 4) {
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
                rows2 -= 4;
            }

            if (rows2 > 1) {
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
                rows2 -= 2;
            }

            if (rows2 >= 1) {
                B = _mm256_set_pd (y[3*k], y[2*k], y[k], y[0]);
                B = _mm256_mul_pd (_mm256_set1_pd(p[0]), B);
                S = _mm256_add_pd (B, S);
                y += 1;
            }

            _mm256_storeu_pd (z, S);
        }

#       elif CAN_USE_SSE2 && ALIGN >= 16 /* works, but slower, when unaligned */
        {
            /* This loop adds two products to the sums for each of the
               four dot products, adjusting p and y as it goes.
               Another product may be added afterwards.  The sums may
               be initialized to one product or two producs, as helps
               alignment. */

            __m128d S0, S1, B, P;
            int rows2;

            P = _mm_set1_pd(x[0]);
            S0 = _mm_mul_pd (P, _mm_set_pd (y[k], y[0]));
            S1 = _mm_mul_pd (P, _mm_set_pd (y[3*k], y[2*k]));
            y += 1;
            if (add) {
                S0 = _mm_add_pd (_mm_loadu_pd(z), S0);
                S1 = _mm_add_pd (_mm_loadu_pd(z+2), S1);
            }

#           if (ALIGN_FORWARD & 8) == 0
                P = _mm_set1_pd(x[1]);
                S0 = _mm_add_pd(S0, _mm_mul_pd(P, _mm_set_pd (y[k], y[0])));
                S1 = _mm_add_pd(S1, _mm_mul_pd(P, _mm_set_pd (y[3*k], y[2*k])));
                y += 1;
                p = x+2;
                rows2 = rows-2;
#           else
                p = x+1;
                rows2 = rows-1;
#           endif

            if (k & 1) {  /* second column not aligned if first is */
                while (rows2 > 1) {
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
                    rows2 -= 2;
                }
            }
            else {  /* second column has same 16-byte alignment as first */
                while (rows2 > 1) {
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
                    rows2 -= 2;
                }
            }

            if (rows2 >= 1) {
                __m128d B;
                __m128d P = _mm_set1_pd(p[0]);
                B = _mm_mul_pd (P, _mm_set_pd(y[k],y[0]));
                S0 = _mm_add_pd (B, S0);
                B = _mm_mul_pd (P, _mm_set_pd(y[3*k],y[2*k]));
                S1 = _mm_add_pd (B, S1);
                y += 1;
            }

            _mm_storeu_pd (z, S0);
            _mm_storeu_pd (z+2, S1);
        }

#       else  /* non-SIMD code */
        {
            /* This loop adds two products to the sums for each of the
               four dot products, adjusting p and y as it goes.
               Another product may be added afterwards. */

            int rows2 = rows;

            double s[4];
            if (add) {
                s[0] = z[0];
                s[1] = z[1];
                s[2] = z[2];
                s[3] = z[3];
            }
            else
                s[0] = s[1] = s[2] = s[3] = 0.0;

            p = x;
            while (rows2 > 1) {
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
                rows2 -= 2;
            }

            if (rows2 >= 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[2] += p[0] * y[2*k];
                s[3] += p[0] * y[3*k];
                y += 1;
            }

            z[0] = s[0];
            z[1] = s[1];
            z[2] = s[2];
            z[3] = s[3];
        }

#       endif

        ys += 4*k;
        y = ys;
        z += 4;
        m -= 4;
    }

    /* Compute the final few dot products left over from the loop above. */

    if (m == 3) {  /* Do three more dot products */

        double *p;               /* Pointer that goes along pairs in x */

#       if CAN_USE_AVX || CAN_USE_SSE2 /* && ALIGN >= 16? slower otherwise? */
        {
            __m128d S, S2;
            int rows2;

#           if (ALIGN_FORWARD & 8)
            {
                __m128d P = _mm_set1_pd(x[0]);
                S = _mm_mul_pd (P, _mm_set_pd (y[k], y[0]));
                S2 = _mm_mul_sd (P, _mm_set_sd (y[2*k]));
                if (add) {
                    S = _mm_add_pd (_mm_loadu_pd(z), S);
                    S2 = _mm_add_sd (_mm_load_sd(z+2), S2);
                }
                p = x+1;
                y += 1;
                rows2 = rows-1;
            }
#           else
            {
                if (add) {
                    S = _mm_loadu_pd(z);
                    S2 = _mm_load_sd(z+2);
                }
                else {
                    S = _mm_setzero_pd ();
                    S2 = _mm_setzero_pd ();
                }
                p = x;
                rows2 = rows;
            }
#           endif

#           if CAN_USE_AVX
                while (rows2 >= 4) {
                    __m256d P = _mm256_loadu_pd(p);
                    __m256d T0 = _mm256_mul_pd (_mm256_loadu_pd(y), P);
                    __m256d T1 = _mm256_mul_pd (_mm256_loadu_pd(y+k), P);
                    __m256d T2 = _mm256_mul_pd (_mm256_loadu_pd(y+2*k), P);
                    __m128d L2 = cast128 (T2);
                    __m128d H2 = _mm256_extractf128_pd (T2, 1);
                    __m256d L = _mm256_unpacklo_pd(T0,T1);
                    __m256d H = _mm256_unpackhi_pd(T0,T1);
                    S = _mm_add_pd (cast128(L), S);
                    S = _mm_add_pd (cast128(H), S);
                    S = _mm_add_pd (_mm256_extractf128_pd(L,1), S);
                    S = _mm_add_pd (_mm256_extractf128_pd(H,1), S);
                    S2 = _mm_add_sd (L2, S2);
                    S2 = _mm_add_sd (_mm_unpackhi_pd(L2,L2), S2);
                    S2 = _mm_add_sd (H2, S2);
                    S2 = _mm_add_sd (_mm_unpackhi_pd(H2,H2), S2);
                    p += 4;
                    y += 4;
                    rows2 -= 4;
                }
#           else  /* CAN_USE_SSE2 */
                if (k & 1) {  /* second column not aligned if first is */
                    while (rows2 >= 4) {
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
                        rows2 -= 4;
                    }
               }
               else {  /* second column has same 16-byte alignment as first */
                    while (rows2 >= 4) {
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
                        rows2 -= 4;
                    }
                }
#           endif

            if (rows2 > 1) {
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
                rows2 -= 2;
            }

            if (rows2 >= 1) {
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
            int rows2 = rows;
            double s[3];
            if (add) {
                s[0] = z[0];
                s[1] = z[1];
                s[2] = z[2];
            }
            else
                s[0] = s[1] = s[2] = 0.0;

            p = x;

            while (rows2 > 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[2] += p[0] * y[2*k];
                s[0] += p[1] * y[1];
                s[1] += p[1] * y[k+1];
                s[2] += p[1] * y[2*k+1];
                p += 2;
                y += 2;
                rows2 -= 2;
            }

            if (rows2 >= 1) {
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

#       if CAN_USE_AVX || CAN_USE_SSE2 && ALIGN >= 16 /* slower when unaligned*/
        {
            __m128d S;
            int rows2;

#           if (ALIGN_FORWARD & 8)
            {
                S = _mm_mul_pd (_mm_set1_pd(x[0]), _mm_set_pd (y[k], y[0]));
                if (add)
                    S = _mm_add_pd (_mm_loadu_pd(z), S);
                p = x+1;
                y += 1;
                rows2 = rows-1;
            }
#           else
            {
                if (add)
                    S = _mm_loadu_pd(z);
                else
                    S = _mm_setzero_pd ();
                p = x;
                rows2 = rows;
            }
#           endif

#           if CAN_USE_AVX
            {
                while (rows2 >= 4) {
                    __m256d P = _mm256_loadAA_pd(p);
                    __m256d T0 = _mm256_mul_pd (_mm256_loadAA_pd(y), P);
                    __m256d T1 = _mm256_mul_pd (_mm256_loadAA_pd(y+k), P);
                    __m256d L = _mm256_unpacklo_pd(T0,T1);
                    __m256d H = _mm256_unpackhi_pd(T0,T1);
                    S = _mm_add_pd (cast128(L), S);
                    S = _mm_add_pd (cast128(H), S);
                    S = _mm_add_pd (_mm256_extractf128_pd(L,1), S);
                    S = _mm_add_pd (_mm256_extractf128_pd(H,1), S);
                    p += 4;
                    y += 4;
                    rows2 -= 4;
                }
            }
#           else  /* CAN_USE_SSE2 */
            {
                if (k & 1) {  /* second column not aligned if first is */
                    while (rows2 >= 4) {
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
                        rows2 -= 4;
                    }
                }
                else {  /* second column has same 16-byte alignment as first */
                    while (rows2 >= 4) {
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
                        rows2 -= 4;
                    }
                }
            }
#           endif

            if (rows2 > 1) {
                __m128d P = _mm_loadA_pd(p);
                __m128d T0 = _mm_mul_pd (_mm_loadA_pd(y), P);
                __m128d T1 = _mm_mul_pd (_mm_loadu_pd(y+k), P);
                S = _mm_add_pd (_mm_unpacklo_pd(T0,T1), S);
                S = _mm_add_pd (_mm_unpackhi_pd(T0,T1), S);
                p += 2;
                y += 2;
                rows2 -= 2;
            }

            if (rows2 >= 1) {
                __m128d B;
                B = _mm_mul_pd (_mm_set1_pd(p[0]), _mm_set_pd(y[k],y[0]));
                S = _mm_add_pd (B, S);
                y += 1;
            }

            _mm_storeu_pd (z, S);
        }

#       else  /* non-SIMD code */
        {
            int rows2 = rows;
            double s[2];
            if (add) {
                s[0] = z[0];
                s[1] = z[1];
            }
            else
                s[0] = s[1] = 0.0;

            p = x;

            while (rows2 > 1) {
                s[0] += p[0] * y[0];
                s[1] += p[0] * y[k];
                s[0] += p[1] * y[1];
                s[1] += p[1] * y[k+1];
                p += 2;
                y += 2;
                rows2 -= 2;
            }

            if (rows2 >= 1) {
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

        int rows2 = rows;
        double *p = x;
        double s;

        if (add)
            s = z[0];
        else
            s = 0.0;

        while (rows2 >= 4) {
            s += p[0] * y[0];
            s += p[1] * y[1];
            s += p[2] * y[2];
            s += p[3] * y[3];
            p += 4;
            y += 4;
            rows2 -= 4;
        }

        if (rows2 > 1) {
            s += p[0] * y[0];
            s += p[1] * y[1];
            p += 2;
            y += 2;
            rows2 -= 2;
        }

        if (rows2 >= 1) {
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

   If x has more than MAT_VEC_ROWS (defined below) rows, the operation
   is done on successive parts of the matrix, each consisting of at
   most MAT_VEC_ROWS of the rows.  The matprod_sub_mat_vec procedure
   below does one such part.

   Cases where k is 0 or 1 and cases where n is 2 are handled specially. */

static void matprod_sub_mat_vec (double * MATPROD_RESTRICT x, 
                                 double * MATPROD_RESTRICT y, 
                                 double * MATPROD_RESTRICT z, 
                                 int n, int k, int rows);

void matprod_mat_vec (double * MATPROD_RESTRICT x, 
                      double * MATPROD_RESTRICT y, 
                      double * MATPROD_RESTRICT z, int n, int k)
{
    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    if (n <= 1) {
        if (n == 1)
            z[0] = matprod_vec_vec (x, y, k);
        return;
    }

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

            double s[2] = { 0, 0 };  /* sums for the two values in the result */

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

        double s[3] = { 0, 0, 0 }; /* sums for the three values in the result */

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

    /* The general case with n > 3.  Calls matprod_sub_mat_vec to do parts
       (only one part for a matrix with fewer than MAT_VEC_ROWS rows). */

#   define MAT_VEC_ROWS (1024+256) /* be multiple of 64 to keep any alignment */

    if (n <= MAT_VEC_ROWS)
        matprod_sub_mat_vec (x, y, z, n, k, n);
    else {
        int rows = n;
        while (rows >= 2*MAT_VEC_ROWS) {
            matprod_sub_mat_vec (x, y, z, n, k, MAT_VEC_ROWS);
            x += MAT_VEC_ROWS;
            z += MAT_VEC_ROWS;
            rows -= MAT_VEC_ROWS;
        }
        if (rows > MAT_VEC_ROWS) {
            int nr = (rows/2) & ~7; /* ensure any alignment of x, z preserved */
            matprod_sub_mat_vec (x, y, z, n, k, nr);
            x += nr;
            z += nr;
            rows -= nr;
        }
        matprod_sub_mat_vec (x, y, z, n, k, rows);
    }
}


/* Multiply the first 'rows' of x with the elements of y, storing the
   result in z.  Note that x and z may not be the start of the
   original matrix/vector.  The k argument is the number of columns in
   x and elements in y.  The n argument is the number of rows in the
   original matrix x, which is the amount to step to go right to an
   element in the same row and the next column. 

   The same alignment assumptions hold for x, y, and z as with the
   visible procedures.

   Note that k will be at least 2, and 'rows' will be at least 4. */

static void matprod_sub_mat_vec (double * MATPROD_RESTRICT x, 
                                 double * MATPROD_RESTRICT y, 
                                 double * MATPROD_RESTRICT z,
                                 int n, int k, int rows)
{
    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    /* To start, set the result, z, to the sum from the first two
       columns of x times elements of y.  A few initial products may
       be done to help alignment.  This is also done later, in the
       same way, allowing reuse of n2. */

    int n2;     /* number of elements in z after those done to help alignment */
    int n3;
    double *q;  /* pointer going along z */
    double *xs; /* starting x pointer */

    q = z;
    n2 = rows;
    xs = x;

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

#   if CAN_USE_AVX || CAN_USE_SSE2
    {
#       if CAN_USE_AVX
            __m256d Y0 = _mm256_set1_pd(y[0]);
            __m256d Y1 = _mm256_set1_pd(y[1]);
#       else  /* CAN_USE_SSE2 */
            __m128d Y0 = _mm_set1_pd(y[0]);
            __m128d Y1 = _mm_set1_pd(y[1]);
#       endif
        while (n3 >= 4) { 
#           if CAN_USE_AVX
                __m256d X = _mm256_mul_pd (_mm256_loadA_pd(x), Y0);
                __m256d P = _mm256_mul_pd (_mm256_loadu_pd(x+n), Y1);
                _mm256_storeA_pd (q, _mm256_add_pd (X, P));
#           else  /* CAN_USE_SSE2 */
                __m128d X, P;
                X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
                P = _mm_mul_pd (_mm_loadu_pd(x+n), Y1);
                _mm_storeA_pd (q, _mm_add_pd (X, P));
                X = _mm_mul_pd (_mm_loadA_pd(x+2), Y0);
                P = _mm_mul_pd (_mm_loadu_pd(x+n+2), Y1);
                _mm_storeA_pd (q+2, _mm_add_pd (X, P));
#           endif
            x += 4;
            q += 4;
            n3 -= 4;
        }
        if (n3 > 1) {
            __m128d X = _mm_mul_pd (_mm_loadA_pd(x), cast128(Y0));
            __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), cast128(Y1));
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

    xs += 2*n;
    x = xs;
    y += 2;
    k -= 2;

    /* Each time around this loop, add the products of two columns of x 
       with two elements of y to the result vector, z.  Adjust x and y
       to account for this. */

    while (k > 1) {

        q = z;
        xs = x;

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

#       if CAN_USE_AVX || CAN_USE_SSE2
        {
#           if CAN_USE_AVX
                __m256d Y0 = _mm256_set1_pd(y[0]);
                __m256d Y1 = _mm256_set1_pd(y[1]);
#           else  /* CAN_USE_SSE2 */
                __m128d Y0 = _mm_load1_pd(y);
                __m128d Y1 = _mm_load1_pd(y+1);
#           endif
            while (n3 >= 4) { 
#               if CAN_USE_AVX
                {
                    __m256d Q = _mm256_loadA_pd(q);
                    __m256d X = _mm256_mul_pd (_mm256_loadA_pd(x), Y0);
                    __m256d P = _mm256_mul_pd (_mm256_loadu_pd(x+n), Y1);
                    Q = _mm256_add_pd (_mm256_add_pd (Q, X), P);
                    _mm256_storeA_pd (q, Q);
                }
#               else  /* CAN_USE_SSE2 */
                {
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
                }
#               endif
                x += 4;
                q += 4;
                n3 -= 4;
            }
            if (n3 > 1) {
                __m128d Q = _mm_loadA_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadA_pd(x), cast128(Y0));
                __m128d P = _mm_mul_pd (_mm_loadu_pd(x+n), cast128(Y1));
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

        xs += 2*n;
        x = xs;
        y += 2;
        k -= 2;
    }

    /* Add the last column if there are an odd number of columns. */

    if (k >= 1) {

        q = z;

#       if ALIGN_FORWARD & 8
        {
            q[0] += x[0] * y[0];
            x += 1;
            q += 1;
        }
#       endif

#       if CAN_USE_AVX && (ALIGN_FORWARD & 16)
        {
            __m128d Y0 = _mm_set1_pd(y[0]);
            __m128d Q = _mm_loadA_pd(q);
            __m128d X = _mm_mul_pd (_mm_loadA_pd(x), Y0);
            Q = _mm_add_pd (Q, X);
            _mm_storeA_pd (q, Q);
            x += 2;
            q += 2;
        }
#       endif

        n3 = n2;

#       if CAN_USE_AVX || CAN_USE_SSE2
        {
#           if CAN_USE_AVX
                __m256d Y = _mm256_set1_pd(y[0]);
#           else
                __m128d Y = _mm_set1_pd(y[0]);
#           endif
            while (n3 >= 4) { 
#               if CAN_USE_AVX
                {
                    __m256d Q = _mm256_loadA_pd(q);
                    __m256d X = _mm256_mul_pd (_mm256_loadA_pd(x), Y);
                    Q = _mm256_add_pd (Q, X);
                    _mm256_storeA_pd (q, Q);
                }
#               else  /* CAN_USE_SSE2 */
                {
                    __m128d Q, X;
                    Q = _mm_loadA_pd(q);
                    X = _mm_mul_pd (_mm_loadA_pd(x), Y);
                    Q = _mm_add_pd (Q, X);
                    _mm_storeA_pd (q, Q);
                    Q = _mm_loadA_pd(q+2);
                    X = _mm_mul_pd (_mm_loadA_pd(x+2), Y);
                    Q = _mm_add_pd (Q, X);
                    _mm_storeA_pd (q+2, Q);
                }
#               endif
                x += 4;
                q += 4;
                n3 -= 4;
            }
            if (n3 > 1) {
                __m128d Q = _mm_loadA_pd(q);
                __m128d X = _mm_mul_pd (_mm_loadA_pd(x), cast128(Y));
                Q = _mm_add_pd (Q, X);
                _mm_storeA_pd (q, Q);
                x += 2;
                q += 2;
                n3 -= 2;
            }
        }
#       else  /* non-SIMD code */
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
    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

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
                _mm_storeu_pd (z, _mm_mul_pd (cast128(Xa), Y));
                _mm_store_sd (z+2, _mm_mul_sd (_mm256_extractf128_pd(Xa,1), Y));
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
                _mm_storeu_pd (z, _mm_mul_pd (cast128(X), Y));
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
    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    if (n <= 1) {
        if (n == 1)
            matprod_vec_mat (x, y, z, k, m);
        return;
    }
    if (m <= 1) {
        if (m == 1)
            matprod_mat_vec (x, y, z, n, k);
        return;
    }

    if (k <= 1) {
        if (k == 1)
            matprod_outer (x, y, z, n, m);
        else
            set_to_zeros (z, n*m);
        return;
    }

    /* Handle n=2 specially. */

    if (n == 2) {

        /* Compute two columns of the result each time around this loop, 
           updating y, z, and m accordingly. */

        while (m > 1) {

#           if CAN_USE_SSE2

                double *r = x;
                int k2 = k;

                __m128d S0 = _mm_setzero_pd();
                __m128d S1 = _mm_setzero_pd();

                /* Each time around this loop, add the products of two columns
                   of x with elements of the next two columns of y to the sums.
                   Adjust r and y to account for this. */

                while (k2 > 1) {
                    __m128d R;
                    R = _mm_loadAA_pd(r);
                    S0 = _mm_add_pd (_mm_mul_pd (R, _mm_set1_pd(y[0])), S0);
                    S1 = _mm_add_pd (_mm_mul_pd (R, _mm_set1_pd(y[k])), S1);
                    R = _mm_loadAA_pd(r+2);
                    S0 = _mm_add_pd (_mm_mul_pd (R, _mm_set1_pd(y[1])), S0);
                    S1 = _mm_add_pd (_mm_mul_pd (R, _mm_set1_pd(y[k+1])), S1);
                    r += 4;
                    y += 2;
                    k2 -= 2;
                }

                if (k2 >= 1) {
                    __m128d R;
                    R = _mm_loadAA_pd(r);
                    S0 = _mm_add_pd (_mm_mul_pd (R, _mm_set1_pd(y[0])), S0);
                    S1 = _mm_add_pd (_mm_mul_pd (R, _mm_set1_pd(y[k])), S1);
                    y += 1;
                }

                /* Store sums in the next two result columns. */

                _mm_storeAA_pd (z, S0);
                _mm_storeAA_pd (z+2, S1);

#           else  /* non-SIMD code */

                double s[4] = { 0, 0, 0, 0 };
                double *r = x;
                int k2 = k;

                /* Each time around this loop, add the products of two columns
                   of x with elements of the next two columns of y to the sums.
                   Adjust r and y to account for this. */

                while (k2 > 1) {
                    double b11 = y[0];
                    double b12 = y[1];
                    double b21 = y[k];
                    double b22 = y[k+1];
                    s[0] = (s[0] + (r[0] * b11)) + (r[2] * b12);
                    s[1] = (s[1] + (r[1] * b11)) + (r[3] * b12);
                    s[2] = (s[2] + (r[0] * b21)) + (r[2] * b22);
                    s[3] = (s[3] + (r[1] * b21)) + (r[3] * b22);
                    r += 4;
                    y += 2;
                    k2 -= 2;
                }

                if (k2 >= 1) {
                    double b1 = y[0];
                    double b2 = y[k];
                    s[0] += r[0] * b1;
                    s[1] += r[1] * b1;
                    s[2] += r[0] * b2;
                    s[3] += r[1] * b2;
                    y += 1;
                }

                /* Store sums in the next two result columns. */

                z[0] = s[0];
                z[1] = s[1];
                z[2] = s[2];
                z[3] = s[3];

#           endif

            /* Move forward two to next column of the result and the next
               column of y. */

            y += k;
            z += 4;
            m -= 2;
        }

        /* If m is odd, compute the last column of the result. */

        if (m >= 1) {

            double s[2] = { 0, 0 };  /* sums for the two values in the result */
            double *r = x;
            int k2 = k;

            /* Each time around this loop, add the products of two
               columns of x with two elements of the last column of y
               to s[0] and s[1]. */

            while (k2 > 1) {
                double b1 = y[0];
                double b2 = y[1];
                s[0] = (s[0] + (r[0] * b1)) + (r[2] * b2);
                s[1] = (s[1] + (r[1] * b1)) + (r[3] * b2);
                r += 4;
                y += 2;
                k2 -= 2;
            }

            if (k2 >= 1) {
                double b = y[0];
                s[0] += r[0] * b;
                s[1] += r[1] * b;
                /* y += 1; */
            }

            /* Store the two sums in s[0] and s[1] in the result vector. */

            z[0] = s[0];
            z[1] = s[1];
        }

        return;
    }

    /* The general case.  Computes two columns of the result each time
       around this loop, updating y, z, and m accordingly. */

    while (m > 1) {

        double *r = x;  /* r set to x, then modified as columns of x summed */
        int k2;

        /* Initialize sums in next two columns of z to the sum of the
           first two products, which will exist, since k is at least
           two here.  Note also that n is at least three. */

        int n2 = n;

#       if CAN_USE_AVX
        {
            __m256d B11 = _mm256_set1_pd(y[0]);
            __m256d B12 = _mm256_set1_pd(y[1]);
            __m256d B21 = _mm256_set1_pd(y[k]);
            __m256d B22 = _mm256_set1_pd(y[k+1]);
            double *q = z;
#           if ALIGN_FORWARD & 8
            {
                __m128d S1 = _mm_set_sd(r[0]);
                __m128d S2 = _mm_set_sd(r[n]);
                _mm_store_sd (q, _mm_add_sd (
                    _mm_mul_sd(S1,cast128(B11)),
                    _mm_mul_sd(S2,cast128(B12))));
                _mm_store_sd (q+n, _mm_add_sd (
                    _mm_mul_sd(S1,cast128(B21)),
                    _mm_mul_sd(S2,cast128(B22))));
                r += 1;
                q += 1;
                n2 -= 1;
            }
#           endif
#           if ALIGN_FORWARD & 16
            {
                __m128d S1 = _mm_loadA_pd(r);
                __m128d S2 = _mm_loadu_pd(r+n);
                _mm_storeA_pd (q, _mm_add_pd (
                    _mm_mul_pd(S1,cast128(B11)),
                    _mm_mul_pd(S2,cast128(B12))));
                _mm_storeu_pd (q+n, _mm_add_pd (
                    _mm_mul_pd(S1,cast128(B21)),
                    _mm_mul_pd(S2,cast128(B22))));
                r += 2;
                q += 2;
                n2 -= 2;
            }
#           endif
            while (n2 >= 4) {
                __m256d S1 = _mm256_loadA_pd(r);
                __m256d S2 = _mm256_loadu_pd(r+n);
                _mm256_storeu_pd (q, _mm256_add_pd (_mm256_mul_pd(S1,B11),
                                                    _mm256_mul_pd(S2,B12)));
                _mm256_storeu_pd (q+n, _mm256_add_pd (_mm256_mul_pd(S1,B21),
                                                      _mm256_mul_pd(S2,B22)));
                r += 4;
                q += 4;
                n2 -= 4;
            }
            if (n2 > 1) {
                __m128d S1 = _mm_loadA_pd(r);
                __m128d S2 = _mm_loadu_pd(r+n);
                _mm_storeA_pd (q, _mm_add_pd (
                    _mm_mul_pd(S1,cast128(B11)),
                    _mm_mul_pd(S2,cast128(B12))));
                _mm_storeu_pd (q+n, _mm_add_pd (
                    _mm_mul_pd(S1,cast128(B21)),
                    _mm_mul_pd(S2,cast128(B22))));
                r += 2;
                q += 2;
                n2 -= 2;
            }
            if (n2 >= 1) {
                __m128d S1 = _mm_set_sd(r[0]);
                __m128d S2 = _mm_set_sd(r[n]);
                _mm_store_sd (q, _mm_add_sd (
                    _mm_mul_sd(S1,cast128(B11)),
                    _mm_mul_sd(S2,cast128(B12))));
                _mm_store_sd (q+n, _mm_add_sd (
                    _mm_mul_sd(S1,cast128(B21)),
                    _mm_mul_sd(S2,cast128(B22))));
                r += 1;
            }
        }
#       elif CAN_USE_SSE2
        {
            __m128d B11 = _mm_set1_pd(y[0]);
            __m128d B12 = _mm_set1_pd(y[1]);
            __m128d B21 = _mm_set1_pd(y[k]);
            __m128d B22 = _mm_set1_pd(y[k+1]);
            double *q = z;
#           if ALIGN_FORWARD & 8
                __m128d S1 = _mm_set_sd(r[0]);
                __m128d S2 = _mm_set_sd(r[n]);
                _mm_store_sd (q, _mm_add_sd (_mm_mul_sd(S1,B11),
                                             _mm_mul_sd(S2,B12)));
                _mm_store_sd (q+n, _mm_add_sd (_mm_mul_sd(S1,B21),
                                               _mm_mul_sd(S2,B22)));
                r += 1;
                q += 1;
                n2 -= 1;
#           endif
            while (n2 > 1) {
                __m128d S1 = _mm_loadA_pd(r);
                __m128d S2 = _mm_loadu_pd(r+n);
                _mm_storeA_pd (q, _mm_add_pd (_mm_mul_pd(S1,B11),
                                               _mm_mul_pd(S2,B12)));
                _mm_storeu_pd (q+n, _mm_add_pd (_mm_mul_pd(S1,B21),
                                                _mm_mul_pd(S2,B22)));
                r += 2;
                q += 2;
                n2 -= 2;
            }
            if (n2 >= 1) {
                __m128d S1 = _mm_set_sd(r[0]);
                __m128d S2 = _mm_set_sd(r[n]);
                _mm_store_sd (q, _mm_add_sd (_mm_mul_sd(S1,B11),
                                             _mm_mul_sd(S2,B12)));
                _mm_store_sd (q+n, _mm_add_sd (_mm_mul_sd(S1,B21),
                                               _mm_mul_sd(S2,B22)));
                r += 1;
            }
        }
#       else
        {
            double b11 = y[0];
            double b12 = y[1];
            double b21 = y[k];
            double b22 = y[k+1];
            double *q = z;
            do {
                double s1 = r[0];
                double s2 = r[n];
                q[0] = (s1 * b11) + (s2 * b12);
                q[n] = (s1 * b21) + (s2 * b22);
                r += 1;
                q += 1;
                n2 -= 1;
            } while (n2 > 0);
        }
#       endif

        r += n;  /* already advanced by n, so total advance is 2*n */
        y += 2;
        k2 = k-2;

        /* Each time around this loop, add the products of two columns of x 
           with two elements of the next two columns of y to the next two
           columns of the result vector, z.  Adjust r and y to account 
           for this. */

        while (k2 > 1) {
            int n2 = n;
#           if CAN_USE_AVX
            {
                __m256d B11 = _mm256_set1_pd(y[0]);
                __m256d B12 = _mm256_set1_pd(y[1]);
                __m256d B21 = _mm256_set1_pd(y[k]);
                __m256d B22 = _mm256_set1_pd(y[k+1]);
                double *q = z;
#               if ALIGN_FORWARD & 8
                {
                    __m128d S1 = _mm_set_sd(r[0]);
                    __m128d S2 = _mm_set_sd(r[n]);
                    _mm_store_sd (q, _mm_add_sd (_mm_add_sd (_mm_load_sd(q),
                        _mm_mul_sd(S1,cast128(B11))),
                        _mm_mul_sd(S2,cast128(B12))));
                    _mm_store_sd (q+n, _mm_add_sd (_mm_add_sd (_mm_load_sd(q+n),
                        _mm_mul_sd(S1,cast128(B21))),
                        _mm_mul_sd(S2,cast128(B22))));
                    r += 1;
                    q += 1;
                    n2 -= 1;
                }
#               endif
#               if ALIGN_FORWARD & 16
                {
                    __m128d S1 = _mm_loadA_pd(r);
                    __m128d S2 = _mm_loadu_pd(r+n);
                    _mm_storeA_pd (q, _mm_add_pd (_mm_add_pd (_mm_loadA_pd(q),
                        _mm_mul_pd(S1,cast128(B11))),
                        _mm_mul_pd(S2,cast128(B12))));
                    _mm_storeu_pd(q+n,_mm_add_pd (_mm_add_pd (_mm_loadu_pd(q+n),
                        _mm_mul_pd(S1,cast128(B21))),
                        _mm_mul_pd(S2,cast128(B22))));
                    r += 2;
                    q += 2;
                    n2 -= 2;
                }
#               endif
                while (n2 >= 4) {
                    __m256d S1 = _mm256_loadu_pd(r);
                    __m256d S2 = _mm256_loadu_pd(r+n);
                    _mm256_storeu_pd (q, _mm256_add_pd (_mm256_add_pd (
                                     _mm256_loadA_pd(q),_mm256_mul_pd(S1,B11)),
                                                        _mm256_mul_pd(S2,B12)));
                    _mm256_storeu_pd(q+n,_mm256_add_pd (_mm256_add_pd (
                                   _mm256_loadA_pd(q+n),_mm256_mul_pd(S1,B21)),
                                                        _mm256_mul_pd(S2,B22)));
                    r += 4;
                    q += 4;
                    n2 -= 4;
                }
                if (n2 > 1) {
                    __m128d S1 = _mm_loadA_pd(r);
                    __m128d S2 = _mm_loadu_pd(r+n);
                    _mm_storeA_pd (q, _mm_add_pd (_mm_add_pd (_mm_loadA_pd(q),
                        _mm_mul_pd(S1,cast128(B11))),
                        _mm_mul_pd(S2,cast128(B12))));
                    _mm_storeu_pd(q+n,_mm_add_pd (_mm_add_pd (_mm_loadu_pd(q+n),
                        _mm_mul_pd(S1,cast128(B21))),
                        _mm_mul_pd(S2,cast128(B22))));
                    r += 2;
                    q += 2;
                    n2 -= 2;
                }
                if (n2 >= 1) {
                    __m128d S1 = _mm_set_sd(r[0]);
                    __m128d S2 = _mm_set_sd(r[n]);
                    _mm_store_sd (q, _mm_add_sd (_mm_add_sd (_mm_load_sd(q),
                        _mm_mul_sd(S1,cast128(B11))),
                        _mm_mul_sd(S2,cast128(B12))));
                    _mm_store_sd (q+n, _mm_add_sd (_mm_add_sd (_mm_load_sd(q+n),
                        _mm_mul_sd(S1,cast128(B21))),
                        _mm_mul_sd(S2,cast128(B22))));
                    r += 1;
                }
            }
#           elif CAN_USE_SSE2
            {
                __m128d B11 = _mm_set1_pd(y[0]);
                __m128d B12 = _mm_set1_pd(y[1]);
                __m128d B21 = _mm_set1_pd(y[k]);
                __m128d B22 = _mm_set1_pd(y[k+1]);
                double *q = z;
#               if ALIGN_FORWARD & 8
                {
                    __m128d S1 = _mm_set_sd(r[0]);
                    __m128d S2 = _mm_set_sd(r[n]);
                    _mm_store_sd (q, _mm_add_sd (_mm_add_sd (_mm_load_sd(q),
                                                   _mm_mul_sd(S1,B11)),
                                                   _mm_mul_sd(S2,B12)));
                    _mm_store_sd (q+n, _mm_add_sd (_mm_add_sd (_mm_load_sd(q+n),
                                                   _mm_mul_sd(S1,B21)),
                                                   _mm_mul_sd(S2,B22)));
                    r += 1;
                    q += 1;
                    n2 -= 1;
                }
#               endif
                while (n2 > 1) {
                    __m128d S1 = _mm_loadA_pd(r);
                    __m128d S2 = _mm_loadu_pd(r+n);
                    _mm_storeA_pd (q, _mm_add_pd (_mm_add_pd (_mm_loadA_pd(q),
                                                   _mm_mul_pd(S1,B11)),
                                                   _mm_mul_pd(S2,B12)));
                    _mm_storeu_pd(q+n,_mm_add_pd (_mm_add_pd (_mm_loadu_pd(q+n),
                                                    _mm_mul_pd(S1,B21)),
                                                    _mm_mul_pd(S2,B22)));
                    r += 2;
                    q += 2;
                    n2 -= 2;
                }
                if (n2 >= 1) {
                    __m128d S1 = _mm_set_sd(r[0]);
                    __m128d S2 = _mm_set_sd(r[n]);
                    _mm_store_sd (q, _mm_add_sd (_mm_add_sd (_mm_load_sd(q),
                                                   _mm_mul_sd(S1,B11)),
                                                   _mm_mul_sd(S2,B12)));
                    _mm_store_sd (q+n, _mm_add_sd (_mm_add_sd (_mm_load_sd(q+n),
                                                   _mm_mul_sd(S1,B21)),
                                                   _mm_mul_sd(S2,B22)));
                    r += 1;
                }
            }
#           else
            {
                double b11 = y[0];
                double b12 = y[1];
                double b21 = y[k];
                double b22 = y[k+1];
                double s1, s2;
                double *q = z;
                while (n2 > 1) {
                    s1 = r[0];
                    s2 = r[n];
                    q[0] = (q[0] + (s1 * b11)) + (s2 * b12);
                    q[n] = (q[n] + (s1 * b21)) + (s2 * b22);
                    s1 = r[1];
                    s2 = r[n+1];
                    q[1] = (q[1] + (s1 * b11)) + (s2 * b12);
                    q[n+1] = (q[n+1] + (s1 * b21)) + (s2 * b22);
                    r += 2;
                    q += 2;
                    n2 -= 2;
                }
                if (n2 >= 1) {
                    s1 = r[0];
                    s2 = r[n];
                    q[0] = (q[0] + (s1 * b11)) + (s2 * b12);
                    q[n] = (q[n] + (s1 * b21)) + (s2 * b22);
                    r += 1;
                }
            }
#           endif
            r += n;  /* already advanced by n, so total advance is 2*n */
            y += 2;
            k2 -= 2;
        }

        if (k2 >= 1) {
            double b1 = y[0];
            double b2 = y[k];
            double *q = z;
            int n2 = n;
            do {
                double s = r[0];
                q[0] += s * b1;
                q[n] += s * b2;
                r += 1;
                q += 1;
                n2 -= 1;
            } while (n2 > 0);
            y += 1;
        }

        /* Move to the next two columns. */

        y += k;  /* already advanced by k, so total advance is 2*k */
        z += 2*n;
        m -= 2;
    }

    /* If m is odd, compute the last column of the result. */

    if (m >= 1) {

        double *r = x;    /* r set to x, then modified as columns of x summed */
        double *e = y+k;  /* where y should stop */

        /* Initialize sums in z to zero, if k is even, or to the product of
           the first element of the next column of y with the first column 
           of x (in which case adjust r and y accordingly). */

        if (k & 1) {
            double b = y[0];
            double *q = z;
            int n2 = n;
            do { 
                *q++ = *r++ * b; 
                n2 -= 1;
            } while (n2 > 0);
            y += 1;
        }
        else {
            double *q = z;
            int n2 = n;
            do {
                *q++ = 0.0;
                n2 -= 1;
            } while (n2 > 0);
        }

        /* Each time around this loop, add the products of two columns of x 
           with two elements of the next column of y to the result vector, z.
           Adjust r and y to account for this.  Note that e-y will be even 
           when we start. */

        while (y < e) {
            double b1 = y[0];
            double b2 = y[1];
            double *q = z;
            int n2 = n;
            do {
                *q = (*q + (*r * b1)) + (*(r+n) * b2);
                r += 1;
                q += 1;
                n2 -= 1;
            } while (n2 > 0);
            r += n;
            y += 2;
        }
    }
}


/* Product of the transpose of a k x n matrix (x) and a k x m matrix (y) 
   with result stored in z.  

   Each element of the result is the dot product of a column of x and
   a column of y.  Eight elements of this result are computed at once,
   using four consecutive columns of x and two consecutive columns of
   y (except perhaps for odd columns at the end), thereby reducing the
   number of memory accesses.

   The case of k=2 is handled specially.

   When the two operands are the same, the result will be a symmetric
   matrix.  During computation of each column or pair of columns, elements
   are copied to the corresponding rows; hence each column need be
   computed only from the diagonal element down. */

void matprod_trans1 (double * MATPROD_RESTRICT x, 
                     double * MATPROD_RESTRICT y, 
                     double * MATPROD_RESTRICT z, int n, int k, int m)
{
    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    if (n <= 1) {
        if (n == 1)
            matprod_vec_mat (x, y, z, k, m);
        return;
    }
    if (m <= 1) {
        if (m == 1)
            matprod_vec_mat (y, x, z, k, n);
        return;
    }

    if (k <= 1) {
        if (k == 1)
            matprod_outer (x, y, z, n, m);
        else
            set_to_zeros (z, n*m);
        return;
    }

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
                    __m128d M0 = _mm_mul_pd (cast128(Y), _mm_loadu_pd(p));
                    __m128d M1 = _mm_mul_pd (cast128(Y), _mm_loadu_pd(p+2));
                    _mm_storeu_pd (z, _mm_hadd_pd (M0, M1));
                    p += 4;
                    z += 2;
                }
                if (n & 1) {
                    __m128d M0 = _mm_mul_pd (cast128(Y), _mm_loadu_pd(p));
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
            nn -= jj;
            r += jj*k;
            rz = z;
        }

        /* Compute sets of four elements in the two columns being
           computed.  Copy them to the corresponding rows too, if the
           result is symmetric. */

        while (nn >= 4) {

#           if CAN_USE_AVX
            {
                double *q, *qe;
                __m256d S0, /* sums for first two cols of x times 2 cols of y */
                        S1; /* sums for last two cols of x times 2 cols of y */
#               if ALIGN_FORWARD & 8
                    __m256d Y = _mm256_set_pd (y[k], y[k], y[0], y[0]);
                    __m256d X;
                    X = _mm256_set_pd (r[k], r[0], r[k], r[0]);
                    S0 = _mm256_mul_pd(X,Y);
                         /* ie, _mm256_set_pd 
                                 (r[k]*y[k], r[0]*y[k], r[k]*y[0], r[0]*y[0]) */
                    X = _mm256_set_pd (r[3*k], r[2*k], r[3*k], r[2*k]);
                    S1 = _mm256_mul_pd(X,Y);
                    r += 1;
                    q = y+1;
                    qe = q+((k-1)-3);
#               else
                    S0 = _mm256_setzero_pd();
                    S1 = _mm256_setzero_pd();
                    q = y;
                    qe = q+(k-3);
#               endif
                while (q < qe) {
                    __m256d R0, Rk, M00, M0k, Mk0, Mkk, L0, H0, Lk, Hk;
                    __m256d Q0 = _mm256_loadu_pd(q);
                    __m256d Qk = _mm256_loadu_pd(q+k);
                    R0 = _mm256_loadu_pd(r);
                    Rk = _mm256_loadu_pd(r+k);
                    M00 = _mm256_mul_pd (Q0, R0);
                    M0k = _mm256_mul_pd (Q0, Rk);
                    Mk0 = _mm256_mul_pd (Qk, R0);
                    Mkk = _mm256_mul_pd (Qk, Rk);
                    L0 = _mm256_unpacklo_pd (M00, M0k);
                    H0 = _mm256_unpackhi_pd (M00, M0k);
                    Lk = _mm256_unpacklo_pd (Mk0, Mkk);
                    Hk = _mm256_unpackhi_pd (Mk0, Mkk);
                    S0 = _mm256_add_pd(S0,_mm256_permute2f128_pd(L0, Lk, 0x20));
                    S0 = _mm256_add_pd(S0,_mm256_permute2f128_pd(H0, Hk, 0x20));
                    S0 = _mm256_add_pd(S0,_mm256_permute2f128_pd(L0, Lk, 0x31));
                    S0 = _mm256_add_pd(S0,_mm256_permute2f128_pd(H0, Hk, 0x31));
                    R0 = _mm256_loadu_pd(r+2*k);
                    Rk = _mm256_loadu_pd(r+3*k);
                    M00 = _mm256_mul_pd (Q0, R0);
                    M0k = _mm256_mul_pd (Q0, Rk);
                    Mk0 = _mm256_mul_pd (Qk, R0);
                    Mkk = _mm256_mul_pd (Qk, Rk);
                    L0 = _mm256_unpacklo_pd (M00, M0k);
                    H0 = _mm256_unpackhi_pd (M00, M0k);
                    Lk = _mm256_unpacklo_pd (Mk0, Mkk);
                    Hk = _mm256_unpackhi_pd (Mk0, Mkk);
                    S1 = _mm256_add_pd(S1,_mm256_permute2f128_pd(L0, Lk, 0x20));
                    S1 = _mm256_add_pd(S1,_mm256_permute2f128_pd(H0, Hk, 0x20));
                    S1 = _mm256_add_pd(S1,_mm256_permute2f128_pd(L0, Lk, 0x31));
                    S1 = _mm256_add_pd(S1,_mm256_permute2f128_pd(H0, Hk, 0x31));
                    r += 4;
                    q += 4;
                }
                qe = y+k;
                while (q < qe) {
                    __m256d Q = _mm256_set_pd (q[k], q[k], q[0], q[0]);
                    __m256d X;
                    X = _mm256_set_pd (r[k], r[0], r[k], r[0]);
                    S0 = _mm256_add_pd (_mm256_mul_pd(X,Q), S0);
                    X = _mm256_set_pd (r[3*k], r[2*k], r[3*k], r[2*k]);
                    S1 = _mm256_add_pd (_mm256_mul_pd(X,Q), S1);
                    r += 1;
                    q += 1;
                }
                __m128d H0 = _mm256_extractf128_pd(S0,1);
                _mm_storeu_pd (z, cast128(S0));
                _mm_storeu_pd (z+n, H0);
                __m128d H1 = _mm256_extractf128_pd(S1,1);
                _mm_storeu_pd (z+2, cast128(S1));
                _mm_storeu_pd (z+n+2, H1);
                if (sym) {
                    _mm_storeu_pd (rz, _mm_unpacklo_pd(cast128(S0),H0));
                    rz += n;
                    _mm_storeu_pd (rz, _mm_unpackhi_pd(cast128(S0),H0));
                    rz += n;
                    _mm_storeu_pd (rz, _mm_unpacklo_pd(cast128(S1),H1));
                    rz += n;
                    _mm_storeu_pd (rz, _mm_unpackhi_pd(cast128(S1),H1));
                    rz += n;
                }
            }

#           elif CAN_USE_SSE2
            {
                double *q, *qe;
                __m128d S0,   /* sums for first two x cols times first y col */
                        S1,   /* sums for first two x cols times second y col */
                        S2,   /* sums for last two x cols times first y col */
                        S3;   /* sums for last two x cols times second y col */
#               if ALIGN_FORWARD & 8
                    __m128d X, Y0, Yk;
                    X = _mm_set_pd (r[k], r[0]);
                    Y0 = _mm_set_pd (y[0], y[0]);
                    Yk = _mm_set_pd (y[k], y[k]);
                    S0 = _mm_mul_pd(X,Y0);
                         /* ie, _mm_set_pd (r[k]*y[0], r[0]*y[0]) */
                    S1 = _mm_mul_pd(X,Yk);
                         /* ie, _mm_set_pd (r[k]*y[k], r[0]*y[k]) */
                    X = _mm_set_pd (r[3*k], r[2*k]);
                    S2 = _mm_mul_pd(X,Y0);
                         /* ie, _mm_set_pd (r[3*k]*y[0], r[2*k]*y[0]) */
                    S3 = _mm_mul_pd(X,Yk);
                         /* ie, _mm_set_pd (r[3*k]*y[k], r[2*k]*y[k]) */
                    r += 1;
                    q = y+1;
                    qe = q+((k-1)-1);
#               else
                    S0 = _mm_setzero_pd();
                    S1 = _mm_setzero_pd();
                    S2 = _mm_setzero_pd();
                    S3 = _mm_setzero_pd();
                    q = y;
                    qe = q+(k-1);
#               endif
                while (q < qe) {
                    __m128d R0, Rk, M00, M0k, Mk0, Mkk, L0, H0, Lk, Hk;
                    __m128d Q0 = _mm_loadA_pd(q);
                    __m128d Qk = _mm_loadu_pd(q+k);
                    R0 = _mm_loadA_pd(r);
                    Rk = _mm_loadu_pd(r+k);
                    M00 = _mm_mul_pd (Q0, R0);
                    M0k = _mm_mul_pd (Q0, Rk);
                    Mk0 = _mm_mul_pd (Qk, R0);
                    Mkk = _mm_mul_pd (Qk, Rk);
                    L0 = _mm_unpacklo_pd (M00, M0k);
                    H0 = _mm_unpackhi_pd (M00, M0k);
                    Lk = _mm_unpacklo_pd (Mk0, Mkk);
                    Hk = _mm_unpackhi_pd (Mk0, Mkk);
                    S0 = _mm_add_pd (S0, L0);
                    S0 = _mm_add_pd (S0, H0);
                    S1 = _mm_add_pd (S1, Lk);
                    S1 = _mm_add_pd (S1, Hk);
                    R0 = _mm_loadA_pd(r+2*k);
                    Rk = _mm_loadu_pd(r+3*k);
                    M00 = _mm_mul_pd (Q0, R0);
                    M0k = _mm_mul_pd (Q0, Rk);
                    Mk0 = _mm_mul_pd (Qk, R0);
                    Mkk = _mm_mul_pd (Qk, Rk);
                    L0 = _mm_unpacklo_pd (M00, M0k);
                    H0 = _mm_unpackhi_pd (M00, M0k);
                    Lk = _mm_unpacklo_pd (Mk0, Mkk);
                    Hk = _mm_unpackhi_pd (Mk0, Mkk);
                    S2 = _mm_add_pd (S2, L0);
                    S2 = _mm_add_pd (S2, H0);
                    S3 = _mm_add_pd (S3, Lk);
                    S3 = _mm_add_pd (S3, Hk);
                    r += 2;
                    q += 2;
                }
                if (q < y+k) {
                    __m128d X;
                    __m128d Q0 = _mm_set_pd (q[0], q[0]);
                    __m128d Qk = _mm_set_pd (q[k], q[k]);
                    X = _mm_set_pd (r[k], r[0]);
                    S0 = _mm_add_pd (_mm_mul_pd(X,Q0), S0);
                    S1 = _mm_add_pd (_mm_mul_pd(X,Qk), S1);
                    X = _mm_set_pd (r[3*k], r[2*k]);
                    S2 = _mm_add_pd (_mm_mul_pd(X,Q0), S2);
                    S3 = _mm_add_pd (_mm_mul_pd(X,Qk), S3);
                    r += 1;
                    q += 1;
                }
                _mm_storeAA_pd (z, S0);
                _mm_storeu_pd (z+n, S1);
                _mm_storeAA_pd (z+2, S2);
                _mm_storeu_pd (z+n+2, S3);
                if (sym) {
                    _mm_storeu_pd (rz, _mm_unpacklo_pd(S0,S1));
                    rz += n;
                    _mm_storeu_pd (rz, _mm_unpackhi_pd(S0,S1));
                    rz += n;
                    _mm_storeu_pd (rz, _mm_unpacklo_pd(S2,S3));
                    rz += n;
                    _mm_storeu_pd (rz, _mm_unpackhi_pd(S2,S3));
                    rz += n;
                }
            }

#           else  /* non-SIMD code */
            {
                double s[4] = { 0, 0, 0, 0 };
                double s2[4] = { 0, 0, 0, 0 };
                double *q = y;
                int i = k;
                do {
                    double t, t2;
                    double u = *q;
                    double u2 = *(q+k);
                    t = *r;
                    t2 = *(r+k);
                    s[0] += t * u;
                    s[1] += t * u2;
                    s[2] += t2 * u;
                    s[3] += t2 * u2;
                    t = *(r+2*k);
                    t2 = *(r+3*k);
                    s2[0] += t * u;
                    s2[1] += t * u2;
                    s2[2] += t2 * u;
                    s2[3] += t2 * u2;
                    r += 1;
                    q += 1;
                } while (--i > 0);
                z[0] = s[0];
                z[n] = s[1];
                z[1] = s[2];
                z[n+1] = s[3];
                z[2] = s2[0];
                z[n+2] = s2[1];
                z[3] = s2[2];
                z[n+3] = s2[3];
                if (sym) {
                    rz[0] = s[0];
                    rz[1] = s[1];
                    rz[n] = s[2];
                    rz[n+1] = s[3];
                    rz[2*n] = s2[0];
                    rz[2*n+1] = s2[1];
                    rz[3*n] = s2[2];
                    rz[3*n+1] = s2[3];
                    rz += 4*n;
                }
            }
#           endif

            z += 4;
            r += 3*k;
            nn -= 4;
        }

        /* Compute the remaining elements of the columns here.  If result
           is symmetric, store in the symmetric places as well. */

        if (nn > 1) { /* at least two more elements to compute in each column */
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
                _mm_storeu_pd (z, cast128(S));
                _mm_storeu_pd (z+n, H);
                if (sym) {
                    _mm_storeu_pd (rz, _mm_unpacklo_pd(cast128(S),H));
                    rz += n;
                    _mm_storeu_pd (rz, _mm_unpackhi_pd(cast128(S),H));
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
                _mm_storeu_pd (z+n, S1);
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
                z[n] = s[1];
                z[1] = s[2];
                z[n+1] = s[3];
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
            r += k;
            nn -= 2;
        }

        if (nn >= 1) { /* one more element to compute in each column */
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
            z[0] = s0;
            z[n] = s1;
            if (sym) {
                rz[0] = s0;
                rz[1] = s1;
                /* rz += n; */
            }
            z += 1;
            /* nn -= 1; */
        }

        /* Go on to next two columns of y. */

        z += n;
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
    CHK_ALIGN(x); CHK_ALIGN(y); CHK_ALIGN(z);

    x = ASSUME_ALIGNED (x, ALIGN, ALIGN_OFFSET);
    y = ASSUME_ALIGNED (y, ALIGN, ALIGN_OFFSET);
    z = ASSUME_ALIGNED (z, ALIGN, ALIGN_OFFSET);

    if (n <= 1) {
        if (n == 1)
            matprod_mat_vec (y, x, z, m, k);
        return;
    }
    if (m <= 1) {
        if (m == 1)
            matprod_mat_vec (x, y, z, n, k);
        return;
    }

    if (k <= 1) {
        if (k == 1)
            matprod_outer (x, y, z, n, m);
        else
            set_to_zeros (z, n*m);
        return;
    }

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

            double s[4];
            double *q = y;
            double *r = x;

            /* Initialize sums for columns to zero, if k is even, or to the 
               products of the first elements of the next two rows of y with
               the first column of x.  Adjust r and q accordingly. */

            if (k & 1) {
                double b = *q;
                double b2 = *(q+1);
                q += m;
                s[0] = r[0] * b;
                s[1] = r[1] * b;
                s[2] = r[0] * b2;
                s[3] = r[1] * b2;
                r += 2;
            }
            else
                s[0] = s[1] = s[2] = s[3] = 0.0;

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
                s[0] = (s[0] + (r[0] * b11)) + (r[2] * b12);
                s[1] = (s[1] + (r[1] * b11)) + (r[3] * b12);
                s[2] = (s[2] + (r[0] * b21)) + (r[2] * b22);
                s[3] = (s[3] + (r[1] * b21)) + (r[3] * b22);
                r += 4;
            }

            /* Store sums in the next two result columns. */

            z[0] = s[0];
            z[1] = s[1];
            z[2] = s[2];
            z[3] = s[3];

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
