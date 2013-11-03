/* MATPROD - A LIBRARY FOR MATRIX MULTIPLICATION WITH OPTIONAL PIPELINING
             C Procedures for Matrix Multiplication Without Pipelining

   Copyright (c) 2013 Radford M. Neal.

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


#include "matprod.h"


/* Dot product of two vectors of length k. 

   Two implementations are provided.  One uses the obvious loop, which
   maybe the compiler will optimize well.  In the other, the loop is
   unrolled to do two pairs of elements each iteration, with the sum
   initialized either to zero or to the result from the first pair, if
   the number of pairs is odd. 

   Use -DALT_MATPROD_VEC_VEC to switch between these two implementations.
   Change #ifdef to #ifndef or vice versa below to change the default. */

double matprod_vec_vec (double *x, double *y, int k)
{
#   ifdef ALT_MATPROD_VEC_VEC
    {
        double s;
        int i;

        s = 0.0;

        for (i = 0; i<k; i++)
            s += x[i]*y[i];

        return s;
    }
#   else
    {
        double s;

        /* If k is odd, initialize sum to the first product, and adjust x,
           y, and k to account for this.  If k is even, just initialize
           sum to zero. */

        if (k & 1) {
            s = *x++ * *y++;
            k -= 1;
        }
        else
            s = 0.0;

        /* Add two products each time around loop, adjusting x, y, and k as
           we go.  Note that k will be even when we start. */

        while (k > 0) {
            s += *x++ * *y++;
            s += *x++ * *y++;
            k -= 2;
        }

        return s;
    }
#   endif
}


/* Product of row vector (x) of length k and k x m matrix (y), result stored 
   in z. 

   The inner loop is a pair of vector dot products, each implemented 
   similarly to the matprod_vec_vec routine above.  As there, two 
   implementations are provided, one with loop unrolling within each
   dot product, the other without (since maybe the compiler does a
   better job of this).  The loop unrolling to do two dot products at
   one time is done manually in both implementations.

   Use -DALT_MATPROD_VEC_MAT to switch between these two implementations.
   Change #ifdef to #ifndef or vice versa below to change the default. */

void matprod_vec_mat (double *x, double *y, double *z, int k, int m)
{
    double s, s2, t;
    double *p, *y2;
    int i;

    /* If m is odd, compute the first element of the result (the dot product
       of x and the first column of y).  Adjust y, z, and m to account for 
       having handled the first column. */

    if (m & 1) {

#       ifdef ALT_MATPROD_VEC_MAT
        {
            s = 0.0;

            for (i = 0; i<k; i++)
                s += x[i] * y[i];

            y += k;
        }
#       else
        {
            double *e;

            p = x;

            /* Initialize sum to first product, if k odd; otherwise to 0. */

            if (k & 1)
                s = *p++ * *y++;
            else
                s = 0.0;

            /* Add two products each time around loop. */

            e = x+k;
            while (p < e) {
                s += *p++ * *y++;
                s += *p++ * *y++;
            }
        }
#       endif

        /* Store result of dot product as first element, decrement m. */

        *z++ = s;
        m -= 1;
    }

    /* In this loop, compute two consecutive elements of the result vector,
       by doing two dot products of x with columns of y.  Adjust y, z, and
       m as we go.  Note that m, the number of columns left to do, is even 
       when we start. */

    while (m > 0) {

        /* Set y and y2 to point to the starts of the two columns.  Note
           that y was modified in the previous dot product operation 
           so that it is now pointing at the next column to do. */

        y2 = y + k;

#       ifdef ALT_MATPROD_VEC_MAT
        {
            s2 = s = 0.0;

            /* Each time around this loop, add one product for each of the 
               two dot products. */

            for (i = 0; i<k; i++) {
                t = x[i];
                s += t * y[i];
                s2 += t * y2[i];
            }

            y = y2 + k;
        }
#       else
        {
            double *e;

            p = x;

            /* If the two dot products sum an odd number of products, set
               the sums, s and s2, to the first products here, and adjust p, 
               y, and y2.  Otherwise, initialize s and s2 to zero. */

            if (k & 1) {
                t = *p++;
                s = t * *y++;
                s2 = t * *y2++;
            }
            else
                s2 = s = 0.0;

            /* Each time around this loop, add two products for each of the 
               two dot products, adjusting p, y, y2, and i as we go.  Note
               that i will be even when we start. */

            e = x+k;
            while (p < e) {
                t = *p++;
                s += t * *y++;
                s2 += t * *y2++;
                t = *p++;
                s += t * *y++;
                s2 += t * *y2++;
            }

            y = y2;
        }
#       endif

        /* Store the two dot products in the result vector. */

        *z++ = s;
        *z++ = s2;

        m -= 2;
    }
}


/* Product of n x k matrix (x) and column vector of length k (y) with result 
   stored in z. 

   The product is computed using an outer loop that accumulates the sums for 
   all elements of the result vector, iterating over columns of x, in order
   to produce sequential accesses.  This loop is unrolled to accumulate from
   two columns of x at once, which probably reduces the number of memory
   accesses, and may give more potential to overlap accesses with computation.
   The result vector is initialized either to zeros or to the result from the 
   first column, if the number of columns is odd.  Order of summation is
   kept the same as the obvious method, for consistency of round-off errors.

   The case of n=2 may be handled specially, accumulating sums in two
   local variables rather than in the result vector, and then storing
   them in the result at the end.  Whether this is done can be controlled
   using -DALT_MATPROD_MAT_VEC.  Change #ifdef to #ifndef or vice versa below 
   to change the default. */

void matprod_mat_vec (double *x, double *y, double *z, int n, int k)
{
    double *p, *q;
    double b, b2;
    double *e, *f;
    int i;

    if (n <= 0) return;

#   ifndef ALT_MATPROD_MAT_VEC
    {
        if (n == 2) { 
        
            double s1, s2;

            /* Initialize s1 and s2 to zero, if k is even, or to the products
               of the first element of y with the first column of x.  Adjust 
               x, y, and k accordingly. */

            if (k & 1) {
                b = *y++;
                s1 = *x++ * b;
                s2 = *x++ * b;
                k -= 1;
            }
            else
                s1 = s2 = 0.0;

            /* Each time around this loop, add the products of two columns of
               x with two elements of y to s1 and s2.  Adjust x, y, and k to
               account for this.  Note that k will be even when we start. */

            while (k > 0) {
                b = *y++;
                b2 = *y++;
                s1 = (s1 + (x[0] * b)) + (x[2] * b2);
                s2 = (s2 + (x[1] * b)) + (x[3] * b2);
                x += 4;
                k -= 2;
            }

            /* Store s1 and s2 in the result vector. */

            z[0] = s1;
            z[1] = s2;

            return;
        }
    }
#   endif

    /* Initialize sums in z to zero, if k is even, or to the product of
       the first element of y with the first column of x.  Adjust x and y
       accordingly. */

    q = z;
    f = z+n;
    e = y+k;

    if (k & 1) {
        b = *y++;
        do { *q++ = *x++ * b; } while (q < f);
    }
    else {
        do { *q++ = 0.0; } while (q < f);
   }

    /* Each time around this loop, add the products of two columns of x 
       with two elements of y to the result vector, z.  Adjust x, y, and
       k to account for this.  Note that k will be even when we start. */

    while (y < e) {
        q = z;
        f = z+n;
        p = x+n;
        b = *y++;
        b2 = *y++;
        do { *q = (*q + (*x++ * b)) + (*p++ * b2); } while (++q < f);
        x = p;
    }
}


/* Product of an n x k matrix (x) and a k x m matrix (y) with result stored 
   in z. 

   The inner loop does two matrix-vector products each time, implemented 
   much as in matprod_mat_vec above, except for computing two columns. This
   gives a reasonably efficient implementation of an outer product (where
   k is one). Order of summation is kept the same as the obvious method, 
   for consistency of round-off errors.

   The case of n=2 may be handled specially, accumulating sums in two
   local variables rather than in a column of the result, and then storing
   them in the result column at the end.  Whether this is done can be 
   controlled using -DALT_MATPROD.  Change #ifdef to #ifndef or 
   vice versa below to change the default. */

void matprod (double *x, double *y, double *z, int n, int k, int m)
{
    if (n <= 0) return;

#   ifndef ALT_MATPROD

        if (n == 2) {
    
            /* If m is odd, compute the first column of the result, and
               update y, z, and m accordingly. */
    
            if (m & 1) {
    
                double s1, s2;
                double *r = x;    /* r set to x, and then modified */
                double *e = y+k;  /* stop when y reaches here */
    
                /* Initialize s1 and s2 to zero, if k is even, or to the 
                   products of the first element of the first column of y with
                   the first column of x.  Adjust x and y accordingly. */
    
                if (k & 1) {
                    double b = *y++;
                    s1 = *r++ * b;
                    s2 = *r++ * b;
                }
                else
                    s1 = s2 = 0.0;
    
                /* Each time around this loop, add the products of two columns
                   of x and two elements of the first column of y to s1 and s2.
                   Adjust x and y to account for this.  Note that e-y will 
                   be even when we start. */
    
                while (y < e) {
                    double b1, b2;
                    b1 = *y++;
                    b2 = *y++;
                    s1 = (s1 + (r[0] * b1)) + (r[2] * b2);
                    s2 = (s2 + (r[1] * b1)) + (r[3] * b2);
                    r += 4;
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
                double *e = y2;
                double *r = x;  /* r set to x, and then modified */
    
                /* Initialize sums for columns to zero, if k is even, or to the
                   products of the first elements of the next two columns of
                   y with the first column of x. Adjust x and y accordingly. */
    
                if (k & 1) {
                    double b = *y++;
                    double b2 = *y2++;
                    s11 = r[0] * b;
                    s12 = r[1] * b;
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
                    double b11 = *y++;
                    double b12 = *y++;
                    double b21 = *y2++;
                    double b22 = *y2++;
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

        double *r = x; /* r set to x, then modified as columns of x summed */
        double *e = y+k;

        /* Initialize sums in z to zero, if k is even, or to the product of
           the first element of the next column of y with the first column 
           of x (in which case adjust rand y accordingly). */

        if (k & 1) {
            double *q = z;
            double b = *y++;
            int i;
            for (i = n; i > 0; i--)
                *q++ = *r++ * b;
        }
        else {
            double *q = z;
            int i;
            for (i = n; i > 0; i--)
                *q++ = 0;
        }

        /* Each time around this loop, add the products of two columns of x 
           with two elements of the next column of y to the result vector, z.
           Adjust r and y to account for this.  Note that e-y will be even 
           when we start. */

        while (y < e) {
            double *q = z;
            double b = *y++;
            double b2 = *y++;
            int i;
            for (i = n; i > 0; i--) {
                *q = (*q + (*r * b)) + (*(r+n) * b2);
                r += 1;
                q += 1;
            }
            r += n;
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
        double *e = y2;
        double *r = x;  /* r set to x, then modified as columns of x summed */

        /* Initialize sums in next two columns of z to zero, if k is even, 
           or to the products of the first elements of the next two columns
           of y with the first column of x, if k is odd (in which case adjust 
           r and y accordingly). */

        if (k & 1) {
            double *q = z;
            double b = *y++;
            double b2 = *y2++;
            int i;
            for (i = n; i > 0; i--)
                *q++ = *r++ * b;
            r = x;
            for (i = n; i > 0; i--)
                *q++ = *r++ * b2;
        }
        else {
            double *q = z;
            int i;
            for (i = 2*n; i > 0; i--)
                *q++ = 0;
        }

        /* Each time around this loop, add the products of two columns of x 
           with two elements of the next two columns of y to the next two
           columns of the result vector, z.  Adjust r and y to account 
           for this.  Note that e-y will be even. */

        while (y < e) {
            double *q = z;
            double *f = z+n;
            double b11 = *y++;
            double b12 = *y++;
            double b21 = *y2++;
            double b22 = *y2++;
            do {
                double s1 = *r;
                double s2 = *(r+n);
                *q = (*q + (s1 * b11)) + (s2 * b12);
                *(q+n) = (*(q+n) + (s1 * b21)) + (s2 * b22);
                r += 1;
                q += 1;
            } while (q < f);
            r += n;
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
   matrix.  Only the lower-triangular part of the result is computed,
   with the upper-triangular part being copied from the lower triangle
   as columns of the result are produced.
*/

void matprod_trans1 (double *x, double *y, double *z, int n, int k, int m)
{
  int sym = x==y && n==m;  /* same operands, so symmetric result? */
  double *oz = z;          /* original value of z */
  int j = 0;               /* number of columns of result produced so far */

  if (n <= 0) return;

  /* Set result to zeros if k is zero. */

  if (k <= 0) {
      double *e = z + n*m;
      while (z < e) *z++ = 0;
      return;
  }

  /* If m is odd, compute the first column of the result, updating y, z, and 
     m to account for this column having been computed (so that the situation
     is the same as if m had been even to start with). */

  if (m & 1) {

      double *r = x;
      double *e = z+n;

      /* If n is odd, compute the first element of the first column of the
         result here.  Also, move r to point to the second column of x, and
         increment z. */

      if (n & 1) {
          double s = 0;
          double *q = y;
          double *e = y+k;
          do { s += *r++ * *q++; } while (q < e);
          *z++ = s;
      }

      /* Compute the remainder of the first column of the result two
         elements at a time (looking at two columns of x).  Note that 
         e-z will be even. */

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

      y += k;
      j += 1;
  }

  /* Compute two columns of the result each time around this loop, updating
     y, z, and j accordingly.  Note that m-j will be even. */

  while (j < m) {

      double *z2 = z+n;
      double *e = z2;
      double *r = x;

      /* If n is odd, compute the first elements of the two columns here,
         or copy them if they have already been computed from symmetry.
         Also, move r to point to the second column of x, and update z. */

      if (n & 1) {
          if (sym && j > 0) {
              *z++ = *(oz+j);
              *z2++ = *(oz+j+1);
              r += k;
          }
          else {
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
          }
      }

      /* For the symmetric case, copy elements to the remainder of the upper 
         part of these two columns. */
         
      if (sym && j > 0) {
          double *q = r==x ? oz+j : oz+j+n;
          while (q != z) {
              *z++ = *q;
              *z2++ = *(q+1);
              q += n;
              r += k;
          }
      }

      /* Compute the remainder of the two columns of the result, two elements
         at a time. */

      while (z < e) {
          double s00 = 0.0;
          double s01 = 0.0;
          double s10 = 0.0;
          double s11 = 0.0;
          double *q = y;
          int i = k;
          do {
              double t = *r;
              double t2 = *(r+k);
              double u = *q;
              double u2 = *(q+k);
              s00 += t * u;
              s01 += t * u2;
              s10 += t2 * u;
              s11 += t2 * u2;
              r += 1;
              q += 1;
          } while (--i > 0);
          *z++ = s00;
          *z2++ = s01;
          *z++ = s10;
          *z2++ = s11;
          r += k;
      }

      z = z2;
      y += 2*k;
      j += 2;
  }
}


/* Product of an n x k matrix (x) and the transpose of an m x k matrix (y) 
   with result stored in z.

   The case of n=2 may be handled specially, accumulating sums in two
   local variables rather than in a column of the result, and then storing
   them in the result column at the end.  Whether this is done can be 
   controlled using -DALT_MATPROD_TRANS2.  Change #ifdef to #ifndef or
   vice versa below to change the default. */

void matprod_trans2 (double *x, double *y, double *z, int n, int k, int m)
{
  int sym = x==y && n==m;  /* same operands, so symmetric result? */
  double *oz = z;          /* original value of z */
  int j = 0;               /* number of columns of result produced so far */

#   ifndef ALT_MATPROD_MAT_TRANS2

        if (n == 2) {

            int j = 0;
    
            /* If m is odd, compute the first column of the result, and
               update y, z, and mt accordingly. */
    
            if (m & 1) {
    
                double s1, s2;
                double *q = y;
                double *r = x;
                double *e = x+2*k;
    
                /* Initialize s1 and s2 to zero, if k is even, or to the 
                   products of the first element of the first row of y with
                   the first column of x.  Adjust r, q, and kt accordingly. */
    
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
                double *e = x+2*k;

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
    
                while (r < e) {
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

#   endif

    /* If m is odd, compute the first column of the result, updating y, z, and 
       j to account for this column having been computed. */

    if (m & 1) {

        double *q = y;
        double *r = x;
        int kt = k;
        int i;

        /* Initialize sums in z to zero, if k is even, or to the product of
           the first element of the first row of y with the first column 
           of x (in which case adjust r, q, and kt accordingly). */

        if (k & 1) {
            double *t = z;
            double b = *q;
            for (i = n; i > 0; i--)
                *t++ = *r++ * b;
            q += m;
            kt -= 1;
        }
        else {
            double *t = z;
            for (i = n; i > 0; i--)
                *t++ = 0;
        }

        /* Each time around this loop, add the products of two columns of x 
           with two elements of the first row of y to the result vector, z.
           Adjust r, y, and kt to account for this.  Note that kt will be even 
           when we start. */

        while (kt > 0) {
            double *t = z;
            double b1, b2;
            b1 = *q;
            q += m;
            b2 = *q;
            q += m;
            for (i = n; i > 0; i--) {
                *t = (*t + (*r * b1)) + (*(r+n) * b2);
                r += 1;
                t += 1;
            }
            r += n;
            kt -= 2;
        }

        /* Move to next column of the result and the next row of y. */

        z += n;
        y += 1;
        j += 1;
    }

    /* Compute two columns of the result each time around this loop, updating
       y, z, and j accordingly.  Note that m-j will be even. */

    while (j < m) {

        /* These set here for the non-symmetric case, modifed if symmetric */
        double *xs = x;  /* Where to start fetching for sums */
        double *zs = z;  /* Where to start storing sums */
        int u = n;       /* Number of elements to sum */

        double *q = y;
        int kt = k;
        int i;

        /* If result is known to be symmetric, fill in upper part of the
           next two columns from already computed elements (unless these
           are the first two columns).  Adjust xs, zs, and u so that later
           sums are for only elements after those filled in here. */

        if (sym && j > 0) {
            double *s = oz+j;
            double *t1 = z;
            double *t2 = t1 + n;
            for (i = 0; i < j; i++) {
                *t1++ = *s;
                *t2++ = *(s+1);
                s += n;
            }
            u = n-j;
            xs += j;
            zs += j;
        }

        /* Initialize sums in the next two columns of z to zero, if k is 
           even, or to the products of the first elements of the next two
           rows of y with the first column of x (in which case adjust r, 
           q, and kt accordingly). */

        double *t1 = zs;
        double *t2 = t1 + n;

        if (k & 1) {
            double b1 = *q;
            double b2 = *(q+1);
            double *r = xs;
            for (i = u; i > 0; i--) {
                double s = *r++;
                *t1++ = s * b1;
                *t2++ = s * b2;
            }
            xs += n;
            q += m;
            kt -= 1;
        }
        else {
            for (i = u; i > 0; i--) {
                *t1++ = 0;
                *t2++ = 0;
            }
        }

        /* Each time around this loop, add the products of two columns of x 
           with elements of the next two rows of y to the next two columns
           the result vector, z.  Adjust r, y, and kt to account for this.  
           Note that kt will be even. */

        while (kt > 0) {
            double b11, b12, b21, b22;
            double *t1 = zs;
            double *t2 = t1 + n;
            double *r = xs;
            b11 = *q;
            b21 = *(q+1);
            q += m;
            b12 = *q;
            b22 = *(q+1);
            q += m;
            for (i = u; i > 0; i--) {
                double s1 = *r;
                double s2 = *(r+n);
                *t1 = (*t1 + (s1 * b11)) + (s2 * b12);
                *t2 = (*t2 + (s1 * b21)) + (s2 * b22);
                t1 += 1;
                t2 += 1;
                r += 1;
            }
            xs += 2*n;
            kt -= 2;
        }

        /* Move forward two to the next column of the result and the
           next row of y. */

        z += 2*n;
        y += 2;
        j += 2;
    }
}
