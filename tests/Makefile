# MATPROD - A LIBRARY FOR MATRIX MULTIPLICATION WITH OPTIONAL PIPELINING
#           Makefile for Compiling Test Programs
#
# Copyright (c) 2013, 2017 Radford M. Neal.
# 
#   The matprod library is free software; you can redistribute it and/or modify
#   it under the terms of the GNU General Public License as published by
#   the Free Software Foundation; either version 2 of the License, or
#   (at your option) any later version.
# 
#   This program is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU General Public License for more details.
# 
#   You should have received a copy of the GNU General Public License along
#   with this program; if not, write to the Free Software Foundation, Inc.,
#   51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

progs=matprod-test matprod-off-test matprod-noalign-test \
      matprod-nosimd-test matprod-off-nosimd-test matprod-noavx-test \
      piped-matprod-test piped-matprod-noalign-test \
      piped-matprod-nohelpers-test
      # matprod-norestrict-test and piped-matprod-norestrict-test omitted

blas-progs=blas0-test blas1-test blas2-test blas3-test

CFLAGSgen=-g -std=gnu99 -O3 -ffp-contract=off # Generic processor
CFLAGSi32=-g -std=gnu99 -O3 -ffp-contract=off  -msse2 -mfpmath=sse
CFLAGSi64=-g -std=gnu99 -O3 -ffp-contract=off  -msse3 -mtune=core2
CFLAGSavx=-g -std=gnu99 -O3 -ffp-contract=off  -mavx  -mtune=ivybridge
CFLAGSavx2=-g -std=gnu99 -O3 -ffp-contract=off -mavx2 -mtune=skylake
CFLAGSerr=-Xneed-to-set-CFLAGS

CC=gcc-7               # Compiler to be used, must be gcc look-alike for CFLAGS

CFLAGS=$(CFLAGSerr)    # Set to one of the above as appropriate

ALIGN=-DALIGN=32       # Can also test with other settings
ALOFF=-DALIGN_OFFSET=8


all:	$(progs)
	true

blas:	$(blas-progs)
	true

# -----------------------------------------------------------------------------

matprod-test:	test.h test.c matprod-test.c ../matprod.h ../matprod.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) test.c matprod-test.c ../matprod.c \
		-o matprod-test

matprod-off-test:	test.h test.c matprod-test.c ../matprod.h ../matprod.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) $(ALOFF) \
		test.c matprod-test.c ../matprod.c \
		-o matprod-off-test

matprod-noalign-test:	test.h test.c matprod-test.c ../matprod.h ../matprod.c
	$(CC) -I.. -I. $(CFLAGS) test.c matprod-test.c ../matprod.c \
		-o matprod-noalign-test

matprod-nosimd-test:	test.h test.c matprod-test.c ../matprod.h ../matprod.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) -DDISABLE_SIMD_CODE \
		test.c matprod-test.c ../matprod.c \
		-o matprod-nosimd-test

matprod-off-nosimd-test:	test.h test.c matprod-test.c ../matprod.h ../matprod.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) $(ALOFF) -DDISABLE_SIMD_CODE \
		test.c matprod-test.c ../matprod.c \
		-o matprod-off-nosimd-test

matprod-norestrict-test:	test.h test.c matprod-test.c ../matprod.h ../matprod.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) -DMATPROD_NO_RESTRICT \
		test.c matprod-test.c ../matprod.c \
		-o matprod-norestrict-test

matprod-noavx-test:	test.h test.c matprod-test.c ../matprod.h ../matprod.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) -DDISABLE_AVX_CODE \
		test.c matprod-test.c ../matprod.c \
		-o matprod-noavx-test

# -----------------------------------------------------------------------------

piped-matprod-test:	test.h test.c piped-matprod-test.c ../piped-matprod.h \
			../piped-matprod.c helpers-app.h helpers.h helpers.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) -fopenmp \
		test.c piped-matprod-test.c ../piped-matprod.c helpers.c \
		-o piped-matprod-test

piped-matprod-noalign-test:	test.h test.c piped-matprod-test.c ../piped-matprod.h \
			../piped-matprod.c helpers-app.h helpers.h helpers.c
	$(CC) -I.. -I. $(CFLAGS) -fopenmp \
		test.c piped-matprod-test.c ../piped-matprod.c helpers.c \
		-o piped-matprod-noalign-test

piped-matprod-norestrict-test:	test.h test.c piped-matprod-test.c ../piped-matprod.h \
			../piped-matprod.c helpers-app.h helpers.h helpers.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) -fopenmp -DMATPROD_NO_RESTRICT \
		test.c piped-matprod-test.c ../piped-matprod.c helpers.c \
		-o piped-matprod-norestrict-test

piped-matprod-nohelpers-test:	test.h test.c piped-matprod-test.c ../piped-matprod.h \
			../piped-matprod.c helpers-app.h helpers.h helpers.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) -fopenmp -DHELPERS_DISABLED \
		test.c piped-matprod-test.c ../piped-matprod.c helpers.c \
		-o piped-matprod-nohelpers-test

# -----------------------------------------------------------------------------

blas0-test:		test.h test.c blas-test.c \
			ddot.f dgemm.f dgemv.f dsyrk.f lsame.f
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) \
		test.c blas-test.c ddot.f dgemm.f dgemv.f dsyrk.f lsame.f \
		-o blas0-test

blas1-test:		test.h test.c blas-test.c blas.f
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) test.c blas-test.c blas.f \
		-lm -o blas1-test

blas2-test:		test.h test.c blas-test.c blas.f
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) test.c blas-test.c blas2.f \
		-lm -o blas2-test

blas3-test:		test.h test.c blas-test.c
	$(CC) -I.. -I. $(CFLAGS) $(ALIGN) test.c blas-test.c \
		/usr/lib/libblas.so.3 \
		-lm -o blas3-test

clean:
	rm -f $(progs) $(blas-progs)
