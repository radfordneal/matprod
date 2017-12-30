# Makefile for Harpertown processors running OS X El Capitan.

CC=gcc-7

CFLAGS=$(CFLAGSi64)

ALIGN=-DALIGN=16
ALOFF=-DALIGN_OFFSET=8

matprod-extra-progs?=matprod-nosimd-test matprod-off-nosimd-test \
                     matprod-noalign-nosimd-test matprod-off-noalign-nosimd-test

include ../make-common
