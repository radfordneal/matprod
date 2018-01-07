# Makefile for Foster processors running linux

CC=gcc  # version 4.8.4, in Lubuntu 14.04

CFLAGS=$(CFLAGSi32)

ALIGN=-DALIGN=16
ALOFF=-DALIGN_OFFSET=8

matprod-extra-progs=matprod-nosimd-test matprod-off-nosimd-test \
                    matprod-noalign-nosimd-test matprod-off-noalign-nosimd-test

include ../make-common
