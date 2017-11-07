#!/bin/sh

cat colours - | xgraph -zw 0 -lf 9x15 -tf 10x20 -nl -P "$@"
