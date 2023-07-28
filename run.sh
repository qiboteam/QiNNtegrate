#!/bin/sh

python main.py --layers 4 --nqubits 1 --ndim 1\
 --ansatz qpdf --maxiter 200 --target uquark  --nruns 1\
 -o test_function -j 10 --optimizer cma 

