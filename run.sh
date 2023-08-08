#!/bin/sh

python main.py --optimizer lbfgs --absolute --npoints 50 --layers 1 --maxiter 100 --nqubits 1 --ndim 1 --target hardware --ansatz goodscaling -j 16
