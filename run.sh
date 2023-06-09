#!/bin/sh

python main.py --optimizer lbfgs --absolute --npoints 100 --layers 2 --nqubits 2 --ndim 4 --target cosnd --ansatz goodscaling -j 16
