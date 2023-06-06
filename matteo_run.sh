#!/bin/sh

python main.py --optimizer lbfgs --npoints 100 --layers 4 --absolute --nqubits 1 --ndim 2 --target uquark2d --ansatz qpdf2q -j 8
  
