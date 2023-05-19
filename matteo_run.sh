#!/bin/sh

python main.py --optimizer basinhopping --npoints 50 --layers 3 --absolute --nqubits 2 --ndim 2 --target uquark2d --ansatz qpdf2q -j 10
  
