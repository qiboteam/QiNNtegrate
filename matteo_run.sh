#!/bin/sh

python main.py --optimizer basinhopping --npoints 100 --layers 3 --absolute --nqubits 2 --ndim 2 --target uquark --ansatz qpdf2q
  