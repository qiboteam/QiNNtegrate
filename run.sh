#!/bin/sh

python main.py --load plotscripts/uquark_alpha0.92/best_p.npy --layers 4 --nqubits 1 --ndim 1\
 --ansatz qpdf --maxiter 0 --target uquark  --nruns 1\
 -o uquark1d -j 1

