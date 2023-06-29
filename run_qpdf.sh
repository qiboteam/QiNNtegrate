#!/bin/sh

python main.py --load uquark1d/best_p.npy --layers 4 --nqubits 5 --ndim 1\
 --ansatz qpdf_iqm5q --maxiter 0 --target uquark --nshots 1000 --nruns 2\
 -o uquark1d
