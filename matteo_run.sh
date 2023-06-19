#!/bin/sh

python main.py --optimizer lbfgs --npoints 50 --layers 1 --absolute --nqubits 1 --ndim 1 --nshots 10000 --target cosnd --ansatz reuploading -j 8
  
