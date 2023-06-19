#!/bin/sh

python main.py --optimizer cma --npoints 50 --layers 1 --absolute --nqubits 1 --ndim 1 --nshots 1000 --target sin1d --ansatz deepup -j 8
  
