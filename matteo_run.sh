#!/bin/sh

python main.py --optimizer basinhopping --npoints 100 --layers 1 --absolute --nshots 10000 --nqubits 3 --ndim 3 --target cosnd
  