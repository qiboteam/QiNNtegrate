# Numeric integration of arbitrary functions using quantum machine learning

This project implements the ideas of [2211.02834](https://arxiv.org/abs/2211.02834)
in quantum circuits exploting the implementation of the [VQE](https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-write-a-vqe)
and the [Parameter Shift Rule](https://qibo.science/tutorials/parameter_shift_rule) available in [qibo](qibo.science)

The code is based on the `main.py` script and a list of utilities in the `qinntegrate` folder.
It is not a library but rather a collection of scripts that uses the [Qibo](https://github.com/qiboteam/qibo) library underneath which does all the heavy (quantum) lifting.

```bash
  python main.py --help
```

```
usage: main.py [-h] [--xmin XMIN [XMIN ...]] [--xmax XMAX [XMAX ...]] [-o OUTPUT] [-l LOAD] [-j JOBS] [--target TARGET] [--parameters PARAMETERS [PARAMETERS ...]] [--ndim NDIM] [--ansatz ANSATZ]
               [--nqubits NQUBITS] [--layers LAYERS] [--nshots NSHOTS] [--pdf_alpha PDF_ALPHA] [--maxiter MAXITER] [--npoints NPOINTS] [--padding] [--absolute] [--optimizer OPTIMIZER] [--nruns NRUNS]

options:
  -h, --help            show this help message and exit
  --xmin XMIN [XMIN ...]
                        Integration limit xi
  --xmax XMAX [XMAX ...]
                        Integration limit xf
  -o OUTPUT, --output OUTPUT
                        Output folder
  -l LOAD, --load LOAD  Load initial parameters from
  -j JOBS, --jobs JOBS  Number of processes to utilize (default 4)

Target function:
  --target TARGET       Select target function, available: ['sin1d', 'cosnd', 'sind', 'lepage', 'uquark', 'uquark2d', 'cosndalpha', 'toy']
  --parameters PARAMETERS [PARAMETERS ...]
                        List of parameters for the target functions
  --ndim NDIM           Number of dimensions

Circuit definition:
  --ansatz ANSATZ       Circuit ansatz, please choose one among ['base', 'reuploading', 'deepup', 'verticup', 'qpdf', 'qpdf2q', 'goodscaling']
  --nqubits NQUBITS     Number of qubits for the VQE
  --layers LAYERS       Number of layers for the VQE
  --nshots NSHOTS       Number of shots for each < Z > evaluation
  --pdf_alpha PDF_ALPHA
                        (only value for PDF ansatzs) value of alpha in the PDF prefactor

Optimization definition:
  --maxiter MAXITER     Maximum number of iterations (default 1000)
  --npoints NPOINTS     Training points (default 500)
  --padding             Train the function beyond the integration limits
  --absolute            Don't normalize MSE by the size of the integrand
  --optimizer OPTIMIZER
                        Optimizers, available options: ['cma', 'bfgs', 'sgd', 'lbfgs', 'annealing', 'basinhopping']
  --nruns NRUNS         Number of times the optimization is repeated

```

Some example commands:

```bash
python main.py --parameters 1 2 3  --optimizer lbfgs --absolute --layers 2 --nqubits 2 -j 18 --ndim 4 --target toy --ansatz goodscaling --maxiter 200 -o output_folder
python main.py --optimizer lbfgs --absolute --npoints 100 --layers 2 --nqubits 2 --ndim 4 --target cosnd --ansatz goodscaling -j 16
```
