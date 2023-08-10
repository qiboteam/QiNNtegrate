# Numeric integration of arbitrary functions using quantum machine learning

This project implements the ideas of [2211.02834](https://arxiv.org/abs/2211.02834)
in quantum circuits exploting the implementation of the [VQE](https://qibo.science/qibo/stable/code-examples/advancedexamples.html#how-to-write-a-vqe)
and the [Parameter Shift Rule](https://qibo.science/tutorials/parameter_shift_rule) available in [qibo](qibo.science)

An example command to run the code would be:

```python
python main.py --parameters 1 2 3  --optimizer lbfgs --absolute --layers 2 --nqubits 2 -j 18 --ndim 4 --target toy --ansatz goodscaling --maxiter 200 -o output_folder
```
