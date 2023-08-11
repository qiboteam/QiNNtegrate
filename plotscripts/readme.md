The scripts in this folder can be used to reproduced some of the plots in arxiv/2308.05657

### Toy model

Select a target value for `alpha` and run the following:
```sh
python plot_toy.py --data_folder toytarget/ --xmax 3 --error --alpha 1.25 --npoints 16
```

### `uquark2d` plot

```sh
python plot_uquark2d.py --nshots_predictions 1000000 --n_predictions 100 --n_points 20 --data_folder uquark2d_0.001to0.7_lbfgs/
```


### `uquark` plot

```sh
python main.py --load plotscripts/uquark_alpha0.92/best_p.npy --layers 4 --nqubits 1 --ndim 1 --ansatz qpdf --maxiter 0 --target uquark -o uquark1d -j 1 --pdf_alpha 0.92
```
